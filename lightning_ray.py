from __future__ import annotations

import numpy as np
import os
import pynvml
import pytorch_lightning as pl
import random
import ray
import tempfile
import torch

from argparse import ArgumentParser, Namespace
from collections import defaultdict
from copy import deepcopy
from inspect import signature
from pathlib import Path
from pytorch_lightning.accelerators.mps import MPSAccelerator
from ray.air import CheckpointConfig, RunConfig, ScalingConfig
from ray.experimental.tqdm_ray import safe_print
from ray.train import Checkpoint, Result
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment
from ray.train.torch import TorchTrainer
from ray.train.trainer import BaseTrainer
from ray.tune import ExperimentAnalysis, ResultGrid, Trainable, TuneConfig, Tuner
from ray.tune.execution.experiment_state import _ResumeConfig
from ray.tune.execution.tune_controller import TuneController
from ray.tune.experiment import Trial
from ray.tune.schedulers import FIFOScheduler, TrialScheduler
from ray.tune.search import SearchAlgorithm, Searcher
from ray.tune.tuner import TunerInternal
from tensorboard import program as tensorboard
from typing import Any, Callable, Iterable, Literal, Type



def variable_kwargs_fn_wrapper(fn: Callable):
    """
    Function wrapper that supports variable keyword arguments.

    Parameters
    ----------
    fn : Callable
        Function to wrap
    """
    def wrapper_fn(*args, **kwargs):
        parameters = signature(fn).parameters.values()
        if any(p.kind == p.VAR_KEYWORD for p in parameters):
            return fn(*args, **kwargs)
        else:
            kwargs = {
                key: value for key, value in kwargs.items()
                if key in signature(fn).parameters.keys()
            }
            return fn(*args, **kwargs)

    return wrapper_fn

def parse_args_dynamic(parser: ArgumentParser) -> tuple[Namespace, dict[str, Any]]:
    """
    Parse command-line arguments, and dynamically add any extra arguments
    to a Ray-style configuration dictionary.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Argument parser

    Returns
    -------
    args : argparse.Namespace
        Parsed arguments
    config : dict[str, Any]
        Ray-style configuration dictionary of any extra arguments

    Example
    -------
    Python:
    ```
    parser.add_argument('--foo')
    args, config = parse_args_dynamic(parser)
    print('args:', args)
    print('config:', config)
    ```

    Command Line:
    ```
    $ python main.py --foo bar --other 1 --thing a b c
    args: Namespace(foo='bar')
    config: {'other': 1, 'thing': {'grid_search': ['a', 'b', 'c']}}
    ```
    """
    args, extra_args_names = parser.parse_known_args()

    def infer_type(s: str):
        try:
            s = float(s)
            if s // 1 == s:
                return int(s)
            return s
        except ValueError:
            return s

    # For each extra argument name, add it to the parser
    for name in extra_args_names:
        if name.startswith(('-', '--')):
            parser.add_argument(name.split('=', 1)[0], type=infer_type, nargs='+')

    # Create Ray config dictionary from extra arguments
    config = {
        key: values[0] if len(values) == 1 else ray.tune.grid_search(values)
        for key, values in vars(parser.parse_args()).items()
        if key not in vars(args)
    }

    return args, config

def configure_gpus(gpu_memory_per_worker: str | int | float) -> float:
    """
    Configure CUDA_VISIBLE_DEVICES to maximize the total number of workers.

    Parameters
    ----------
    gpu_memory_per_worker : str | int | float
        The amount of GPU memory required per worker. Can be a string with units
        (e.g. "1.5 GB") or a number in bytes.

    Returns
    -------
    gpus_per_worker : float
        The number of GPUs to allocate to each worker
    """
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError_LibraryNotFound:
        return 0

    if not torch.cuda.is_available():
        return 0

    # Convert to bytes
    if isinstance(gpu_memory_per_worker, str):
        UNITS = {
            'KB': 1e3, 'KiB': 2**10,
            'MB': 1e6, 'MiB': 2**20,
            'GB': 1e9, 'GiB': 2**30,
            'TB': 1e12, 'TiB': 2**40,
        }
        for unit, value in UNITS.items():
            if gpu_memory_per_worker.endswith(unit):
                gpu_memory_per_worker = float(gpu_memory_per_worker.strip(unit)) * value
                break

    gpu_memory_per_worker = int(gpu_memory_per_worker)
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if devices == ['']:
        devices = range(pynvml.nvmlDeviceGetCount())

    # For each GPU, calculate the number of workers that can fit in memory
    total_num_gpus = pynvml.nvmlDeviceGetCount()
    num_workers = np.zeros(total_num_gpus)
    for i in devices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(i))
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        num_workers[int(i)] = memory_info.free // gpu_memory_per_worker

    if num_workers.max() == 0:
        return 0

    # Sort GPU indices by number of available workers
    gpu_idx = num_workers.argsort()[::-1]

    # Given n GPUs, the total number of workers is
    # n * the number of workers on the GPU with the least availability
    total_num_workers = np.zeros(total_num_gpus + 1)
    for n in range(1, total_num_gpus + 1):
        idx = gpu_idx[:n] # select the top-n GPUs
        total_num_workers[n] = n * num_workers[idx].min()

    # Select the combination of GPUs that maximizes the total number of workers
    n = total_num_workers.argmax()
    best_gpu_idx = gpu_idx[:n]
    gpus_per_worker = 1 / num_workers[best_gpu_idx].min()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, best_gpu_idx))

    return gpus_per_worker



### Ray Tune Tuner

def get_internal_tuner(tuner: Tuner | LightningTuner) -> TunerInternal:
    """
    Get the internal Ray Tune tuner.

    Parameters
    ----------
    tuner : ray.tune.Tuner or LightningTuner
        Tuner instance
    """
    if isinstance(tuner, LightningTuner):
        tuner = tuner.tuner

    return tuner._local_tuner or tuner._remote_tuner

def restore_ray_tuner(
    path: str | Path,
    trainable: str | Callable | Type[Trainable] | BaseTrainer,
    **kwargs,
) -> Tuner:
    """
    Restore Ray Tuner.

    NOTE: This is a patch to prevent the experiment name and storage path
    from being changed when restoring a tuner. This is necessary because,
    as of Ray 2.7.1, the Tuner.restore() implementation does not support experiment
    names specified as subdirectories (i.e. experiment_name = 'exp/train').

    Parameters
    ----------
    path : str or Path
        Experiment directory
    trainable : str or callable or type[Trainable] or BaseTrainer
        Trainable to be tuned
    """

    class RestoredTunerInternal(TunerInternal):
        """
        Ray TunerInternal class that wraps the RunConfig to prevent
        the experiment name and storage path from being changed on restore.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if isinstance(self._run_config, RestoredRunConfig):
                self._run_config = self._run_config.run_config  # unwrap after init

        def _set_trainable_on_restore(self, *args, **kwargs):
            super()._set_trainable_on_restore(*args, **kwargs)
            self._run_config = RestoredRunConfig(self._run_config)

    class RestoredRunConfig(RunConfig):
        """
        RunConfig wrapper that prevents the experiment name and storage
        path from being changed.
        """

        def __init__(self, run_config: RunConfig):
            self.run_config = run_config

        def __getattribute__(self, name: str) -> Any:
            if name == 'run_config':
                return super().__getattribute__(name)
            else:
                return getattr(self.run_config, name)

        def __setattr__(self, name: str, value: Any):
            if name in ('name', 'storage_path'):
                print(f"{self} cannot set attribute '{name}'")
            else:
                super().__setattr__(name, value)

    if not ray.util.client.ray.is_connected():
        resume_config = _ResumeConfig(
            resume_unfinished=kwargs.get('resume_unfinished', True),
            resume_errored=kwargs.get('resume_errored', False),
            restart_errored=kwargs.get('restart_errored', False),
        )
        tuner_internal = RestoredTunerInternal(
            restore_path=str(path),
            resume_config=resume_config,
            trainable=trainable,
            param_space=kwargs.get('param_space', None),
            storage_filesystem=kwargs.get('storage_filesystem', None),
        )
        return Tuner(_tuner_internal=tuner_internal)

    return Tuner.restore(path, trainable, **kwargs)



### Ray Tune Results

def filter_results(
    filter_fn: Callable[[Result], bool],
    results: ResultGrid) -> ResultGrid:
    """
    Filter results by a given function.

    Parameters
    ----------
    filter_fn : Callable(Result) -> bool
        A function that takes a `Result` object and
        returns a boolean indicating whether to keep the result or not
    results : ResultGrid
        Results to filter
    """
    return ResultGrid(
        ExperimentAnalysis(
            results._experiment_analysis.experiment_path,
            storage_filesystem=results._experiment_analysis._fs,
            trials=list(filter(filter_fn, results._experiment_analysis.trials)),
            default_metric=results._experiment_analysis.default_metric,
            default_mode=results._experiment_analysis.default_mode,
        )
    )

def group_results(
    results: ResultGrid,
    groupby: str | Iterable[str],
) -> dict[str | tuple[str], ResultGrid]:
    """
    Map each unique combination of config values for keys specified by `groupby`
    to a ResultGrid containing only the results with those config values.

    Parameters
    ----------
    results : ResultGrid
        Results to group
    groupby : str or Iterable[str]
        Config key(s) to group by
    """
    trials = defaultdict(list)
    for trial in results._experiment_analysis.trials:
        config = trial.config.get('train_loop_config', trial.config)
        if isinstance(groupby, str):
            group = config[groupby]
            trials[group].append(trial)
        else:
            group = tuple(config[key] for key in groupby)
            trials[group].append(trial)

    return {
        group: ResultGrid(
            ExperimentAnalysis(
                results._experiment_analysis.experiment_path,
                storage_filesystem=results._experiment_analysis._fs,
                trials=trials[group],
                default_metric=results._experiment_analysis.default_metric,
                default_mode=results._experiment_analysis.default_mode,
            )
        )
        for group in trials
    }



### Ray Tune Scheduling

class GroupScheduler(TrialScheduler):
    """
    Group trials by config values and apply a different scheduler instance to each group.
    """

    def __init__(self, scheduler: TrialScheduler, groupby: Iterable[str]):
        """
        Parameters
        ----------
        scheduler : TrialScheduler
            Scheduler to apply to each group
        groupby : Iterable[str]
            Trial config keys to group by
        """
        super().__init__()
        self.groupby = tuple(groupby)
        self.base_scheduler = scheduler
        self.schedulers = defaultdict(self._create_scheduler)

    def set_search_properties(self, *args, **kwargs) -> bool:
        """
        Pass search properties to scheduler.
        """
        self.base_scheduler.set_search_properties(*args, **kwargs)
        for scheduler in self.schedulers.values():
            scheduler.set_search_properties(*args, **kwargs)

        return super().set_search_properties(*args, **kwargs)

    def on_trial_add(self, tune_controller: TuneController, trial: Trial):
        """
        Called when a new trial is added to the trial runner.
        """
        key = self._get_trial_key(trial)
        self.schedulers[key].on_trial_add(tune_controller, trial)

    def on_trial_error(self, tune_controller: TuneController, trial: Trial):
        """
        Notification for the error of trial.
        """
        key = self._get_trial_key(trial)
        self.schedulers[key].on_trial_error(tune_controller, trial)

    def on_trial_result(
        self, tune_controller: TuneController, trial: Trial, result: dict,
    ) -> str:
        """
        Called on each intermediate result returned by a trial.
        """
        key = self._get_trial_key(trial)
        return self.schedulers[key].on_trial_result(tune_controller, trial, result)

    def on_trial_complete(
        self, tune_controller: TuneController, trial: Trial, result: dict,
    ):
        """
        Notification for the completion of trial.
        """
        key = self._get_trial_key(trial)
        self.schedulers[key].on_trial_complete(tune_controller, trial, result)

    def on_trial_remove(self, tune_controller: TuneController, trial: Trial):
        """
        Called to remove trial.
        """
        key = self._get_trial_key(trial)
        self.schedulers[key].on_trial_remove(tune_controller, trial)

    def choose_trial_to_run(self, tune_controller: TuneController) -> Trial | None:
        """
        Called to choose a new trial to run.
        """
        schedulers = list(self.schedulers.values())
        random.shuffle(schedulers)

        for scheduler in schedulers:
            trial = scheduler.choose_trial_to_run(tune_controller)
            if trial is not None:
                return trial

        return None

    def debug_string(self) -> str:
        """
        Returns a human readable message for printing to the console.
        """
        return '\n'.join([
            f'{key}: {scheduler.debug_string()}'
            for key, scheduler in self.schedulers.items()
        ])

    def _create_scheduler(self) -> TrialScheduler:
        """
        Create a new scheduler instance.
        """
        return deepcopy(self.base_scheduler)

    def _get_trial_key(self, trial: Trial) -> tuple:
        """
        Get the group key for the specified trial.
        """
        config = trial.config.get('train_loop_config', trial.config)
        return tuple(config.get(key, None) for key in self.groupby)



### Ray Train Metrics & Checkpoints

class RayCallback(pl.Callback):
    """
    Callback class for using Ray Tune with PyTorch Lightning.
    """

    def __init__(
        self,
        checkpoint_frequency: int = 1,
        checkpoint_at_end: bool = True,
        **kwargs):
        """
        Parameters
        ----------
        checkpoint_frequency : int
            Frequency of checkpoints (i.e. every N epochs)
        checkpoint_at_end : bool
            Whether to checkpoint at the end of training
        """
        super().__init__()
        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint_at_end = checkpoint_at_end
        self.metrics = {}

    def create_checkpoint(
        self,
        trainer: pl.Trainer,
        checkpoint_dir: Path | str,
    ) -> Checkpoint:
        """
        Create a checkpoint.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning trainer
        checkpoint_dir : Path or str
            Directory to save checkpoint to
        """
        checkpoint_path = Path(checkpoint_dir, 'checkpoint.pt')
        trainer.save_checkpoint(checkpoint_path)
        return Checkpoint.from_directory(checkpoint_dir)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Clear metrics when a train epoch starts.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning trainer
        pl_module : pl.LightningModule
            PyTorch Lightning module
        """
        self.metrics.clear()
        torch.cuda.empty_cache()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Report metrics when a train epoch ends.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning trainer
        pl_module : pl.LightningModule
            PyTorch Lightning module
        """
        # Update metrics
        self.metrics.update({k: v.item() for k, v in trainer.callback_metrics.items()})
        self.metrics['epoch'] = self.metrics['step'] = trainer.current_epoch

        # Report metrics
        checkpoint = None
        should_checkpoint = (trainer.current_epoch + 1) % self.checkpoint_frequency == 0
        if ray.train.get_context().get_local_rank() == 0:
            with tempfile.TemporaryDirectory() as temp_dir:
                if should_checkpoint:
                    checkpoint = self.create_checkpoint(trainer, temp_dir)

                ray.train.report(metrics=self.metrics, checkpoint=checkpoint)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Save final checkpoint when training ends.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning trainer
        pl_module : pl.LightningModule
            PyTorch Lightning module
        """
        if self.checkpoint_at_end:
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint = self.create_checkpoint(trainer, temp_dir)
                ray.train.report(metrics=self.metrics, checkpoint=checkpoint)



### Lightning Tuner

class LightningTuner:
    """
    Use Ray Tune to train multiple PyTorch Lightning models in parallel.

    TODO:
    * Implement GroupSearcher
    * Validate distributed training of single models

    Example
    -------
    ```
    >>> class MyModel(pl.LightningModule):
            def __init__(self, hidden_dim, lr):
                ...

    >>> class MyDataModule(pl.LightningDataModule):
            def __init__(self, batch_size):
                ...

    >>> param_space = {
            'batch_size': 64,
            'lr': ray.tune.loguniform(1e-5, 1e-2),
            'hidden_dim': ray.tune.grid_search([32, 64, 128]),
        }
    >>> tuner = MyTuner(metric='val_acc', mode='max', num_samples=5)
    >>> tuner.fit(MyModel, MyDataModule, param_space=param_space)
    ```
    """

    def __init__(
        self,
        metric: str | None = None,
        mode: Literal['min', 'max'] = 'max',
        search_alg: Searcher | SearchAlgorithm | None = None,
        scheduler: TrialScheduler | None = None,
        num_samples: int = 1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        metric : str
            Metric to optimize
        mode : one of {'min', 'max'}
            Whether to minimize or maximize the metric
        search_alg : Searcher or SearchAlgorithm
            Ray search algorithm for optimization (see `ray.tune.search`)
        scheduler : TrialScheduler
            Ray trial scheduler for executing the experiment (see `ray.tune.schedulers`)
        num_samples : int
            Number of times to sample from the hyperparameter space
            (if `grid_search` is provided as an argument,
            the grid will be repeated `num_samples` of times)
        kwargs : Any
            Additional arguments to pass to `ray.tune.TuneConfig()`
        """
        self.tuner = None
        self.tune_config = TuneConfig(
            metric=metric,
            mode=mode,
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=num_samples,
            **kwargs,
        )

    def fit_single_model(
        self,
        config: dict[str, Any],
        model_creator: Callable[..., pl.LightningModule],
        datamodule_creator: Callable[..., pl.LightningDataModule],
        callbacks_creator: Callable[..., list[pl.Callback]] = lambda **kwargs: [],
        num_workers: int = 1,
    ):
        """
        Train a single model with the given configuration.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary
        model_creator : Callable(**config) -> pl.LightningModule
            Model creator
        datamodule_creator : Callable(**config) -> pl.LightningDataModule
            Data module creator
        callbacks_creator : Callable(**config) -> list[pl.Callback]
            Callbacks creator
        """
        model_creator = variable_kwargs_fn_wrapper(model_creator)
        datamodule_creator = variable_kwargs_fn_wrapper(datamodule_creator)
        callbacks_creator = variable_kwargs_fn_wrapper(callbacks_creator)
        trainer = variable_kwargs_fn_wrapper(pl.Trainer)(
            accelerator='cpu' if MPSAccelerator.is_available() else 'auto',
            strategy=RayDDPStrategy(find_unused_parameters=True),
            devices='auto',
            num_nodes=num_workers,
            logger=False, # logging metrics is handled by RayCallback
            callbacks=[*callbacks_creator(**config), RayCallback(**config)],
            enable_checkpointing=False, # checkpointing is handled by RayCallback
            enable_progress_bar=False,
            plugins=[RayLightningEnvironment()],
            **config,
        )
        trainer.fit(model_creator(**config), datamodule_creator(**config))

    def fit(
        self,
        model_creator: Callable[..., pl.LightningModule],
        datamodule_creator: Callable[..., pl.LightningDataModule],
        callbacks_creator: Callable[..., list[pl.Callback]] = lambda **config: [],
        param_space: dict[str, Any] = {},
        save_dir: Path | str = './results',
        experiment_name: str | None = None,
        num_workers_per_trial: int = 1,
        num_cpus_per_worker: int = 1,
        num_gpus_per_worker: int = 1 if torch.cuda.is_available() else 0,
        gpu_memory_per_worker: str | int | float | None = None,
        tensorboard_port: int = 0,
        groupby: str | Iterable[str] = (),
    ) -> ResultGrid:
        """
        Train PyTorch Lightning models using Ray Tune.

        The `param_space` dictionary specifies the hyperparameter search space over
        model, data module, and callback creator keyword arguments.

        Hyperparameters that can be specified in `param_space` include:
            * Model creator keyword arguments
            * Data module creator keyword arguments
            * Callbacks creator keyword arguments
            * PyTorch Lightning `Trainer` keyword arguments
            * 'checkpoint_frequency' (frequency of checkpoints, in epochs)

        Parameters
        ----------
        model_creator : Callable(**config) -> pl.LightningModule
            Model creator
        datamodule_creator : Callable(**config) -> pl.LightningDataModule
            Data module creator
        callbacks_creator : Callable(**config) -> list[pl.Callback]
            Callbacks creator
        param_space : dict[str, Any]
            Ray Tune hyperparameter search space
            (see https://docs.ray.io/en/latest/tune/api/search_space.html)
        save_dir : Path or str
            Directory to save results to
        experiment_name : str or None
            Name of the experiment (generated automatically if not provided)
        num_workers_per_trial : int
            Number of workers per trial
        num_cpus_per_worker : int
            Number of CPUs per worker
        num_gpus_per_worker : int
            Number of GPUs per worker
        gpu_memory_per_worker : str or int or float
            Amount of GPU memory to allocate to each worker
            (overrides `num_gpus_per_worker`)
        tensorboard_port : int
            Port for TensorBoard to visualize results
        groupby : str or Iterable[str]
            Config key(s) to group by
        """
        # Set Ray storage directory
        if self.tuner is not None:
            save_dir = get_internal_tuner(self.tuner).get_run_config().storage_path
        save_dir = Path(save_dir).expanduser().resolve()
        os.environ.setdefault('RAY_AIR_LOCAL_CACHE_DIR', str(save_dir))

        # Create Ray tuner
        if self.tuner is None:
            if groupby:
                # Group trials by config values and schedule each group separately
                groupby = (groupby,) if isinstance(groupby, str) else tuple(groupby)
                self.tune_config.scheduler = GroupScheduler(
                    self.tune_config.scheduler or FIFOScheduler(), groupby)

            self.tuner = Tuner(
                TorchTrainer(lambda config: None), # dummy trainer
                param_space={'train_loop_config': param_space},
                tune_config=self.tune_config,
                run_config=RunConfig(
                    name=experiment_name,
                    storage_path=Path(save_dir).expanduser().resolve(),
                    checkpoint_config=CheckpointConfig(
                        num_to_keep=1,
                        checkpoint_score_attribute=self.tune_config.metric,
                        checkpoint_score_order=self.tune_config.mode,
                    )
                )
            )

        # Configure Ray resources
        if gpu_memory_per_worker:
            num_gpus_per_worker = configure_gpus(gpu_memory_per_worker)
        if not torch.cuda.is_available():
            num_gpus_per_worker = 0
        scaling_config = ScalingConfig(
            num_workers=num_workers_per_trial,
            use_gpu=(num_gpus_per_worker > 0),
            resources_per_worker={
                'CPU': num_cpus_per_worker,
                'GPU': num_gpus_per_worker,
            },
        )

        # Set Ray trainer
        fit_single_model = lambda config: self.fit_single_model(
            config,
            model_creator=model_creator,
            datamodule_creator=datamodule_creator,
            callbacks_creator=callbacks_creator,
            num_workers=num_workers_per_trial,
        )
        get_internal_tuner(self.tuner).trainable = TorchTrainer(
            fit_single_model,
            scaling_config=scaling_config,
        )

        # Launch TensorBoard
        experiment_name = get_internal_tuner(self.tuner).get_run_config().name
        logdir = str(save_dir / experiment_name)
        tb = tensorboard.TensorBoard()
        tb.configure(argv=[None, '--logdir', logdir, '--port', str(tensorboard_port)])
        url = tb.launch()
        safe_print(f"TensorBoard started at {url}", '\n')

        # Run the experiment
        return self.tuner.fit()

    def get_results(
        self,
        groupby: str | Iterable[str] | None = None,
    ) -> ResultGrid | dict[str | tuple[str], ResultGrid]:
        """
        Get results of a hyperparameter tuning run.

        Parameters
        ----------
        groupby : str or Iterable[str] or None
            Config key(s) to group results by
        """
        assert self.tuner is not None, "Must call fit() or restore() first"
        results = self.tuner.get_results()
        return results if not groupby else group_results(results, groupby)

    def load_model(
        self,
        model_creator: Callable[..., pl.LightningModule],
        result: Result,
    ) -> pl.LightningModule:
        """
        Load trained model from the given Ray Tune result.

        Parameters
        ----------
        model_creator : Callable(**config) -> pl.LightningModule
            Model creator
        result : Result
            Ray Tune result
        """
        checkpoint_path = result.get_best_checkpoint(
            self.tune_config.metric, self.tune_config.mode).path
        checkpoint_path = Path(checkpoint_path, 'checkpoint.pt')
        model_creator = variable_kwargs_fn_wrapper(model_creator)
        model = model_creator(**result.config['train_loop_config'])
        state_dict = torch.load(checkpoint_path)['state_dict']

        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print("Loading model failed with strict=True:", e)
            print("Loading model with strict=False")
            model.load_state_dict(state_dict, strict=False)

        return model

    @classmethod
    def restore(cls, path: Path | str, **kwargs) -> 'LightningTuner':
        """
        Restore from a previous run.

        Parameters
        ----------
        path : Path or str
            Experiment directory
        kwargs : Any
            Additional arguments to pass to `ray.tune.Tuner.restore()`
        """
        # Recursively search for 'tuner.pkl' file within the provided directory
        # If multiple are found, use the most recently modified one
        path = Path(path).expanduser().resolve()
        path = path if path.is_dir() else path.parent
        path = sorted(path.glob('**/tuner.pkl'), key=os.path.getmtime)[-1].parent

        # Restore tuner
        lightning_tuner = cls.__new__(cls)
        dummy_trainer = TorchTrainer(lambda config: None)
        lightning_tuner.tuner = restore_ray_tuner(path, dummy_trainer, **kwargs)
        lightning_tuner.tune_config = get_internal_tuner(lightning_tuner)._tune_config

        return lightning_tuner
