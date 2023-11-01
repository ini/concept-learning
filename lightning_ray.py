from __future__ import annotations

import numpy as np
import os
import pynvml
import pytorch_lightning as pl
import random
import ray
import tempfile
import torch

from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from pytorch_lightning.accelerators.mps import MPSAccelerator
from ray.air import CheckpointConfig, RunConfig, ScalingConfig
from ray.experimental.tqdm_ray import safe_print
from ray.train import Checkpoint
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment
from ray.train.torch import TorchTrainer
from ray.train.trainer import BaseTrainer
from ray.tune import ExperimentAnalysis, ResultGrid, Trainable, TuneConfig, Tuner
from ray.tune.execution.experiment_state import _ResumeConfig
from ray.tune.execution.tune_controller import TuneController
from ray.tune.experiment import Trial
from ray.tune.schedulers import FIFOScheduler, TrialScheduler
from ray.tune.tuner import TunerInternal
from tensorboard import program as tensorboard
from typing import Any, Callable, Iterable, Literal, Type



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

def group_results(
    results: ResultGrid, groupby: str | Iterable[str],
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
    total_num_gpus = pynvml.nvmlDeviceGetCount()

    # For each GPU, calculate the number of workers that can fit in memory
    num_workers = np.zeros(total_num_gpus)
    for i in range(total_num_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        num_workers[i] = memory_info.free // gpu_memory_per_worker

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


class RayCallback(pl.Callback):
    """
    Callback class for using Ray Tune with PyTorch Lightning.
    """

    def __init__(self, checkpoint_frequency: int = 1, **kwargs):
        """
        Parameters
        ----------
        checkpoint_frequency : int
            Frequency of checkpoints (i.e. every N epochs)
        """
        super().__init__()
        self.checkpoint_frequency = checkpoint_frequency
        self.metrics = {}

    def create_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint_dir: Path | str,
    ) -> Checkpoint:
        """
        Create a checkpoint.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning trainer
        pl_module : pl.LightningModule
            PyTorch Lightning module
        checkpoint_dir : Path or str
            Path to checkpoint directory
        """
        checkpoint_dir = Path(checkpoint_dir)
        trainer.save_checkpoint(checkpoint_dir / 'checkpoint.pt')
        return Checkpoint.from_directory(checkpoint_dir)

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
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

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        """
        Report metrics when a validation epoch ends.

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
        if ray.train.get_context().get_local_rank() == 0:
            if self.checkpoint_frequency:
                if (trainer.current_epoch + 1) % self.checkpoint_frequency == 0:
                    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                        temp_checkpoint_dir = Path(temp_checkpoint_dir)
                        trainer.save_checkpoint(temp_checkpoint_dir / 'checkpoint.pt')
                        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                        ray.train.report(metrics=self.metrics, checkpoint=checkpoint)

                else:
                    ray.train.report(metrics=self.metrics)


class LightningTuner(ABC):
    """
    Use Ray Tune to run hyperparamter optimization for a PyTorch Lightning model.

    Example
    -------
    ```
    class MyTuner(LightningTuner):

        def get_datamodule(self, config):
            return MyDataModule(**config)

        def get_model(self, config):
            return MyModel(**config)

    >>> config_space = {lr: ray.tune.grid_search([1e-3, 1e-4, 1e-5])}
    >>> tuner = MyTuner(metric='val_acc', mode='max')
    >>> tuner.fit(config_space)
    ```
    """

    def __init__(
        self,
        metric: str | None = None,
        mode: Literal['min', 'max'] = 'max',
        **tune_config_kwargs):
        """
        Parameters
        ----------
        metric : str
            Metric to optimize
        mode : one of {'min', 'max'}
            Whether to minimize or maximize the metric
        tune_config_kwargs : dict[str, Any]
            Additional arguments to pass to TuneConfig
        """
        self.tuner = None
        self.tune_config = TuneConfig(metric=metric, mode=mode, **tune_config_kwargs)

    @abstractmethod
    def get_datamodule(self, config: dict[str, Any]) -> pl.LightningDataModule:
        """
        Get PyTorch Lightning data module for specified configuration.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary
        """
        pass

    @abstractmethod
    def get_model(self, config: dict[str, Any]) -> pl.LightningModule:
        """
        Get PyTorch Lightning model for specified configuration.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary
        """
        pass

    def get_callbacks(self, config: dict[str, Any]) -> list[pl.Callback]:
        """
        Get PyTorch Lightning callbacks for specified configuration.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary
        """
        return []

    def train_model(self, config: dict[str, Any]):
        """
        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary
        """
        trainer = pl.Trainer(
            accelerator='cpu' if MPSAccelerator.is_available() else 'auto',
            strategy=RayDDPStrategy(find_unused_parameters=True),
            devices='auto',
            logger=False, # logging metrics is handled by RayCallback
            callbacks=[*self.get_callbacks(config), RayCallback(**config)],
            max_epochs=config.get('max_epochs', None),
            enable_checkpointing=False, # checkpointing is handled by RayCallback
            enable_progress_bar=False,
            plugins=[RayLightningEnvironment()],
        )
        trainer.fit(self.get_model(config), self.get_datamodule(config))

    def fit(
        self,
        param_space: dict[str, Any],
        save_dir: Path | str | None = '.',
        experiment_name: str | None = None,
        num_workers_per_trial: int = 1,
        num_cpus_per_worker: int = 1,
        num_gpus_per_worker: int = 1,
        gpu_memory_per_worker: str | int | float | None = None,
        tensorboard_port: int = 0,
        groupby: str | Iterable[str] = (),
    ) -> ResultGrid:
        """
        Run an experiment using Ray Tune.

        Parameters
        ----------
        param_space : dict[str, Any]
            Search space of the tuning job
        save_dir : Path or str
            Directory to save results to
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
        if self.tuner is None:
            # Group trials by config values and schedule each group separately
            if groupby:
                groupby = (groupby,) if isinstance(groupby, str) else tuple(groupby)
                self.tune_config.scheduler = GroupScheduler(
                    self.tune_config.scheduler or FIFOScheduler(), groupby)

            # Create Ray Trainer
            if gpu_memory_per_worker:
                num_gpus_per_worker = configure_gpus(gpu_memory_per_worker)
            if not torch.cuda.is_available():
                num_gpus_per_worker = 0

            # Set Ray storage directory
            save_dir = Path(save_dir).expanduser().resolve()
            os.environ.setdefault('RAY_AIR_LOCAL_CACHE_DIR', str(save_dir))

            # Create Ray Tuner
            self.tuner = Tuner(
                TorchTrainer(self.train_model),
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

        # Set Ray storage directory
        save_dir = get_internal_tuner(self.tuner).get_run_config().storage_path
        save_dir = Path(save_dir).expanduser().resolve()
        os.environ.setdefault('RAY_AIR_LOCAL_CACHE_DIR', str(save_dir))

        # Configure GPUs
        if gpu_memory_per_worker:
            num_gpus_per_worker = configure_gpus(gpu_memory_per_worker)
        if not torch.cuda.is_available():
            num_gpus_per_worker = 0

        # Set scaling config
        scaling_config = ScalingConfig(
            num_workers=num_workers_per_trial,
            use_gpu=(num_gpus_per_worker > 0),
            resources_per_worker={
                'CPU': num_cpus_per_worker,
                'GPU': configure_gpus(gpu_memory_per_worker),
            },
        )

        # Update trainer
        trainer = TorchTrainer(self.train_model, scaling_config=scaling_config)
        get_internal_tuner(self.tuner).trainable = trainer

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

    def load_model(self, result: ray.train.Result) -> pl.LightningModule:
        """
        Load trained model from the given Ray Tune result.

        Parameters
        ----------
        result : ray.train.Result
            Ray Tune result
        """
        checkpoint_path = result.get_best_checkpoint(
            self.tune_config.metric, self.tune_config.mode).path
        checkpoint_path = Path(checkpoint_path, 'checkpoint.pt')
        model = self.get_model(result.config['train_loop_config'])
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        return model

    @classmethod
    def restore(cls, path: Path | str, **kwargs) -> 'LightningTuner':
        """
        Restore from a previous run.

        Parameters
        ----------
        path : Path or str
            Experiment directory
        kwargs : dict[str, Any]
            Additional arguments to pass to `ray.tune.Tuner.restore()`
        """
        # Recursively search for 'tuner.pkl' file within the provided directory
        # If multiple are found, use the most recently modified one
        path = Path(path).expanduser().resolve()
        path = path if path.is_dir() else path.parent
        path = sorted(path.glob('**/tuner.pkl'), key=os.path.getmtime)[-1].parent

        # Restore tuner
        lightning_tuner = cls.__new__(cls)
        trainable = TorchTrainer(lightning_tuner.train_model)
        lightning_tuner.tuner = restore_ray_tuner(path, trainable, **kwargs)
        lightning_tuner.tune_config = get_internal_tuner(lightning_tuner)._tune_config

        return lightning_tuner
