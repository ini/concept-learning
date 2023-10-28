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
from dataclasses import dataclass
from pathlib import Path
from pytorch_lightning.accelerators.mps import MPSAccelerator
from ray.air import CheckpointConfig, RunConfig, ScalingConfig
from ray.experimental.tqdm_ray import safe_print
from ray.train import Checkpoint
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment
from ray.train.torch import TorchTrainer
from ray.tune import ExperimentAnalysis, ResultGrid, TuneConfig, Tuner
from ray.tune.execution.tune_controller import TuneController
from ray.tune.experiment import Trial
from ray.tune.schedulers import FIFOScheduler, TrialScheduler
from tensorboard import program as tensorboard
from typing import Any, Iterable, Literal



def group_results(
    results: ResultGrid, groupby: str | Iterable[str],
) -> dict[tuple[str], ResultGrid]:
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
        if isinstance(groupby, str):
            group = trial.config[groupby]
            trials[group].append(trial)
        else:
            group = tuple(trial.config[key] for key in groupby)
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

def set_cuda_visible_devices(available_memory_threshold: float):
    """
    Set CUDA_VISIBLE_DEVICES to the GPUs whose fraction of available memory
    is at least a given threshold.

    When running processes with fractional GPUs, set the threshold to
    the fraction of the GPU memory that is available to each worker.

    Parameters
    ----------
    available_memory_threshold : float in range [0, 1]
        Threshold for fraction of available GPU memory
    """
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError_LibraryNotFound:
        return

    available_devices = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if memory_info.free / memory_info.total >= available_memory_threshold:
            available_devices.append(i)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, available_devices))



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
        self.schedulers = defaultdict(lambda: deepcopy(scheduler))

    def on_trial_add(self, tune_controller: TuneController, trial: Trial):
        """
        Called when a new trial is added to the trial runner.
        """
        key = tuple(trial.config.get(key, None) for key in self.groupby)
        self.schedulers[key].on_trial_add(tune_controller, trial)

    def on_trial_error(self, tune_controller: TuneController, trial: Trial):
        """
        Notification for the error of trial.
        """
        key = tuple(trial.config.get(key, None) for key in self.groupby)
        self.schedulers[key].on_trial_error(tune_controller, trial)

    def on_trial_result(
        self, tune_controller: TuneController, trial: Trial, result: dict,
    ) -> str:
        """
        Called on each intermediate result returned by a trial.
        """
        key = tuple(trial.config.get(key, None) for key in self.groupby)
        return self.schedulers[key].on_trial_result(tune_controller, trial, result)

    def on_trial_complete(
        self, tune_controller: TuneController, trial: Trial, result: dict,
    ):
        """
        Notification for the completion of trial.
        """
        key = tuple(trial.config.get(key, None) for key in self.groupby)
        self.schedulers[key].on_trial_complete(tune_controller, trial, result)

    def on_trial_remove(self, tune_controller: TuneController, trial: Trial):
        """
        Called to remove trial.
        """
        key = tuple(trial.config.get(key, None) for key in self.groupby)
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


class LightningTuner:
    """
    Use Ray Tune to run hyperparamter optimization for a PyTorch Lightning model.
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

    def get_datamodule(self, config: dict[str, Any]) -> pl.LightningDataModule:
        """
        Get PyTorch Lightning data module for specified configuration.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary
        """
        raise NotImplementedError

    def get_model(self, config: dict[str, Any]) -> pl.LightningModule:
        """
        Get PyTorch Lightning model for specified configuration.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary
        """
        raise NotImplementedError

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
            strategy=RayDDPStrategy(),
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
        tensorboard_port : int
            Port for TensorBoard to visualize results
        groupby : str or Iterable[str]
            Config key(s) to group by
        """
        if self.tuner is None:
            # Group trials by config values and schedule each group separately
            if groupby:
                self.tune_config.scheduler = GroupScheduler(
                    self.tune_config.scheduler or FIFOScheduler(), groupby)

            # Create Ray Trainer
            num_gpus_per_worker = num_gpus_per_worker if torch.cuda.is_available() else 0
            trainer = TorchTrainer(
                self.train_model,
                scaling_config=ScalingConfig(
                    num_workers=num_workers_per_trial,
                    use_gpu=(num_gpus_per_worker > 0),
                    resources_per_worker={
                        'CPU': num_cpus_per_worker,
                        'GPU': num_gpus_per_worker,
                    },
                ),
            )

            # Set Ray storage directory
            save_dir = Path(save_dir).expanduser().resolve()
            os.environ.setdefault('RAY_AIR_LOCAL_CACHE_DIR', str(save_dir))

            # Create Ray Tuner
            self.tuner = Tuner(
                trainer,
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
        ray_internal_tuner = self.tuner._local_tuner or self.tuner._remote_tuner
        save_dir = ray_internal_tuner._run_config.storage_path
        save_dir = Path(save_dir).expanduser().resolve()
        os.environ.setdefault('RAY_AIR_LOCAL_CACHE_DIR', str(save_dir))

        # Set CUDA_VISIBLE_DEVICES to the GPUs with enough available memory
        # to run at least one worker.
        num_gpus_per_worker = trainer.scaling_config.resources_per_worker['GPU']
        if num_gpus_per_worker < 1:
            set_cuda_visible_devices(available_memory_threshold=num_gpus_per_worker)

        # Launch TensorBoard
        experiment_name = ray_internal_tuner._run_config.name
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
    ) -> ResultGrid | dict[str, ResultGrid]:
        """
        Get results of a hyperparameter tuning run.

        Parameters
        ----------
        groupby : str or Iterable[str] or None
            Config key(s) to group results by
        """
        assert self.tuner is not None, "Must call fit() or restore() first"
        results = self.tuner.get_results()
    
        if groupby:
            for trial in results._experiment_analysis.trials:
                trial.config = trial.config['train_loop_config']
            return group_results(results, groupby)

        return results

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
        checkpoint_path = Path(checkpoint_path) / 'checkpoint.pt'
        model = self.get_model(result.config['train_loop_config'])
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        return model

    @classmethod
    def restore(cls, path: Path | str, **tuner_restore_kwargs) -> 'LightningTuner':
        """
        Restore from a previous run.

        Parameters
        ----------
        path : Path or str
            Experiment directory
        tuner_restore_kwargs : dict[str, Any]
            Additional arguments to pass to `ray.tune.Tuner.restore()`
        """
        # Recursively search for 'tuner.pkl' file within the provided directory
        # If multiple are found, use the most recently modified one
        paths = Path(path).expanduser().resolve().glob('**/tuner.pkl')
        path = sorted(paths, key=os.path.getmtime)[-1].parent

        # Restore Tuner
        tuner = LightningTuner()
        tuner.tuner = Tuner.restore(
            str(path), TorchTrainer(tuner.train_model), **tuner_restore_kwargs)

        # Restore TuneConfig
        ray_internal_tuner = tuner.tuner._local_tuner or tuner.tuner._remote_tuner
        tuner.tune_config = ray_internal_tuner._tune_config

        return tuner
