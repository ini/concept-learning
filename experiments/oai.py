import os
import ray

from utils import process_grid_search_tuples

from .cub import make_concept_model



def get_config(**kwargs) -> dict:
    config = {
        # 'model_type': ray.tune.grid_search([
        #     'latent_residual',
        #     'decorrelated_residual',
        #     'iter_norm',
        #     'mi_residual',
        # ]),
        'residual_dim': ray.tune.grid_search([0, 1, 2, 4, 8, 16, 32, 64]),
        'dataset': 'oai',
        'data_dir': os.environ.get('CONCEPT_DATA_DIR', './data'),
        'save_dir': os.environ.get('CONCEPT_SAVE_DIR', './saved'),
        'num_cpus': 4,
        'gpu_memory_per_worker': '5000 MiB',
        'training_mode': 'independent',
        'num_epochs': 100,
        'lr': 1e-4,
        'batch_size': 64,
        'alpha': 1.0,
        'beta': 1.0,
        'mi_estimator_hidden_dim': 256,
        'mi_optimizer_lr': 1e-3,
        'cw_alignment_frequency': 20,
        'checkpoint_frequency': 5,
        "backbone": "resnet18",
        "num_target_network_layers": 2,
    }
    config.update(kwargs)
    config = process_grid_search_tuples(config)
    return config
