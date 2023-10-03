
from experiments.cifar_train_ray import main
from ray import tune
if __name__ == '__main__':


    experiment_config = {
        "mode" : "train",
        "model_type" : tune.grid_search(["baseline", "corr", "mi"]),
        "save_dir" : "/data/renos/supervised_concept_learning/",
        "data_dir" : "/data/Datasets/cifar/",
        "ray_storage_dir" : "/data/renos/ray_results/",
        "residual_dim" : 1,
        "lr" : 1e-4,
        "num_epochs" : 1000,
        "alpha" : 1.0,
        "beta" : 1.0,
        "mi_estimator_hidden_dim": 256,
        "mi_optimizer_lr" : 0.001,
        "whitening_alignment_frequency": 20,
        "num_gpus" : 0.2,
        "num_samples" : 1,
        "batch_size" : 64,
        "checkpoint_freq" : 5,
        "norm_type" : "iter_norm",
        "T_whitening" : 3,
    }
    main(experiment_config)


