# Concept Learning

### Training

Promises and Pitfalls of Black-Box Concept Learning Models:

    python train.py --config experiments.pitfalls

CIFAR 100:

    python train.py --config experiments.cifar

CUB:

    python train.py --config experiments.cub

Run with `--help` for more options.

### Evaluation

    python evaluate.py --exp-dir <GENERATED_EXPERIMENT_DIR>
    python plot.py --exp-dir <GENERATED_EXPERIMENT_DIR>

Run with `--help` for more options.
