from experiments.cifar import make_bottleneck_model as make_bottleneck_model_cifar
from experiments.cifar import make_whitening_model as make_whitening_model_cifar

from experiments.cub import make_bottleneck_model as make_bottleneck_model_cub
from experiments.cub import make_whitening_model as make_whitening_model_cub



def bottleneck_model_fn(config):
    if config['dataset'] == 'cifar100':
        return make_bottleneck_model_cifar
    elif config['dataset'] == 'cub':
        return make_bottleneck_model_cub
    else:
        raise ValueError('Unknown dataset:', config['dataset'])

def whitening_model_fn(config):
    if config['dataset'] == 'cifar100':
        return make_whitening_model_cifar
    elif config['dataset'] == 'cub':
        return make_whitening_model_cub
    else:
        raise ValueError('Unknown dataset:', config['dataset'])