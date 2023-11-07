from __future__ import annotations

from collections import ChainMap
from ray.tune.search.variant_generator import generate_variants, grid_search
from typing import Any



def process_grid_search_tuples(config: dict):
    """
    Process configuration dictionary with grid search tuples.

    Parameters
    ----------
    config : dict
        Configuration dictionary, with entries of the form:
            * k: v
            * (k_0, k_1, ..., k_n): grid_search([
                (v0_0, v0_1, ..., v0_n),
                (v1_0, v1_1, ..., v1_n),
                (v2_0, v2_1, ..., v2_n),
                ...,
            ])

    Example
    -------
    ```
    config = {
        ('a', 'b'): grid_search([(1, 2), (3, 4), (5, 6)]),
        'c': 7,
    }
    ```
    results in the following combinations:
        * `{'a': 1, 'b': 2, 'c': 7}`
        * `{'a': 3, 'b': 4, 'c': 7}`
        * `{'a': 5, 'b': 6, 'c': 7}`
    """
    # Turn all keys into tuples, and all values into a grid search over tuples
    config = {
        k if isinstance(k, tuple) else (k,): v if isinstance(k, tuple) else (v,)
        for k, v in config.items()
    }

    # Convert into a grid search over individual config dictionaries
    merge_dicts = lambda dicts: dict(ChainMap(*dicts))
    config = grid_search([
        merge_dicts(dict(zip(k, v)) for k, v in reversed(spec.items()))
        for _, spec in generate_variants(config)
    ])

    return config


class RayConfig(dict):
    """
    Handles get / set in config dictionaries with grid search.
    """

    def get(self, key: str, default: Any = ...) -> Any:
        """
        Get a value from the config.

        Parameters
        ----------
        key : str
            Config key
        default : Any
            Default value if key is not found
        """
        try:
            if key in self:
                value = self[key]
                if isinstance(value, dict) and len(value) == 1 and 'grid_search' in value:
                    return value['grid_search']
                else:
                    return value

            elif 'train_loop_config' in self:
                return self.get(self['train_loop_config'], key, default=default)

            elif 'grid_search' in self:
                values = {item[key] for item in self['grid_search']}
                assert len(values) == 1, f'Inconsistent values for {key}: {values}'
                return next(iter(values))

            raise KeyError

        except KeyError:

            if default is not ...:
                return default

        raise KeyError(f'Key not found: {key}')

    def set(self, key: str, value: Any):
        """
        Set a value in the config.

        Parameters
        ----------
        key : str
            Config key
        value : Any
            Config value
        """
        if 'grid_search' in self:
            for item in self['grid_search']:
                item[key] = value
        else:
            self[key] = value

    def update(self, other: dict):
        """
        Update the config from another dictionary.
        """
        for key, value in other.items():
            self.set(key, value)
