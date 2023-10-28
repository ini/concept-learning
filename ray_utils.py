from __future__ import annotations
from typing import Any



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
