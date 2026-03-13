"""Configuration loading with dot-notation access and CLI override support."""

import yaml
from typing import Any, Dict, Optional


class DotDict(dict):
    """Dictionary that supports attribute-style (dot-notation) access."""

    def __getattr__(self, key: str):
        try:
            val = self[key]
            if isinstance(val, dict):
                return DotDict(val)
            return val
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any):
        self[key] = value

    def __delattr__(self, key: str):
        del self[key]

    def get(self, key: str, default=None):
        val = super().get(key, default)
        if isinstance(val, dict):
            return DotDict(val)
        return val


def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> DotDict:
    """
    Load a YAML config file and apply optional key=value overrides.

    Args:
        config_path: Path to the YAML file.
        overrides: Dict of dot-separated key paths to new values,
                   e.g. {"training.learning_rate": 0.001, "model.dropout": 0.5}

    Returns:
        DotDict config with dot-notation access.
    """
    with open(config_path, "r") as f:
        config: dict = yaml.safe_load(f)

    if overrides:
        for key_path, value in overrides.items():
            _set_nested(config, key_path.split("."), value)

    return DotDict(config)


def save_config(config: DotDict, path: str) -> None:
    """Serialize config to a YAML file."""
    with open(path, "w") as f:
        yaml.dump(dict(config), f, default_flow_style=False, sort_keys=False)


def _set_nested(d: dict, keys: list, value: Any) -> None:
    """Recursively set a nested dict value from a list of keys."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value
