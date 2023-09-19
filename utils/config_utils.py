from __future__ import annotations

import pathlib

import lightning.pytorch.utilities
import yaml

loaded_config_files = {}


def override_dict(old_config: dict, new_config: dict):
    for k, v in new_config.items():
        if isinstance(v, dict) and k in old_config:
            override_dict(old_config[k], new_config[k])
        else:
            old_config[k] = v


def read_full_config(config_path: pathlib.Path) -> dict:
    config_path = config_path.resolve()
    config_path_str = config_path.as_posix()
    if config_path in loaded_config_files:
        return loaded_config_files[config_path_str]

    with open(config_path, 'r', encoding='utf8') as f:
        config = yaml.safe_load(f)
    if 'base_config' not in config:
        loaded_config_files[config_path_str] = config
        return config

    if not isinstance(config['base_config'], list):
        config['base_config'] = [config['base_config']]
    squashed_config = {}
    for base_config in config['base_config']:
        c_path = pathlib.Path(base_config)
        full_base_config = read_full_config(c_path)
        override_dict(squashed_config, full_base_config)
    override_dict(squashed_config, config)
    squashed_config.pop('base_config')
    loaded_config_files[config_path_str] = squashed_config
    return squashed_config


@lightning.pytorch.utilities.rank_zero.rank_zero_only
def print_config(config: dict):
    for i, (k, v) in enumerate(sorted(config.items())):
        print(f"\033[0;33m{k}\033[0m: {v}", end='')
        if i < len(config) - 1:
            print(", ", end="")
        if i % 5 == 4:
            print()
    print()
