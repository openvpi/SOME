import pathlib

import click
import yaml

from utils import print_config


@click.command(help='Process the raw dataset into binary dataset')
@click.option('--config', required=True, metavar='FILE', help='Path to the configuration file')
def binarize(config):
    config = pathlib.Path(config)
    with open(config, 'r', encoding='utf8') as f:
        config = yaml.safe_load(f)
    print_config(config)


if __name__ == '__main__':
    binarize()
