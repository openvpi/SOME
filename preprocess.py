import pathlib

import click
import yaml


@click.command(help='Process the raw dataset into binary dataset')
@click.option('--config', metavar='FILE', help='Path to the configuration file')
def preprocess(config):
    config = pathlib.Path(config)
    with open(config, 'r', encoding='utf8') as f:
        config = yaml.safe_load(f)
    for i, (k, v) in enumerate(sorted(config.items())):
        print(f"\033[0;33m{k}\033[0m: {v}, ", end="\n" if i % 5 == 4 else "")
    print("")


if __name__ == '__main__':
    preprocess()
