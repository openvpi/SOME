import importlib
import pathlib

import click

import preprocessing
from utils.config_utils import read_full_config, print_config


@click.command(help='Process the raw dataset into binary dataset')
@click.option('--config', required=True, metavar='FILE', help='Path to the configuration file')
def binarize(config):
    config = pathlib.Path(config)
    config = read_full_config(config)
    print_config(config)
    binarizer_cls = config['binarizer_cls']
    pkg = ".".join(binarizer_cls.split(".")[:-1])
    cls_name = binarizer_cls.split(".")[-1]
    binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
    assert issubclass(binarizer_cls, preprocessing.BaseBinarizer), \
        f'Binarizer class {binarizer_cls} is not a subclass of {preprocessing.BaseBinarizer}.'
    print("| Binarizer: ", binarizer_cls)
    binarizer_cls(config=config).process()


if __name__ == '__main__':
    binarize()
