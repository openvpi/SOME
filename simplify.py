import pathlib

import click
import torch


@click.command(help='Simplify a checkpoint file, dropping all useless keys for inference.')
@click.argument('input_ckpt', metavar='INPUT_CKPT')
@click.argument('output_ckpt', metavar='OUTPUT_CKPT')
def simplify(input_ckpt, output_ckpt):
    input_ckpt_path = pathlib.Path(input_ckpt)
    output_ckpt_path = pathlib.Path(output_ckpt)
    ckpt = torch.load(input_ckpt_path)
    ckpt = {
        'state_dict': ckpt['state_dict']
    }
    torch.save(ckpt, output_ckpt_path)


if __name__ == '__main__':
    simplify()
