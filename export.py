import importlib
import pathlib
from typing import Dict, Tuple, Union

import click
import onnx
import onnxsim
import torch
import yaml

import deployment
from utils.config_utils import print_config


def onnx_override_io_shapes(
        model,  # ModelProto
        input_shapes: Dict[str, Tuple[Union[str, int]]] = None,
        output_shapes: Dict[str, Tuple[Union[str, int]]] = None,
):
    """
    Override the shapes of inputs/outputs of the model graph (in-place operation).
    :param model: model to perform the operation on
    :param input_shapes: a dict with keys as input/output names and values as shape tuples
    :param output_shapes: the same as input_shapes
    """
    def _override_shapes(
            shape_list_old,  # RepeatedCompositeFieldContainer[ValueInfoProto]
            shape_dict_new: Dict[str, Tuple[Union[str, int]]]):
        for value_info in shape_list_old:
            if value_info.name in shape_dict_new:
                name = value_info.name
                dims = value_info.type.tensor_type.shape.dim
                assert len(shape_dict_new[name]) == len(dims), \
                    f'Number of given and existing dimensions mismatch: {name}'
                for i, dim in enumerate(shape_dict_new[name]):
                    if isinstance(dim, int):
                        dims[i].dim_param = ''
                        dims[i].dim_value = dim
                    else:
                        dims[i].dim_value = 0
                        dims[i].dim_param = dim

    if input_shapes is not None:
        _override_shapes(model.graph.input, input_shapes)
    if output_shapes is not None:
        _override_shapes(model.graph.output, output_shapes)


@click.command(help='Run inference with a trained model')
@click.option('--model', required=True, metavar='CKPT_PATH', help='Path to the model checkpoint (*.ckpt)')
@click.option('--out', required=False, metavar='ONNX_PATH', help='Path to the output model (*.onnx)')
def export(model, out):
    model_path = pathlib.Path(model)
    with open(model_path.with_name('config.yaml'), 'r', encoding='utf8') as f:
        config = yaml.safe_load(f)
    print_config(config)
    module_cls = deployment.task_module_mapping[config['task_cls']]

    pkg = ".".join(module_cls.split(".")[:-1])
    cls_name = module_cls.split(".")[-1]
    module_cls = getattr(importlib.import_module(pkg), cls_name)
    assert issubclass(module_cls, deployment.BaseONNXModule), \
        f'Module class {module_cls} is not a subclass of {deployment.BaseONNXModule}.'
    module_ins = module_cls(config=config, model_path=model_path)

    waveform = torch.randn((1, 114514), dtype=torch.float32, device=module_ins.device)
    out_path = pathlib.Path(out) if out is not None else model_path.with_suffix('.onnx')
    torch.onnx.export(
        module_ins,
        waveform,
        out_path,
        input_names=['waveform'],
        output_names=[
            'note_midi',
            'note_rest',
            'note_dur'
        ],
        dynamic_axes={
            'waveform': {
                1: 'n_samples'
            },
            'note_midi': {
                1: 'n_notes'
            },
            'note_rest': {
                1: 'n_notes'
            },
            'note_dur': {
                1: 'n_notes'
            },
        },
        opset_version=17
    )
    onnx_model = onnx.load(out_path.as_posix())
    onnx_override_io_shapes(onnx_model, output_shapes={
        'note_midi': (1, 'n_notes'),
        'note_rest': (1, 'n_notes'),
        'note_dur': (1, 'n_notes'),
    })
    print('Running ONNX Simplifier...')
    onnx_model, check = onnxsim.simplify(
        onnx_model,
        include_subgraph=True
    )
    assert check, 'Simplified ONNX model could not be validated'
    onnx.save(onnx_model, out_path)


if __name__ == '__main__':
    export()
