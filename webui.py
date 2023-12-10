import importlib
import os
import pathlib
import time
from typing import Dict, Tuple

import click
import gradio as gr
import librosa
import yaml

import inference
from inference import BaseInference
from utils.infer_utils import build_midi_file
from utils.slicer2 import Slicer

_work_dir: pathlib.Path = None
_infer_instances: Dict[str, Tuple[BaseInference, dict]] = {}  # dict mapping model_rel_path to (infer_ins, config)


def infer(model_rel_path, input_audio_path, tempo_value):
    if not model_rel_path or not input_audio_path or tempo_value is None:
        return None, "Error: required inputs not specified."
    if model_rel_path not in _infer_instances:
        model_path = _work_dir / model_rel_path
        with open(model_path.with_name('config.yaml'), 'r', encoding='utf8') as f:
            config = yaml.safe_load(f)
        infer_cls = inference.task_inference_mapping[config['task_cls']]

        pkg = ".".join(infer_cls.split(".")[:-1])
        cls_name = infer_cls.split(".")[-1]
        infer_cls = getattr(importlib.import_module(pkg), cls_name)
        assert issubclass(infer_cls, inference.BaseInference), \
            f'Binarizer class {infer_cls} is not a subclass of {inference.BaseInference}.'
        infer_ins = infer_cls(config=config, model_path=model_path)
        print(f"Initialized: {infer_ins}")
        _infer_instances[model_rel_path] = (infer_ins, config)
    else:
        infer_ins, config = _infer_instances[model_rel_path]

    input_audio_path = pathlib.Path(input_audio_path)
    total_duration = librosa.get_duration(filename=input_audio_path)
    if total_duration > 20 * 60:  # 20 minutes
        return None, f"Error: the input audio is too long (>= 20 minutes)."

    try:
        waveform, _ = librosa.load(input_audio_path, sr=config['audio_sample_rate'], mono=True)
    except:
        return None, f"Error: unsupported or corrupt file format: {input_audio_path.name}"

    start_time = time.time()
    slicer = Slicer(sr=config['audio_sample_rate'], max_sil_kept=1000)
    chunks = slicer.slice(waveform)
    midis = infer_ins.infer([c['waveform'] for c in chunks])
    infer_time = time.time() - start_time
    rtf = infer_time / total_duration
    print(f'RTF: {rtf}')

    midi_file = build_midi_file([c['offset'] for c in chunks], midis, tempo=tempo_value)

    output_midi_path = input_audio_path.with_suffix('.mid')
    midi_file.save(output_midi_path)
    os.remove(input_audio_path)

    return output_midi_path, f"Cost {round(infer_time, 2)} s, RTF: {round(rtf, 3)}"


@click.command(help='Launch the web UI for inference')
@click.option('--port', type=int, default=7860, help='Server port')
@click.option('--addr', type=str, required=False, help='Server address')
@click.option('--work_dir', type=str, required=False, help='Directory to read the experiments')
def webui(port, work_dir, addr):
    if work_dir is None:
        work_dir = pathlib.Path(__file__).with_name('experiments')
    else:
        work_dir = pathlib.Path(work_dir)
    assert work_dir.is_dir(), f'{work_dir} is not a directory.'
    global _work_dir
    _work_dir = work_dir
    choices = [
        p.relative_to(work_dir).as_posix()
        for p in work_dir.rglob('*.ckpt')
    ]
    if len(choices) == 0:
        raise FileNotFoundError(f'No checkpoints found in {work_dir}.')
    iface = gr.Interface(
        title="SOME: Singing-Oriented MIDI Extractor",
        description="Submit an audio file and download the extracted MIDI file.",
        theme="default",
        fn=infer,
        inputs=[
            gr.components.Dropdown(
                label="Model Checkpoint", choices=choices, value=choices[0],
                multiselect=False, allow_custom_value=False
            ),
            gr.components.Audio(label="Input Audio File", type="filepath"),
            gr.components.Number(label='Tempo Value', minimum=20, maximum=200, value=120),
        ],
        outputs=[
            gr.components.File(label="Output MIDI File", file_types=['.mid']),
            gr.components.Label(label="Inference Statistics"),
        ]
    )
    iface.queue(concurrency_count=10)
    iface.launch(server_port=port, server_name=addr)


if __name__ == "__main__":
    webui()
