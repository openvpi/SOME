import importlib
import pathlib

import click
import librosa
import mido
import numpy as np
import yaml

import inference
from utils import print_config
from utils.slicer2 import Slicer

task_inference_mapping = {
    'training.MIDIExtractionTask': 'inference.MIDIExtractionInference',
    'training.QuantizedMIDIExtractionTask': 'inference.QuantizedMIDIExtractionInference',
}


@click.command(help='Run inference with a trained model')
@click.option('--model', required=True, metavar='CKPT_PATH', help='Path to the model checkpoint (*.ckpt)')
@click.option('--wav', required=True, metavar='WAV_PATH', help='Path to the input wav file (*.wav)')
@click.option('--midi', required=False, metavar='MIDI_PATH', help='Path to the output MIDI file (*.mid)')
def infer(model, wav, midi):
    model_path = pathlib.Path(model)
    with open(model_path.with_name('config.yaml'), 'r', encoding='utf8') as f:
        config = yaml.safe_load(f)
    print_config(config)
    infer_cls = task_inference_mapping[config['task_cls']]

    pkg = ".".join(infer_cls.split(".")[:-1])
    cls_name = infer_cls.split(".")[-1]
    infer_cls = getattr(importlib.import_module(pkg), cls_name)
    assert issubclass(infer_cls, inference.BaseInference), \
        f'Binarizer class {infer_cls} is not a subclass of {inference.BaseInference}.'
    infer_ins = infer_cls(config=config, model_path=model_path)

    wav_path = pathlib.Path(wav)
    waveform, _ = librosa.load(wav_path, sr=config['audio_sample_rate'], mono=True)
    slicer = Slicer(sr=config['audio_sample_rate'], max_sil_kept=1000)
    chunks = slicer.slice(waveform)
    midis = infer_ins.infer([c['waveform'] for c in chunks])

    # TODO: write MIDI
    midi_path = pathlib.Path(midi) if midi is not None else wav_path.with_suffix('.mid')
    midi_file = mido.MidiFile(charset='utf8')
    midi_track = mido.MidiTrack()
    last_time = 0
    for offset, segment in zip([c['offset'] for c in chunks], midis):
        note_midi = np.round(segment['note_midi']).astype(np.int64).tolist()
        # tempo = 120
        offset_tick = round(offset * 960)
        note_tick = np.diff(np.round(np.cumsum(segment['note_dur']) * 960).astype(np.int64), prepend=0).tolist()
        note_rest = segment['note_rest'].tolist()
        start = offset_tick
        for i in range(len(note_midi)):
            end = start + note_tick[i]
            if not note_rest[i]:
                midi_track.append(mido.Message('note_on', note=note_midi[i], time=start - last_time))
                midi_track.append(mido.Message('note_off', note=note_midi[i], time=note_tick[i]))
                last_time = end
            start = end
    midi_file.tracks.append(midi_track)
    midi_file.save(midi_path)


if __name__ == '__main__':
    infer()
