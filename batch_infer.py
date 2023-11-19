import importlib
import pathlib
from csv import DictReader, DictWriter
from typing import List

import click
import librosa
import tqdm
import yaml

import inference
from utils.config_utils import print_config
from utils.slicer2 import Slicer

task_inference_mapping = {
    'training.MIDIExtractionTask': 'inference.MIDIExtractionInference',
    'training.QuantizedMIDIExtractionTask': 'inference.QuantizedMIDIExtractionInference',
}


def model_init(model_path):
    model_path = pathlib.Path(model_path)
    with open(model_path.with_name('config.yaml'), 'r', encoding='utf8') as f:
        config = yaml.safe_load(f)
    print_config(config)
    infer_cls = task_inference_mapping[config['task_cls']]

    pkg = ".".join(infer_cls.split(".")[:-1])
    cls_name = infer_cls.split(".")[-1]
    infer_cls = getattr(importlib.import_module(pkg), cls_name)
    assert issubclass(infer_cls, inference.BaseInference), \
        f'Binarizer class {infer_cls} is not a subclass of {inference.BaseInference}.'
    model = infer_cls(config=config, model_path=model_path)
    return model, config


def calc_seq(note_midi, note_rest):
    midi_num = round(note_midi, 0)
    cent = int(round(note_midi - midi_num, 2) * 100)
    if cent > 0:
        cent = f"+{cent}"
    elif cent == 0:
        cent = ""

    seq = f"{librosa.midi_to_note(midi_num, unicode=False)}{cent}"
    return seq if not note_rest else 'rest'


def infer(wav, infer_ins, config):
    wav_path = pathlib.Path(wav)
    waveform, _ = librosa.load(wav_path, sr=config['audio_sample_rate'], mono=True)
    slicer = Slicer(sr=config['audio_sample_rate'], max_sil_kept=1000)
    chunks = slicer.slice(waveform)
    midis = infer_ins.infer([c['waveform'] for c in chunks])

    res: list = []
    for offset, segment in zip([c['offset'] for c in chunks], midis):
        offset = round(offset, 6)
        note_midi = segment['note_midi'].tolist()
        # tempo = 120
        note_dur = segment['note_dur'].tolist()
        note_rest = segment['note_rest'].tolist()
        assert len(note_midi) == len(note_dur) == len(note_rest)

        last_time = 0
        for mid, dur, rest in zip(note_midi, note_dur, note_rest):
            dur = round(dur, 6)
            last_time = round(last_time, 6)
            seq = calc_seq(mid, rest)
            midi_info: dict = {
                'start_time': round(offset + last_time, 6),
                'end_time': round(offset + last_time + dur, 6),
                'note_seq': seq
            }
            if res:
                if midi_info['start_time'] < res[-1]['end_time']:
                    midi_info['start_time'] = res[-1]['end_time']
            midi_info['note_dur'] = round(midi_info['end_time'] - midi_info['start_time'], 6)
            res.append(midi_info)
            last_time += dur
    return res


def get_word_durs(ph_durs, ph_nums):
    res = []
    cur = 0
    s_time = 0
    for num_phonemes in ph_nums:
        word_dur = round(sum(ph_durs[cur:cur + num_phonemes]), 6)
        ed_time = s_time + word_dur
        res.append((round(s_time, 6), round(ed_time, 6)))
        cur += num_phonemes
        s_time += word_dur
    return res


def midi_align(midi_res, midi_durs, tolerance=0.05):
    res = []
    bound = [x[0] for x in midi_durs] + [midi_durs[-1][1]]

    for mid in midi_res:
        for i in range(len(bound)):
            if bound[i] - tolerance <= mid['start_time'] <= bound[i] + tolerance:
                mid['start_time'] = bound[i]
            if bound[i] - tolerance <= mid['end_time'] <= bound[i] + tolerance:
                mid['end_time'] = bound[i]
        mid['note_dur'] = round(mid['end_time'] - mid['start_time'], 6)
        if mid['note_dur'] > 0:
            res.append(mid)
    return res


def get_all_overlap_midis(interval, segments):
    res = []
    for segment in segments:
        if interval[0] < segment['start_time'] < interval[1]:
            res.append(segment)
        elif interval[0] < segment['end_time'] < interval[1]:
            res.append(segment)
        elif segment['start_time'] <= interval[0] and interval[1] <= segment['end_time']:
            res.append(segment)
    return res


def get_max_overlap_midi(interval, segments):
    matching_segment = 'rest'
    max_overlap = 0

    for segment in segments:
        overlap = max(0, min(interval[1], segment['end_time']) - max(interval[0], segment['start_time']))
        if overlap > max_overlap:
            max_overlap = overlap
            matching_segment = segment['note_seq']
    return matching_segment


@click.command(help='Batch inference on existing DiffSinger dataset.')
@click.option(
    '--dataset', required=True, metavar='RAW_DATA_DIR',
    help='Path to the dataset directory. Equivalent to \'raw_data_dir\' in DiffSinger configuration files.'
)
@click.option('--model', required=True, metavar='CKPT_PATH', help='Path to the model checkpoint (*.ckpt)')
@click.option('--round_midi', is_flag=True, help='Round MIDI values to integers')
@click.option(
    '--csv', required=False, metavar='CSV_PATH',
    help='Path to the output transcriptions.csv file (default to the same file in the dataset)'
)
@click.option('--overwrite', is_flag=True, help='Overwrite the existing transcriptions.csv file')
def batch_infer(dataset, model, round_midi, csv, overwrite):
    data_path = pathlib.Path(dataset)
    model_path = pathlib.Path(model)
    csv_path = pathlib.Path(csv) if csv is not None else data_path / 'transcriptions.csv'
    if csv_path.exists() and not overwrite:
        raise FileExistsError(f'The CSV path \'{csv_path}\' already exists. Please re-try with --overwrite option.')
    infer_ins, config = model_init(model_path)

    # count = 0
    csv_data: List[dict] = []
    with open(f'{data_path}/transcriptions.csv', 'r', encoding='utf8', newline='') as f:
        reader = DictReader(f)
        for row in reader:
            csv_data.append(row)

    for row in tqdm.tqdm(csv_data):
        audio_path = data_path / 'wavs' / f"{row['name']}.wav"
        if not audio_path.exists():
            print(f'WARNING: audio file does not exist: \'{audio_path}\'')
            continue
        # print(f"\r\n{audio_path}: start")
        result = infer(audio_path, infer_ins, config)

        ph_dur = [round(float(x), 6) for x in row['ph_dur'].split(" ")]
        ph_num = [int(x) for x in row['ph_num'].split(" ")]
        note_seq = []
        note_dur = []

        midi_dur_list = get_word_durs(ph_dur, ph_num)
        result = midi_align(result, midi_dur_list)

        for (start_time, end_time) in midi_dur_list:
            word_duration = round(end_time - start_time, 6)
            if round_midi:
                match_seq = get_max_overlap_midi((start_time, end_time), result)
                note_seq.append(match_seq)
                note_dur.append(word_duration)
            else:
                temp_seq = []
                temp_dur = []
                match_midi = get_all_overlap_midis((start_time, end_time), result)

                for midi in match_midi:
                    if midi['start_time'] <= start_time:
                        temp_seq.append(midi['note_seq'])
                        midi_dur = round(min(end_time, midi['end_time']) - start_time, 6)
                    elif midi['end_time'] >= end_time:
                        temp_seq.append(midi['note_seq'])
                        midi_dur = round(end_time - max(start_time, midi['start_time']), 6)
                    elif midi['start_time'] <= start_time and midi['end_time'] >= end_time:
                        temp_seq.append(midi['note_seq'])
                        midi_dur = word_duration
                    else:
                        temp_seq.append(midi['note_seq'])
                        midi_dur = round(midi['note_dur'], 6)
                    temp_dur.append(midi_dur)

                if not match_midi:
                    temp_seq.append('rest')
                    temp_dur.append(word_duration)

                if round(sum(temp_dur), 6) < word_duration:
                    temp_seq.append('rest')
                    temp_dur.append(word_duration - round(sum(temp_dur), 6))

                note_seq.extend(temp_seq)
                note_dur.extend(temp_dur)

        assert len(note_seq) == len(note_dur)
        row['note_seq'] = " ".join([str(x) for x in note_seq])
        row['note_dur'] = " ".join([str(round(x, 6)) for x in note_dur])
        # print(f" {audio_path}:\r\nnote_seq: {note_seq}\r\nnote_dur: {note_dur}")
        # count += 1

    with open(csv_path, 'w', encoding='utf8', newline='') as f:
        writer = DictWriter(f, fieldnames=['name', 'ph_seq', 'ph_dur', 'ph_num', 'note_seq', 'note_dur'])
        writer.writeheader()
        writer.writerows(csv_data)


if __name__ == "__main__":
    batch_infer()
