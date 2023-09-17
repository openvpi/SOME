import copy
import csv
import json
import os
import pathlib
import random

import librosa
import torch

import modules.contentvec
import modules.rmvpe
from .me_binarizer import MIDIExtractionBinarizer, mel_spec

os.environ["OMP_NUM_THREADS"] = "1"
QUANTIZED_MIDI_EXTRACTION_ITEM_ATTRIBUTES = [
    'units',  # contentvec units, float32[T_s, 256]
    'pitch',  # actual pitch in semitones, float32[T_s,]
    'note_midi',  # note-level MIDI pitch, int64[T_n,]
    'note_rest',  # flags for rest notes, bool[T_n,]
    'note_dur',  # durations of notes, in number of frames, int64[T_n,]
    'unit2note',  # mel2ph format for alignment between units and notes
]


class QuantizedMIDIExtractionBinarizer(MIDIExtractionBinarizer):
    def load_meta_data(self, raw_data_dir: pathlib.Path, ds_id):
        meta_data_dict = {}
        if (raw_data_dir / 'transcriptions.csv').exists():
            for utterance_label in csv.DictReader(
                    open(raw_data_dir / 'transcriptions.csv', 'r', encoding='utf-8')
            ):
                item_name = utterance_label['name']
                temp_dict = {
                    'wav_fn': str(raw_data_dir / 'wavs' / f'{item_name}.wav')
                }
                ds_path = raw_data_dir / 'wavs' / f'{item_name}.ds'
                with open(ds_path, 'r', encoding='utf8') as f:
                    ds = json.load(f)
                    if isinstance(ds, list):
                        ds = ds[0]
                note_seq = [
                    librosa.midi_to_note(
                        librosa.note_to_midi(n, round_midi=True), unicode=False
                    ) if n != 'rest' else 'rest'
                    for n in ds['note_seq'].split()
                ]
                note_slur = [bool(int(s)) for s in ds['note_slur'].split()]
                note_dur = [float(x) for x in ds['note_dur'].split()]
                assert len(note_seq) == len(note_slur) == len(note_dur), \
                    f'Lengths of note_seq, note_slur and note_dur mismatch in \'{item_name}\'.'
                assert any([note != 'rest' for note in note_seq]), \
                    f'All notes are rest in \'{item_name}\'.'

                # merge slurs with the same pitch
                i = 1
                note_seq_merge_slur = [note_seq[0]]
                note_dur_merge_slur = [note_dur[0]]
                for i in range(1, len(note_seq)):
                    if note_slur[i] and note_seq[i] == note_seq[i - 1]:
                        note_dur[-1] += note_dur[i]
                    else:
                        note_seq_merge_slur.append(note_seq[i])
                        note_dur_merge_slur.append(note_dur[i])

                # merge continuous rest notes
                i = 0
                note_seq_merge_rest = []
                note_dur_merge_rest = []
                while i < len(note_seq_merge_slur):
                    if note_seq_merge_slur[i] != 'rest':
                        note_seq_merge_rest.append(note_seq_merge_slur[i])
                        note_dur_merge_rest.append(note_dur_merge_slur[i])
                        i += 1
                    else:
                        j = i
                        rest_dur = 0
                        while j < len(note_seq_merge_slur) and note_seq_merge_slur[j] == 'rest':
                            rest_dur += note_dur_merge_slur[j]
                            j += 1
                        note_seq_merge_rest.append('rest')
                        note_dur_merge_rest.append(rest_dur)
                        i = j
                temp_dict['note_seq'] = note_seq_merge_rest
                temp_dict['note_dur'] = note_dur_merge_rest

                meta_data_dict[f'{ds_id}:{item_name}'] = temp_dict
        else:
            raise FileNotFoundError(
                f'transcriptions.csv not found in {raw_data_dir}.'
            )
        self.items.update(meta_data_dict)

    def process_item(self, item_name, meta_data, allow_aug=False):
        waveform, _ = librosa.load(meta_data['wav_fn'], sr=self.config['audio_sample_rate'], mono=True)

        processed_input = self._process_item(waveform, meta_data, round_midi=True)
        items = [processed_input]
        if not allow_aug:
            return items

        wav_tensor = torch.from_numpy(waveform).to(self.device)
        for _ in range(self.config['key_shift_factor']):
            assert mel_spec is not None, 'Units encoder must be mel if augmentation is applied!'
            key_shift = random.randint(int(self.key_shift_min), int(self.key_shift_max))
            processed_input_aug = copy.deepcopy(processed_input)
            assert isinstance(mel_spec, modules.rmvpe.MelSpectrogram)
            processed_input_aug['units'] = mel_spec(
                wav_tensor.unsqueeze(0), keyshift=key_shift
            ).transpose(1, 2).squeeze(0).cpu().numpy()
            processed_input_aug['pitch'] += key_shift
            processed_input_aug['note_midi'] += key_shift
            items.append(processed_input_aug)

        return items
