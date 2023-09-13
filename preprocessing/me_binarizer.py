import csv
import os
import pathlib

import librosa
import numpy as np
import torch

import modules.contentvec
from modules.commons import LengthRegulator
from utils.binarizer_utils import get_mel2ph_torch
from utils.plot import distribution_to_figure
from .base_binarizer import BaseBinarizer

os.environ["OMP_NUM_THREADS"] = "1"
MIDI_EXTRACTION_ITEM_ATTRIBUTES = [
    'units',  # contentvec units, float32[T_s, 256]
    'pitch',  # actual pitch in semitones, float32[T_s,]
    'midi_prob',
    'midi_sep',
    'note_midi',  # note-level MIDI pitch, float32[T_n,]
    'note_dur',  # durations of notes, in number of frames, int64[T_n,]
    'note_rest',  # flags for rest notes, bool[T_n,]
]
contentvec = modules.contentvec.Audio2ContentVec()


class MIDIExtractionBinarizer(BaseBinarizer):
    def __init__(self, config: dict):
        super().__init__(config, data_attrs=MIDI_EXTRACTION_ITEM_ATTRIBUTES)
        self.lr = LengthRegulator().to(self.device)

    def load_meta_data(self, raw_data_dir: pathlib.Path, ds_id):
        meta_data_dict = {}
        if (raw_data_dir / 'transcriptions.csv').exists():
            for utterance_label in csv.DictReader(
                    open(raw_data_dir / 'transcriptions.csv', 'r', encoding='utf-8')
            ):
                item_name = utterance_label['name']
                temp_dict = {
                    'wav_fn': str(raw_data_dir / 'wavs' / f'{item_name}.wav'),
                    'note_seq': utterance_label['note_seq'].split(),
                    'note_dur': [float(x) for x in utterance_label['note_dur'].split()]
                }
                assert len(temp_dict['note_seq']) == len(temp_dict['note_dur']), \
                    f'Lengths of note_seq and note_dur mismatch in \'{item_name}\'.'
                meta_data_dict[f'{ds_id}:{item_name}'] = temp_dict
        else:
            raise FileNotFoundError(
                f'transcriptions.csv not found in {raw_data_dir}.'
            )
        self.items.update(meta_data_dict)

    def check_coverage(self):
        super().check_coverage()
        # MIDI pitch distribution summary
        midi_map = {}
        for item_name in self.items:
            for midi in self.items[item_name]['note_seq']:
                if midi == 'rest':
                    continue
                midi = librosa.note_to_midi(midi, round_midi=True)
                if midi in midi_map:
                    midi_map[midi] += 1
                else:
                    midi_map[midi] = 1

        print('===== MIDI Pitch Distribution Summary =====')
        for i, key in enumerate(sorted(midi_map.keys())):
            if i == len(midi_map) - 1:
                end = '\n'
            elif i % 10 == 9:
                end = ',\n'
            else:
                end = ', '
            print(f'\'{librosa.midi_to_note(key, unicode=False)}\': {midi_map[key]}', end=end)

        # Draw graph.
        midis = sorted(midi_map.keys())
        notes = [librosa.midi_to_note(m, unicode=False) for m in range(midis[0], midis[-1] + 1)]
        plt = distribution_to_figure(
            title='MIDI Pitch Distribution Summary',
            x_label='MIDI Key', y_label='Number of occurrences',
            items=notes, values=[midi_map.get(m, 0) for m in range(midis[0], midis[-1] + 1)]
        )
        filename = self.binary_data_dir / 'midi_distribution.jpg'
        plt.savefig(fname=filename,
                    bbox_inches='tight',
                    pad_inches=0.25)
        print(f'| save summary to \'{filename}\'')

    @torch.no_grad()
    def process_item(self, item_name, meta_data, binarization_args):
        waveform, _ = librosa.load(meta_data['wav_fn'], sr=self.config['audio_sample_rate'], mono=True)
        seconds = length * self.config['hop_size'] / self.config['audio_sample_rate']
        processed_input = {
            'name': item_name,
            'wav_fn': meta_data['wav_fn'],
            'seconds': seconds,
            'length': length,
        }

        # get ground truth dur
        processed_input['mel2ph'] = get_mel2ph_torch(
            self.lr, torch.from_numpy(processed_input['ph_dur']), length, self.timestep, device=self.device
        ).cpu().numpy()

        # get ground truth f0
        global pitch_extractor
        gt_f0, uv = pitch_extractor.get_pitch(
            wav, length, hparams, interp_uv=hparams['interp_uv']
        )
        if uv.all():  # All unvoiced
            print(f'Skipped \'{item_name}\': empty gt f0')
            return None
        processed_input['f0'] = gt_f0.astype(np.float32)

        return processed_input

    def arrange_data_augmentation(self, data_iterator):
        return {}
