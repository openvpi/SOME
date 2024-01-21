import csv
import json
import os
import pathlib

import librosa
import numpy as np
import torch
from scipy import interpolate

import modules.contentvec
import modules.rmvpe
from modules.commons import LengthRegulator
from utils.binarizer_utils import merge_rests, get_mel2ph_torch
from utils.plot import distribution_to_figure
from .base_binarizer import BaseBinarizer

os.environ["OMP_NUM_THREADS"] = "1"
MIDI_EXTRACTION_ITEM_ATTRIBUTES = [
    'waveform',  # the raw audio waveform, float32[T_wav]
    'mel',  # mel spectrogram, float32[T_s, M]
    'note_midi',  # note-level MIDI pitch, float32[T_n,]
    'note_rest',  # flags for rest notes, bool[T_n,]
    'note_dur',  # durations of notes, in number of frames, int64[T_n,]
    'word_split',  # boundary flags of words, bool[T_s,]
    'slur_split',  # splitting flags of slurs, bool[T_s,]
]

# These modules are used as global variables due to a PyTorch shared memory bug on Windows platforms.
# See https://github.com/pytorch/pytorch/issues/100358
contentvec = None
mel_spec = None
rmvpe = None


class MIDIExtractionBinarizer(BaseBinarizer):
    def __init__(self, config: dict):
        super().__init__(config, data_attrs=MIDI_EXTRACTION_ITEM_ATTRIBUTES)
        self.lr = LengthRegulator().to(self.device)
        self.skip_glide = self.binarization_args['skip_glide']
        self.merge_rest = self.binarization_args['merge_rest']
        self.merge_slur = self.binarization_args['merge_slur']
        self.slur_tolerance = self.binarization_args.get('slur_tolerance')
        self.round_midi = self.binarization_args.get('round_midi', False)
        self.key_shift_min, self.key_shift_max = self.config['key_shift_range']

    def load_meta_data(self, raw_data_dir: pathlib.Path, ds_id):
        meta_data_dict = {}
        transcriptions_path = raw_data_dir / 'transcriptions.csv'
        if not transcriptions_path.exists():
            raise FileNotFoundError(
                f'transcriptions.csv not found in {raw_data_dir}.'
            )

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
            if self.skip_glide and ds.get('note_glide') is not None and any(
                    g != 'none' for g in ds['note_glide'].split()
            ):
                print(f'Item {ds_id}:{item_name} contains glide notes. Skipping.')
                continue
            # normalize
            note_seq = [
                librosa.midi_to_note(
                    np.clip(
                        librosa.note_to_midi(n, round_midi=self.round_midi),
                        a_min=0, a_max=127
                    ),
                    cents=not self.round_midi, unicode=False
                ) if n != 'rest' else 'rest'
                for n in ds['note_seq'].split()
            ]
            note_slur = [bool(int(s)) for s in ds['note_slur'].split()]
            note_dur = [float(x) for x in ds['note_dur'].split()]

            assert len(note_seq) == len(note_slur) == len(note_dur), \
                f'Lengths of note_seq, note_slur and note_dur mismatch in \'{item_name}\'.'
            assert any([note != 'rest' for note in note_seq]), \
                f'All notes are rest in \'{item_name}\'.'

            if self.merge_rest:
                # merge continuous rest notes
                note_seq, note_dur, note_slur = merge_rests(note_seq, note_dur, note_slur)

            temp_dict['note_seq'] = note_seq
            temp_dict['note_dur'] = note_dur
            temp_dict['note_slur'] = note_slur

            meta_data_dict[f'{ds_id}:{item_name}'] = temp_dict

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
    def process_item(self, item_name, meta_data):
        waveform, _ = librosa.load(meta_data['wav_fn'], sr=self.config['audio_sample_rate'], mono=True)

        wav_tensor = torch.from_numpy(waveform).to(self.device)
        # extract mel spectrogram
        global mel_spec
        if mel_spec is None:
            mel_spec = modules.rmvpe.MelSpectrogram(
                n_mel_channels=self.config['audio_num_mel_bins'],
                sampling_rate=self.config['audio_sample_rate'],
                win_length=self.config['win_size'], hop_length=self.config['hop_size'],
                mel_fmin=self.config['fmin'], mel_fmax=self.config['fmax'], clamp=1e-9
            ).to(self.device)
        mel = mel_spec(wav_tensor.unsqueeze(0)).transpose(1, 2).squeeze(0).cpu().numpy()
        length = mel.shape[0]
        seconds = length * self.config['hop_size'] / self.config['audio_sample_rate']
        processed_input = {
            'seconds': seconds,
            'length': length,
            'waveform': waveform,
            'mel': mel,
        }

        note_midi = np.array(
            [(librosa.note_to_midi(n, round_midi=False) if n != 'rest' else -1) for n in meta_data['note_seq']],
            dtype=np.float32
        )
        note_rest = note_midi < 0
        interp_func = interpolate.interp1d(
            np.where(~note_rest)[0], note_midi[~note_rest],
            kind='nearest', fill_value='extrapolate'
        )
        note_midi[note_rest] = interp_func(np.where(note_rest)[0])
        processed_input['note_midi'] = note_midi
        processed_input['note_rest'] = note_rest

        note_dur_sec = torch.FloatTensor(meta_data['note_dur']).to(self.device)
        mel2note = get_mel2ph_torch(
            self.lr, note_dur_sec, processed_input['length'], self.timestep, device=self.device
        )
        note_dur = mel2note.new_zeros(note_dur_sec.shape[0] + 1).scatter_add(
            0, mel2note, torch.ones_like(mel2note)
        )[1:]
        processed_input['note_dur'] = note_dur.cpu().numpy()

        note_slur = torch.BoolTensor(meta_data['note_slur']).to(self.device)
        note2word = torch.cumsum(~note_slur, dim=0)
        word_dur = note_dur.new_zeros(note2word.max() + 1).scatter_add(
            0, note2word, note_dur
        )[1:]  # slice, because note2word starts from 1
        mel2word = self.lr(word_dur[None])[0]
        word_split = torch.diff(
            mel2word, dim=0, prepend=mel2word.new_ones(1)
        ).gt(0)
        slur_acc = torch.cumsum(note_slur, dim=0)
        non_slur_dur = slur_acc.new_zeros(slur_acc.max() + 1).scatter_add(
            0, slur_acc, note_dur
        )  # do not slice, because slur_acc starts from 0
        mel2slur = self.lr(non_slur_dur[None])[0]
        slur_split = torch.diff(
            mel2slur, dim=0, prepend=mel2slur.new_ones(1)
        ).gt(0)
        processed_input['word_split'] = word_split.cpu().numpy()
        processed_input['slur_split'] = slur_split.cpu().numpy()

        return processed_input
