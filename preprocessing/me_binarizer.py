import copy
import csv
import json
import os
import pathlib
import random

import librosa
import numpy as np
import torch
from scipy import interpolate

import modules.contentvec
import modules.rmvpe
from modules.commons import LengthRegulator
from utils.binarizer_utils import merge_slurs, merge_rests, get_mel2ph_torch, get_pitch_parselmouth
from utils.pitch_utils import resample_align_curve
from utils.plot import distribution_to_figure
from .base_binarizer import BaseBinarizer

os.environ["OMP_NUM_THREADS"] = "1"
MIDI_EXTRACTION_ITEM_ATTRIBUTES = [
    'units',  # contentvec units, float32[T_s, 256]
    'pitch',  # actual pitch in semitones, float32[T_s,]
    'note_midi',  # note-level MIDI pitch, float32[T_n,]
    'note_rest',  # flags for rest notes, bool[T_n,]
    'note_dur',  # durations of notes, in number of frames, int64[T_n,]
    'unit2note',  # mel2ph format for alignment between units and notes
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

                # if not len(note_seq) == len(note_slur) == len(note_dur):
                #     continue
                assert len(note_seq) == len(note_slur) == len(note_dur), \
                    f'Lengths of note_seq, note_slur and note_dur mismatch in \'{item_name}\'.'
                assert any([note != 'rest' for note in note_seq]), \
                    f'All notes are rest in \'{item_name}\'.'

                if self.merge_slur:
                    # merge slurs with the same pitch
                    note_seq, note_dur = merge_slurs(note_seq, note_dur, note_slur, tolerance=self.slur_tolerance)

                if self.merge_rest:
                    # merge continuous rest notes
                    note_seq, note_dur = merge_rests(note_seq, note_dur)

                temp_dict['note_seq'] = note_seq
                temp_dict['note_dur'] = note_dur

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

    def _process_item(self, waveform, meta_data, int_midi=False):
        wav_tensor = torch.from_numpy(waveform).to(self.device)
        units_encoder = self.config['units_encoder']
        if units_encoder == 'contentvec768l12':
            global contentvec
            if contentvec is None:
                contentvec = modules.contentvec.ContentVec768L12(self.config['units_encoder_ckpt'], device=self.device)
            units = contentvec(wav_tensor).squeeze(0).cpu().numpy()
        elif units_encoder == 'mel':
            global mel_spec
            if mel_spec is None:
                mel_spec = modules.rmvpe.MelSpectrogram(
                    n_mel_channels=self.config['units_dim'], sampling_rate=self.config['audio_sample_rate'],
                    win_length=self.config['win_size'], hop_length=self.config['hop_size'],
                    mel_fmin=self.config['fmin'], mel_fmax=self.config['fmax']
                ).to(self.device)
            units = mel_spec(wav_tensor.unsqueeze(0)).transpose(1, 2).squeeze(0).cpu().numpy()
        else:
            raise NotImplementedError(f'Invalid units encoder: {units_encoder}')
        assert len(units.shape) == 2 and units.shape[1] == self.config['units_dim'], \
            f'Shape of units must be [T, units_dim], but is {units.shape}.'
        length = units.shape[0]
        seconds = length * self.config['hop_size'] / self.config['audio_sample_rate']
        processed_input = {
            'seconds': seconds,
            'length': length,
            'units': units
        }

        f0_algo = self.config['pe']
        if f0_algo == 'parselmouth':
            f0, _ = get_pitch_parselmouth(
                waveform, sample_rate=self.config['audio_sample_rate'],
                hop_size=self.config['hop_size'], length=length, interp_uv=True
            )
        elif f0_algo == 'rmvpe':
            global rmvpe
            if rmvpe is None:
                rmvpe = modules.rmvpe.RMVPE(self.config['pe_ckpt'], device=self.device)
            f0, _ = rmvpe.get_pitch(
                waveform, sample_rate=self.config['audio_sample_rate'],
                hop_size=rmvpe.mel_extractor.hop_length,
                length=(waveform.shape[0] + rmvpe.mel_extractor.hop_length - 1) // rmvpe.mel_extractor.hop_length,
                interp_uv=True
            )
            f0 = resample_align_curve(
                f0,
                original_timestep=rmvpe.mel_extractor.hop_length / self.config['audio_sample_rate'],
                target_timestep=self.config['hop_size'] / self.config['audio_sample_rate'],
                align_length=length
            )
        else:
            raise NotImplementedError(f'Invalid pitch extractor: {f0_algo}')
        pitch = librosa.hz_to_midi(f0)
        processed_input['pitch'] = pitch

        note_midi = np.array(
            [(librosa.note_to_midi(n, round_midi=int_midi) if n != 'rest' else -1) for n in meta_data['note_seq']],
            dtype=np.int64 if int_midi else np.float32
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
        note_acc = torch.round(torch.cumsum(note_dur_sec, dim=0) / self.timestep + 0.5).long()
        note_dur = torch.diff(note_acc, dim=0, prepend=torch.LongTensor([0]).to(self.device))
        processed_input['note_dur'] = note_dur.cpu().numpy()
        unit2note = get_mel2ph_torch(
            self.lr, note_dur_sec, processed_input['length'], self.timestep, device=self.device
        )
        processed_input['unit2note'] = unit2note.cpu().numpy()
        return processed_input

    @torch.no_grad()
    def process_item(self, item_name, meta_data, allow_aug=False):
        waveform, _ = librosa.load(meta_data['wav_fn'], sr=self.config['audio_sample_rate'], mono=True)

        processed_input = self._process_item(waveform, meta_data, int_midi=False)
        items = [processed_input]
        if not allow_aug:
            return items

        wav_tensor = torch.from_numpy(waveform).to(self.device)
        for _ in range(self.config['key_shift_factor']):
            assert mel_spec is not None, 'Units encoder must be mel if augmentation is applied!'
            key_shift = random.random() * (self.key_shift_max - self.key_shift_min) + self.key_shift_min
            if self.round_midi:
                key_shift = round(key_shift)
            processed_input_aug = copy.deepcopy(processed_input)
            assert isinstance(mel_spec, modules.rmvpe.MelSpectrogram)
            processed_input_aug['units'] = mel_spec(
                wav_tensor.unsqueeze(0), keyshift=key_shift
            ).transpose(1, 2).squeeze(0).cpu().numpy()
            processed_input_aug['pitch'] += key_shift
            processed_input_aug['note_midi'] += key_shift
            items.append(processed_input_aug)

        return items
