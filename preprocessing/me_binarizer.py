import csv
import os
import pathlib

import librosa
import numpy as np
from scipy import interpolate
import torch

import modules.contentvec
import modules.rmvpe
from modules.commons import LengthRegulator
from utils.binarizer_utils import get_mel2ph_torch, get_pitch_parselmouth
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
rmvpe = None


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
        wav_tensor = torch.from_numpy(waveform).to(self.device)
        global contentvec
        if contentvec is None:
            contentvec = modules.contentvec.ContentVec(self.config['units_encoder_ckpt'], device=self.device)
        units = contentvec(wav_tensor).squeeze(0).cpu().numpy()
        assert len(units.shape) == 2 and units.shape[1] == self.config['units_dim'], \
            f'Shape of units must be [T, units_dim], but is {units.shape}.'
        length = units.shape[0]
        seconds = length * self.config['hop_size'] / self.config['audio_sample_rate']
        processed_input = {
            'name': item_name,
            'wav_fn': meta_data['wav_fn'],
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
                hop_size=self.config['hop_size'],
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
        note_acc = torch.round(torch.cumsum(note_dur_sec, dim=0) / self.timestep + 0.5).long()
        note_dur = torch.diff(note_acc, dim=0, prepend=torch.LongTensor([0]).to(self.device))
        processed_input['note_dur'] = note_dur.cpu().numpy()
        unit2note = get_mel2ph_torch(
            self.lr, note_dur_sec, length, self.timestep, device=self.device
        )
        processed_input['unit2note'] = unit2note.cpu().numpy()

        return [processed_input]
