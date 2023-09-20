import pathlib
from typing import Dict, List

import librosa
import numpy as np
import torch

import modules.rmvpe
from utils.binarizer_utils import get_pitch_parselmouth
from utils.infer_utils import decode_bounds_to_alignment, decode_gaussian_blurred_probs, decode_note_sequence
from utils.pitch_utils import resample_align_curve
from .base_infer import BaseInference


class MIDIExtractionInference(BaseInference):
    def __init__(self, config: dict, model_path: pathlib.Path, device=None):
        super().__init__(config, model_path, device=device)
        self.mel_spec = modules.rmvpe.MelSpectrogram(
            n_mel_channels=self.config['units_dim'], sampling_rate=self.config['audio_sample_rate'],
            win_length=self.config['win_size'], hop_length=self.config['hop_size'],
            mel_fmin=self.config['fmin'], mel_fmax=self.config['fmax']
        ).to(self.device)
        self.rmvpe = None
        self.midi_min = self.config['midi_min']
        self.midi_max = self.config['midi_max']
        self.midi_deviation = self.config['midi_prob_deviation']
        self.rest_threshold = self.config['rest_threshold']

    def preprocess(self, waveform: np.ndarray) -> Dict[str, torch.Tensor]:
        wav_tensor = torch.from_numpy(waveform).unsqueeze(0).to(self.device)
        units = self.mel_spec(wav_tensor).transpose(1, 2)
        length = units.shape[1]
        # f0_algo = self.config['pe']
        # if f0_algo == 'parselmouth':
        #     f0, _ = get_pitch_parselmouth(
        #         waveform, sample_rate=self.config['audio_sample_rate'],
        #         hop_size=self.config['hop_size'], length=length, interp_uv=True
        #     )
        # elif f0_algo == 'rmvpe':
        #     if self.rmvpe is None:
        #         self.rmvpe = modules.rmvpe.RMVPE(self.config['pe_ckpt'], device=self.device)
        #     f0, _ = self.rmvpe.get_pitch(
        #         waveform, sample_rate=self.config['audio_sample_rate'],
        #         hop_size=self.rmvpe.mel_extractor.hop_length,
        #         length=(waveform.shape[0] + self.rmvpe.mel_extractor.hop_length - 1) // self.rmvpe.mel_extractor.hop_length,
        #         interp_uv=True
        #     )
        #     f0 = resample_align_curve(
        #         f0,
        #         original_timestep=self.rmvpe.mel_extractor.hop_length / self.config['audio_sample_rate'],
        #         target_timestep=self.config['hop_size'] / self.config['audio_sample_rate'],
        #         align_length=length
        #     )
        # else:
        #     raise NotImplementedError(f'Invalid pitch extractor: {f0_algo}')
        # pitch = librosa.hz_to_midi(f0)
        # pitch = torch.from_numpy(pitch).unsqueeze(0).to(self.device)
        pitch = torch.zeros(units.shape[:2], dtype=torch.float32, device=self.device)
        return {
            'units': units,
            'pitch': pitch,
            'masks': torch.ones_like(pitch, dtype=torch.bool)
        }

    @torch.no_grad()
    def forward_model(self, sample: Dict[str, torch.Tensor]):
        probs, bounds = self.model(x=sample['units'], f0=sample['pitch'], mask=sample['masks'], sig=True)

        return {
            'probs': probs,
            'bounds': bounds,
            'masks': sample['masks'],
        }

    def postprocess(self, results: Dict[str, torch.Tensor]) -> List[Dict[str, np.ndarray]]:
        probs = results['probs']
        bounds = results['bounds']
        masks = results['masks']
        probs *= masks[..., None]
        bounds *= masks
        unit2note_pred = decode_bounds_to_alignment(bounds) * masks
        midi_pred, rest_pred = decode_gaussian_blurred_probs(
            probs, vmin=self.midi_min, vmax=self.midi_max,
            deviation=self.midi_deviation, threshold=self.rest_threshold
        )
        note_midi_pred, note_dur_pred, note_mask_pred = decode_note_sequence(
            unit2note_pred, midi_pred, ~rest_pred & masks
        )
        note_rest_pred = ~note_mask_pred
        return {
            'note_midi': note_midi_pred.squeeze(0).cpu().numpy(),
            'note_dur': note_dur_pred.squeeze(0).cpu().numpy() * self.timestep,
            'note_rest': note_rest_pred.squeeze(0).cpu().numpy()
        }
