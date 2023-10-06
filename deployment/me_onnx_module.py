import pathlib

import torch

from utils.infer_utils import decode_bounds_to_alignment, decode_gaussian_blurred_probs, decode_note_sequence
from .base_onnx_module import BaseONNXModule, MelSpectrogram_ONNX


class MIDIExtractionONNXModule(BaseONNXModule):
    def __init__(self, config: dict, model_path: pathlib.Path, device=None):
        super().__init__(config, model_path, device=device)
        self.mel_extractor = MelSpectrogram_ONNX(
            n_mel_channels=self.config['units_dim'], sampling_rate=self.config['audio_sample_rate'],
            win_length=self.config['win_size'], hop_length=self.config['hop_size'],
            mel_fmin=self.config['fmin'], mel_fmax=self.config['fmax']
        ).to(self.device)
        self.rmvpe = None
        self.midi_min = self.config['midi_min']
        self.midi_max = self.config['midi_max']
        self.midi_deviation = self.config['midi_prob_deviation']
        self.rest_threshold = self.config['rest_threshold']

    def forward(self, waveform: torch.Tensor):
        units = self.mel_extractor(waveform).transpose(1, 2)
        pitch = torch.zeros(units.shape[:2], dtype=torch.float32, device=self.device)
        masks = torch.ones_like(pitch, dtype=torch.bool)
        probs, bounds = self.model(x=units, f0=pitch, mask=masks, sig=True)
        probs *= masks[..., None]
        bounds *= masks
        unit2note_pred = decode_bounds_to_alignment(bounds, use_diff=False) * masks
        midi_pred, rest_pred = decode_gaussian_blurred_probs(
            probs, vmin=self.midi_min, vmax=self.midi_max,
            deviation=self.midi_deviation, threshold=self.rest_threshold
        )
        note_midi_pred, note_dur_pred, note_mask_pred = decode_note_sequence(
            unit2note_pred, midi_pred, ~rest_pred & masks
        )
        note_rest_pred = ~note_mask_pred
        return note_midi_pred, note_rest_pred, note_dur_pred * self.timestep
