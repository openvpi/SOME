import copy
import os
import random

import librosa
import torch

import modules.contentvec
import modules.rmvpe
from .me_binarizer import MIDIExtractionBinarizer

os.environ["OMP_NUM_THREADS"] = "1"
QUANTIZED_MIDI_EXTRACTION_ITEM_ATTRIBUTES = [
    'units',  # contentvec units, float32[T_s, 256]
    'pitch',  # actual pitch in semitones, float32[T_s,]
    'note_midi',  # note-level MIDI pitch (0-127: MIDI, 128: rest) int64[T_n,]
    'note_dur',  # durations of notes, in number of frames, int64[T_n,]
    'unit2note',  # mel2ph format for alignment between units and notes
]


class QuantizedMIDIExtractionBinarizer(MIDIExtractionBinarizer):
    def __init__(self, config: dict):
        super().__init__(config)
        self.round_midi = True
        self.data_attrs = QUANTIZED_MIDI_EXTRACTION_ITEM_ATTRIBUTES

    def process_item(self, item_name, meta_data, allow_aug=False):
        waveform, _ = librosa.load(meta_data['wav_fn'], sr=self.config['audio_sample_rate'], mono=True)

        processed_input = self._process_item(waveform, meta_data, int_midi=True)
        processed_input['note_midi'][processed_input['note_rest']] = 128
        items = [processed_input]
        if not allow_aug:
            return items

        from .me_binarizer import mel_spec
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
            processed_input_aug['note_midi'][~processed_input_aug['note_rest']] += key_shift
            items.append(processed_input_aug)

        return items
