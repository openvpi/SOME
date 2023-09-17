import torch
import torch.nn.functional as F
from torch import nn

import modules.losses
import modules.metrics
from utils import build_object_from_class_name, collate_nd
from utils.infer_utils import decode_gaussian_blurred_probs, decode_bounds_to_alignment, decode_note_sequence

from .base_task import BaseDataset
from .me_task import MIDIExtractionTask


class QuantizedMIDIExtractionDataset(BaseDataset):
    def collater(self, samples):
        batch = super().collater(samples)
        batch['units'] = collate_nd([s['units'] for s in samples])  # [B, T_s, C]
        batch['pitch'] = collate_nd([s['pitch'] for s in samples])  # [B, T_s]
        batch['note_midi'] = collate_nd([s['note_midi'] for s in samples], pad_value=-1)  # [B, T_n]
        batch['note_rest'] = collate_nd([s['note_rest'] for s in samples])  # [B, T_n]
        batch['note_dur'] = collate_nd([s['note_dur'] for s in samples])  # [B, T_n]
        unit2note = collate_nd([s['unit2note'] for s in samples])
        batch['unit2note'] = unit2note
        batch['midi_idx'] = torch.gather(F.pad(batch['note_midi'], [1, 0], value=-1), 1, unit2note)
        bounds = torch.diff(
            unit2note, dim=1, prepend=unit2note.new_zeros((batch['size'], 1))
        ) > 0
        batch['bounds'] = bounds.float()


class QuantizedMIDIExtractionTask(MIDIExtractionTask):
    def __init__(self, config: dict):
        super().__init__(config)
        self.dataset_cls = QuantizedMIDIExtractionDataset
        self.config = config

    # noinspection PyAttributeOutsideInit
    def build_losses_and_metrics(self):
        self.midi_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.bound_loss = modules.losses.BinaryEMDLoss(bidirectional=False)
        self.register_metric('midi_acc', modules.metrics.MIDIAccuracy(tolerance=0.5))

    def run_model(self, sample, infer=False):
        raise NotImplementedError()

    def _validation_step(self, sample, batch_idx):
        raise NotImplementedError()
