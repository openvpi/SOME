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
    def build_model(self):

        model = build_object_from_class_name(self.config['model_cls'], nn.Module, config=self.config)

        return model

    def run_model(self, sample, infer=False):
        """
        steps:
            1. run the full model
            2. calculate losses if not infer
        """
        spec = sample['units']  # [B, T_ph]
        # target = (sample['probs'],sample['bounds'])  # [B, T_s, M]
        mask = sample['unit2note'] > 0
        # mask=None

        f0 = sample['pitch']
        probs, bounds = self.model(x=spec, f0=f0, mask=mask,softmax=infer)

        if infer:
            return probs, bounds
        else:
            losses = {}

            if  self.cfg['use_buond_loss']:
                bound_loss = self.bound_loss(bounds, sample['bounds'])

                losses['bound_loss'] = bound_loss
            if self.cfg['use_midi_loss']:
                midi_loss = self.midi_loss(probs.transpose(1,2), sample['midi_idx'])

                losses['midi_loss'] = midi_loss

            return losses
    def build_losses_and_metrics(self):
        self.midi_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.bound_loss = modules.losses.BinaryEMDLoss(bidirectional=False)
        self.register_metric('midi_acc', modules.metrics.MIDIAccuracy(tolerance=0.5))



    def _validation_step(self, sample, batch_idx):
        raise NotImplementedError()
