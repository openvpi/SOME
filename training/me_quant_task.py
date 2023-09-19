import torch
import torch.nn.functional as F
from torch import nn

import modules.losses
import modules.metrics
from utils import build_object_from_class_name, collate_nd
from utils.infer_utils import decode_bounds_to_alignment, decode_note_sequence
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
        return batch


class QuantizedMIDIExtractionTask(MIDIExtractionTask):
    def __init__(self, config: dict):
        super().__init__(config)
        self.dataset_cls = QuantizedMIDIExtractionDataset
        self.config = config

    # noinspection PyAttributeOutsideInit
    def build_model(self):

        model = build_object_from_class_name(self.config['model_cls'], nn.Module, config=self.config)

        return model

    def build_losses_and_metrics(self):
        self.midi_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.bound_loss = modules.losses.BinaryEMDLoss(bidirectional=False)
        self.register_metric('midi_acc', modules.metrics.MIDIAccuracy(tolerance=0.5))

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
        probs, bounds = self.model(x=spec, f0=f0, mask=mask, softmax=infer)

        if infer:
            return probs, bounds
        else:
            losses = {}

            if  self.cfg['use_bound_loss']:
                bound_loss = self.bound_loss(bounds, sample['bounds'])

                losses['bound_loss'] = bound_loss
            if self.cfg['use_midi_loss']:
                midi_loss = self.midi_loss(probs.transpose(1, 2), sample['midi_idx'])

                losses['midi_loss'] = midi_loss

            return losses

    def _validation_step(self, sample, batch_idx):
        losses = self.run_model(sample, infer=False)
        if batch_idx < self.config['num_valid_plots']:
            probs, bounds = self.run_model(sample, infer=True)
            unit2note_gt = sample['unit2note']
            masks = unit2note_gt > 0
            probs *= masks[..., None]
            bounds *= masks
            # probs: [B, T, 129] => [B, T, 128]
            probs_pred = probs[:, :, :-1]
            probs_gt = F.one_hot(sample['midi_idx'], num_classes=129)[:, :, :-1]
            self.plot_prob(batch_idx, probs_gt, probs_pred)

            unit2note_pred = decode_bounds_to_alignment(bounds) * masks
            midi_pred = probs.argmax(dim=-1)
            rest_pred = midi_pred == 128
            note_midi_pred, note_dur_pred, note_mask_pred = decode_note_sequence(
                unit2note_pred, midi_pred.clip(min=0, max=127), ~rest_pred & masks
            )
            note_rest_pred = ~note_mask_pred
            self.plot_boundary(
                batch_idx, bounds_gt=sample['bounds'], bounds_pred=bounds,
                dur_gt=sample['note_dur'], dur_pred=note_dur_pred
            )
            self.plot_final(
                batch_idx, sample['note_midi'], sample['note_dur'], sample['note_midi'] == 128,
                note_midi_pred, note_dur_pred, note_rest_pred, sample['pitch']
            )

            midi_pred = midi_pred.float()
            midi_pred[rest_pred] = -torch.inf  # rest part is set to -inf
            note_midi_gt = sample['note_midi'].float()
            note_rest_gt = sample['note_midi'] == 128
            note_midi_gt[note_rest_gt] = -torch.inf
            midi_gt = torch.gather(F.pad(note_midi_gt, [1, 0], value=-torch.inf), 1, unit2note_gt)
            self.plot_midi_curve(
                batch_idx, midi_gt=midi_gt, midi_pred=midi_pred, pitch=sample['pitch']
            )
            self.midi_acc.update(
                midi_pred=midi_pred, rest_pred=rest_pred, midi_gt=midi_gt, rest_gt=midi_gt < 0, mask=masks
            )

        return losses, sample['size']
