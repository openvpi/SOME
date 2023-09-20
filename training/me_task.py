import torch
import torch.nn.functional as F
from torch import nn

import modules.losses
import modules.metrics
from utils import build_object_from_class_name, collate_nd
from utils.infer_utils import decode_gaussian_blurred_probs, decode_bounds_to_alignment, decode_note_sequence
from utils.plot import boundary_to_figure, curve_to_figure, spec_to_figure, pitch_notes_to_figure
from .base_task import BaseDataset, BaseTask


class MIDIExtractionDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.midi_min = self.config['midi_min']
        self.midi_max = self.config['midi_max']
        self.num_bins = self.config['midi_num_bins']
        self.midi_deviation = self.config['midi_prob_deviation']
        self.interval = (self.midi_max - self.midi_min) / (self.num_bins - 1)  # align with centers of bins
        self.sigma = self.midi_deviation / self.interval

    def midi_to_bin(self, midi):
        return (midi - self.midi_min) / self.interval

    def collater(self, samples):
        batch = super().collater(samples)
        batch['units'] = collate_nd([s['units'] for s in samples])  # [B, T_s, C]
        batch['pitch'] = collate_nd([s['pitch'] for s in samples])  # [B, T_s]
        batch['note_midi'] = collate_nd([s['note_midi'] for s in samples])  # [B, T_n]
        batch['note_rest'] = collate_nd([s['note_rest'] for s in samples])  # [B, T_n]
        batch['note_dur'] = collate_nd([s['note_dur'] for s in samples])  # [B, T_n]

        miu = self.midi_to_bin(batch['note_midi'])[:, :, None]  # [B, T_n, 1]
        x = torch.arange(self.num_bins).float().reshape(1, 1, -1).to(miu.device)  # [1, 1, N]
        probs = ((x - miu) / self.sigma).pow(2).div(-2).exp()  # gaussian blur, [B, T_n, N]
        note_mask = collate_nd([torch.ones_like(s['note_rest']) for s in samples], pad_value=False)
        probs *= (note_mask[..., None] & ~batch['note_rest'][..., None])

        probs = F.pad(probs, [0, 0, 1, 0])
        unit2note = collate_nd([s['unit2note'] for s in samples])
        unit2note_ = unit2note[..., None].repeat([1, 1, self.num_bins])
        probs = torch.gather(probs, 1, unit2note_)
        batch['probs'] = probs  # [B, T_s, N]
        batch['unit2note'] = unit2note
        bounds = torch.diff(
            unit2note, dim=1, prepend=unit2note.new_zeros((batch['size'], 1))
        ) > 0
        batch['bounds'] = bounds.float()  # [B, T_s]

        return batch


# todo
class MIDIExtractionTask(BaseTask):
    def __init__(self, config: dict):
        super().__init__(config)
        self.midiloss = None
        self.dataset_cls = MIDIExtractionDataset
        self.midi_min = self.config['midi_min']
        self.midi_max = self.config['midi_max']
        self.midi_deviation = self.config['midi_prob_deviation']
        self.rest_threshold = self.config['rest_threshold']
        self.cfg=config

    def build_model(self):

        model = build_object_from_class_name(self.config['model_cls'], nn.Module, config=self.config)

        return model

    def build_losses_and_metrics(self):

        self.midi_loss = nn.BCEWithLogitsLoss()
        self.bound_loss = modules.losses.BinaryEMDLoss()
        # self.bound_loss = modules.losses.BinaryEMDLoss(bidirectional=True)
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




        if infer:
            probs, bounds = self.model(x=spec, f0=f0, mask=mask, sig=True)
            return probs, bounds
        else:
            losses = {}
            probs, bounds = self.model(x=spec, f0=f0, mask=mask, sig=False)

            if  self.cfg['use_bound_loss']:
                bound_loss = self.bound_loss(bounds, sample['bounds'])

                losses['bound_loss'] = bound_loss
            if self.cfg['use_midi_loss']:
                midi_loss = self.midi_loss(probs, sample['probs'])

                losses['midi_loss'] = midi_loss

            return losses

        # raise NotImplementedError()

    def _validation_step(self, sample, batch_idx):
        losses = self.run_model(sample, infer=False)
        if batch_idx < self.config['num_valid_plots']:
            probs, bounds = self.run_model(sample, infer=True)
            unit2note_gt = sample['unit2note']
            masks = unit2note_gt > 0
            probs *= masks[..., None]
            bounds *= masks
            self.plot_prob(batch_idx, sample['probs'], probs)

            unit2note_pred = decode_bounds_to_alignment(bounds) * masks
            midi_pred, rest_pred = decode_gaussian_blurred_probs(
                probs, vmin=self.midi_min, vmax=self.midi_max,
                deviation=self.midi_deviation, threshold=self.rest_threshold
            )
            note_midi_pred, note_dur_pred, note_mask_pred = decode_note_sequence(
                unit2note_pred, midi_pred, ~rest_pred & masks
            )
            note_rest_pred = ~note_mask_pred
            self.plot_boundary(
                batch_idx, bounds_gt=sample['bounds'], bounds_pred=bounds,
                dur_gt=sample['note_dur'], dur_pred=note_dur_pred
            )
            self.plot_final(
                batch_idx, sample['note_midi'], sample['note_dur'], sample['note_rest'],
                note_midi_pred, note_dur_pred, note_rest_pred, sample['pitch']
            )

            midi_pred[rest_pred] = -torch.inf  # rest part is set to -inf
            note_midi_gt = sample['note_midi'].clone()
            note_midi_gt[sample['note_rest']] = -torch.inf
            midi_gt = torch.gather(F.pad(note_midi_gt, [1, 0], value=-torch.inf), 1, unit2note_gt)
            self.plot_midi_curve(
                batch_idx, midi_gt=midi_gt, midi_pred=midi_pred, pitch=sample['pitch']
            )
            self.midi_acc.update(
                midi_pred=midi_pred, rest_pred=rest_pred, midi_gt=midi_gt, rest_gt=midi_gt < 0, mask=masks
            )

        return losses, sample['size']

    ############
    # validation plots
    ############
    def plot_prob(self, batch_idx, probs_gt, probs_pred):
        name = f'prob/{batch_idx}'
        vmin, vmax = 0, 1
        spec_cat = torch.cat([(probs_pred - probs_gt).abs() + vmin, probs_gt, probs_pred], -1)
        self.logger.experiment.add_figure(name, spec_to_figure(spec_cat[0], vmin, vmax), self.global_step)

    def plot_boundary(self, batch_idx, bounds_gt, bounds_pred, dur_gt, dur_pred):
        name = f'boundary/{batch_idx}'
        bounds_gt = bounds_gt[0].cpu().numpy()
        bounds_pred = bounds_pred[0].cpu().numpy()
        dur_gt = dur_gt[0].cpu().numpy()
        dur_pred = dur_pred[0].cpu().numpy()
        self.logger.experiment.add_figure(name, boundary_to_figure(
            bounds_gt, bounds_pred, dur_gt, dur_pred
        ), self.global_step)

    def plot_midi_curve(self, batch_idx, midi_gt, midi_pred, pitch):
        name = f'midi/{batch_idx}'
        midi_gt = midi_gt[0].cpu().numpy()
        midi_pred = midi_pred[0].cpu().numpy()
        pitch = pitch[0].cpu().numpy()
        self.logger.experiment.add_figure(name, curve_to_figure(
            midi_gt, midi_pred, curve_base=pitch, grid=1, base_label='pitch'
        ), self.global_step)

    def plot_final(self, batch_idx, midi_gt, dur_gt, rest_gt, midi_pred, dur_pred, rest_pred, pitch):
        name = f'final/{batch_idx}'
        midi_gt = midi_gt[0].cpu().numpy()
        midi_pred = midi_pred[0].cpu().numpy()
        dur_gt = dur_gt[0].cpu().numpy()
        dur_pred = dur_pred[0].cpu().numpy()
        rest_gt = rest_gt[0].cpu().numpy()
        rest_pred = rest_pred[0].cpu().numpy()
        pitch = pitch[0].cpu().numpy()
        self.logger.experiment.add_figure(name, pitch_notes_to_figure(
            pitch=pitch, note_midi_gt=midi_gt, note_dur_gt=dur_gt, note_rest_gt=rest_gt,
            note_midi_pred=midi_pred, note_dur_pred=dur_pred, note_rest_pred=rest_pred
        ), self.global_step)
