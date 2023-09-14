import torch
import torchmetrics
from torch import Tensor


class MIDIAccuracy(torchmetrics.Metric):
    def __init__(self, *, tolerance, **kwargs):
        super().__init__(**kwargs)
        self.tolerance = tolerance
        self.add_state('correct', default=torch.tensor(0, dtype=torch.int), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0, dtype=torch.int), dist_reduce_fx='sum')

    def update(self, midi_pred: Tensor, rest_pred: Tensor, midi_gt: Tensor, rest_gt: Tensor, mask=None) -> None:
        """

        :param midi_pred: predicted MIDI
        :param rest_pred: predict rest flags
        :param midi_gt: reference MIDI
        :param rest_gt: reference rest flags
        :param mask: valid or non-padding mask
        """
        assert midi_gt.shape == rest_gt.shape == midi_pred.shape == rest_pred.shape, \
            (f'shapes of pred and gt mismatch: '
             f'{midi_pred.shape}, {rest_pred.shape}, {midi_gt.shape}, {rest_gt.shape}')
        if mask is not None:
            assert midi_gt.shape == mask.shape, \
                f'shapes of pred, target and mask mismatch: {midi_pred.shape}, {rest_pred.shape}, {mask.shape}'
        midi_close = ~rest_pred & ~rest_gt & (torch.abs(midi_pred - midi_gt) <= self.tolerance)
        rest_correct = rest_pred == rest_gt
        overall = midi_close & rest_correct
        if mask is not None:
            overall &= mask

        self.correct += overall.sum()
        self.total += midi_gt.numel() if mask is None else mask.sum()

    def compute(self) -> Tensor:
        return self.correct / self.total
