import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MultipleLocator

LINE_COLORS = ['w', 'r', 'y', 'cyan', 'm', 'b', 'lime']


def spec_to_figure(spec, vmin=None, vmax=None):
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    fig = plt.figure(figsize=(12, 15))
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    plt.tight_layout()
    return fig


def dur_to_figure(dur_gt, dur_pred, txt):
    if isinstance(dur_gt, torch.Tensor):
        dur_gt = dur_gt.cpu().numpy()
    if isinstance(dur_pred, torch.Tensor):
        dur_pred = dur_pred.cpu().numpy()
    dur_gt = dur_gt.astype(np.int64)
    dur_pred = dur_pred.astype(np.int64)
    dur_gt = np.cumsum(dur_gt)
    dur_pred = np.cumsum(dur_pred)
    width = max(12, min(48, len(txt) // 2))
    fig = plt.figure(figsize=(width, 8))
    plt.vlines(dur_pred, 12, 22, colors='r', label='pred')
    plt.vlines(dur_gt, 0, 10, colors='b', label='gt')
    for i in range(len(txt)):
        shift = (i % 8) + 1
        plt.text((dur_pred[i - 1] + dur_pred[i]) / 2 if i > 0 else dur_pred[i] / 2, 12 + shift, txt[i],
                 size=16, horizontalalignment='center')
        plt.text((dur_gt[i - 1] + dur_gt[i]) / 2 if i > 0 else dur_gt[i] / 2, shift, txt[i],
                 size=16, horizontalalignment='center')
        plt.plot([dur_pred[i], dur_gt[i]], [12, 10], color='black', linewidth=2, linestyle=':')
    plt.yticks([])
    plt.xlim(0, max(dur_pred[-1], dur_gt[-1]))
    fig.legend()
    fig.tight_layout()
    return fig


def boundary_to_figure(
        bounds_gt: np.ndarray, bounds_pred: np.ndarray,
        dur_gt: np.ndarray = None, dur_pred: np.ndarray = None
):
    fig = plt.figure(figsize=(12, 6))
    bounds_acc_gt = np.cumsum(bounds_gt)
    bounds_acc_pred = np.cumsum(bounds_pred)
    plt.plot(bounds_acc_gt, color='b', label='gt')
    plt.plot(bounds_acc_pred, color='r', label='pred')
    if dur_gt is not None and dur_pred is not None:
        height = math.ceil(max(bounds_acc_gt[-1], bounds_acc_pred[-1]))
        dur_acc_gt = np.cumsum(dur_gt)
        dur_acc_pred = np.cumsum(dur_pred)
        plt.vlines(dur_acc_gt[:-1], 0, height / 2, colors='b', linestyles='--')
        plt.vlines(dur_acc_pred[:-1], height / 2, height, colors='r', linestyles='--')
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))
    plt.grid(axis='y')
    plt.legend()
    plt.tight_layout()
    return fig


def pitch_notes_to_figure(
        pitch, note_midi_gt, note_dur_gt, note_rest_gt,
        note_midi_pred=None, note_dur_pred=None, note_rest_pred=None
):
    fig = plt.figure()

    def draw_notes(note_midi, note_dur, note_rest, color, label):
        note_dur_acc = np.cumsum(note_dur)
        if note_rest is None:
            note_rest = np.zeros_like(note_midi, dtype=np.bool_)
        labeled = False
        for i in range(len(note_midi)):
            if note_rest[i]:
                continue
            x0 = note_dur_acc[i - 1] if i > 0 else 0
            y0 = note_midi[i] - 0.5
            rec = plt.Rectangle(
                xy=(x0, y0),
                width=note_dur[i], height=1,
                edgecolor=color, fill=False,
                linewidth=1.5, label=label if not labeled else None,
                # linestyle='--' if note_rest[i] else '-'
            )
            plt.gca().add_patch(rec)
            plt.fill_between([x0, x0 + note_dur[i]], y0, y0 + 1, color='none', facecolor=color, alpha=0.2)
            labeled = True

    draw_notes(note_midi_gt, note_dur_gt, note_rest_gt, color='b', label='gt')
    draw_notes(note_midi_pred, note_dur_pred, note_rest_pred, color='r', label='pred')
    plt.plot(pitch, color='grey', label='pitch')
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))
    plt.grid(axis='y')
    plt.legend()
    plt.tight_layout()
    return fig


def curve_to_figure(curve_gt, curve_pred=None, curve_base=None, grid=None, base_label='base'):
    if isinstance(curve_gt, torch.Tensor):
        curve_gt = curve_gt.cpu().numpy()
    if isinstance(curve_pred, torch.Tensor):
        curve_pred = curve_pred.cpu().numpy()
    if isinstance(curve_base, torch.Tensor):
        curve_base = curve_base.cpu().numpy()
    fig = plt.figure()
    if curve_base is not None:
        plt.plot(curve_base, color='grey', label=base_label)
    plt.plot(curve_gt, color='b', label='gt')
    if curve_pred is not None:
        plt.plot(curve_pred, color='r', label='pred')
    if grid is not None:
        plt.gca().yaxis.set_major_locator(MultipleLocator(grid))
    plt.grid(axis='y')
    plt.legend()
    plt.tight_layout()
    return fig


def distribution_to_figure(title, x_label, y_label, items: list, values: list, zoom=0.8):
    fig = plt.figure(figsize=(int(len(items) * zoom), 10))
    plt.bar(x=items, height=values)
    plt.tick_params(labelsize=15)
    plt.xlim(-1, len(items))
    for a, b in zip(items, values):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=15)
    plt.grid()
    plt.title(title, fontsize=30)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    return fig
