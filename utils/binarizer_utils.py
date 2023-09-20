from typing import Tuple

import librosa
import numpy as np
import parselmouth
import torch

from utils.pitch_utils import interp_f0


def merge_slurs(note_seq: list, note_dur: list, note_slur: list, tolerance=None) -> Tuple[list, list]:
    """
    merge slurs with the similar pitch
    """
    note_midi = [librosa.note_to_midi(n, round_midi=False) if n != 'rest' else 'rest' for n in note_seq]
    prev_min = prev_max = None
    note_midi_merge_slur = [note_midi[0]]
    note_dur_merge_slur = [note_dur[0]]

    def can_be_merged(midi):
        if tolerance is None or midi == 'rest' or note_midi_merge_slur[-1] == 'rest':
            return note_midi_merge_slur[-1] == midi
        return (
            abs(midi - note_midi_merge_slur[-1]) <= tolerance
            and (prev_min is None or abs(midi - prev_min) <= tolerance)
            and (prev_max is None or abs(midi - prev_max) <= tolerance)
        )

    def get_merged_midi(midi1, dur1, midi2, dur2):
        if midi1 == midi2:
            return midi1
        return (midi1 * dur1 + midi2 * dur2) / (dur1 + dur2)

    for i in range(1, len(note_seq)):
        if note_slur[i] and can_be_merged(note_midi[i]):
            # update min and max
            prev_min = min(note_midi[i], note_midi_merge_slur[-1]) if prev_min is None else min(prev_min, note_midi[i])
            prev_max = max(note_midi[i], note_midi_merge_slur[-1]) if prev_max is None else max(prev_max, note_midi[i])
            note_midi_merge_slur[-1] = get_merged_midi(
                note_midi_merge_slur[-1], note_dur_merge_slur[-1], note_midi[i], note_dur[i]
            )
            note_dur_merge_slur[-1] += note_dur[i]
        else:
            note_midi_merge_slur.append(note_midi[i])
            note_dur_merge_slur.append(note_dur[i])
            prev_min = prev_max = None
    note_seq_merge_slur = [
        librosa.midi_to_note(n, cents=True, unicode=False) if n != 'rest' else 'rest' for n in note_midi_merge_slur
    ]
    return note_seq_merge_slur, note_dur_merge_slur


def merge_rests(note_seq: list, note_dur: list) -> Tuple[list, list]:
    i = 0
    note_seq_merge_rest = []
    note_dur_merge_rest = []
    while i < len(note_seq):
        if note_seq[i] != 'rest':
            note_seq_merge_rest.append(note_seq[i])
            note_dur_merge_rest.append(note_dur[i])
            i += 1
        else:
            j = i
            rest_dur = 0
            while j < len(note_seq) and note_seq[j] == 'rest':
                rest_dur += note_dur[j]
                j += 1
            note_seq_merge_rest.append('rest')
            note_dur_merge_rest.append(rest_dur)
            i = j
    return note_seq_merge_rest, note_dur_merge_rest


@torch.no_grad()
def get_mel2ph_torch(lr, durs, length, timestep, device='cpu'):
    ph_acc = torch.round(torch.cumsum(durs.to(device), dim=0) / timestep + 0.5).long()
    ph_dur = torch.diff(ph_acc, dim=0, prepend=torch.LongTensor([0]).to(device))
    mel2ph = lr(ph_dur[None])[0]
    num_frames = mel2ph.shape[0]
    if num_frames < length:
        mel2ph = torch.cat((mel2ph, torch.full((length - num_frames,), fill_value=mel2ph[-1], device=device)), dim=0)
    elif num_frames > length:
        mel2ph = mel2ph[:length]
    return mel2ph


def pad_frames(frames, hop_size, n_samples, n_expect):
    n_frames = frames.shape[0]
    lpad = (int(n_samples // hop_size) - n_frames + 1) // 2
    rpad = n_expect - n_frames - lpad
    if rpad < 0:
        frames = frames[:rpad]
        rpad = 0
    if lpad > 0 or rpad > 0:
        frames = np.pad(frames, (lpad, rpad), mode='constant', constant_values=(frames[0], frames[-1]))
    return frames


def get_pitch_parselmouth(waveform, sample_rate, hop_size, length, interp_uv=False):
    """

    :param waveform: [T]
    :param hop_size: size of each frame
    :param sample_rate: sampling rate of waveform
    :param length: Expected number of frames
    :param interp_uv: Interpolate unvoiced parts
    :return: f0, uv
    """

    time_step = hop_size / sample_rate
    f0_min = 65
    f0_max = 800

    # noinspection PyArgumentList
    f0 = parselmouth.Sound(waveform, sampling_frequency=sample_rate).to_pitch_ac(
        time_step=time_step, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max
    ).selected_array['frequency'].astype(np.float32)
    f0 = pad_frames(f0, hop_size, waveform.shape[0], length)
    uv = f0 == 0
    if interp_uv:
        f0, uv = interp_f0(f0, uv)
    return f0, uv


class SinusoidalSmoothingConv1d(torch.nn.Conv1d):
    def __init__(self, kernel_size):
        super().__init__(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=False,
            padding='same',
            padding_mode='replicate'
        )
        smooth_kernel = torch.sin(torch.from_numpy(
            np.linspace(0, 1, kernel_size).astype(np.float32) * np.pi
        ))
        smooth_kernel /= smooth_kernel.sum()
        self.weight.data = smooth_kernel[None, None]
