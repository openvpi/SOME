import pathlib
from collections import OrderedDict

from librosa.filters import mel
import torch
from torch import nn

from utils import build_object_from_class_name


class BaseONNXModule(nn.Module):
    def __init__(self, config: dict, model_path: pathlib.Path, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = config
        self.model_path = model_path
        self.device = device
        self.timestep = self.config['hop_size'] / self.config['audio_sample_rate']
        self.model: torch.nn.Module = self.build_model()

    def build_model(self) -> nn.Module:
        model: nn.Module = build_object_from_class_name(
            self.config['model_cls'], nn.Module, config=self.config
        ).eval().to(self.device)
        state_dict = torch.load(self.model_path, map_location=self.device)['state_dict']
        prefix_in_ckpt = 'model'
        state_dict = OrderedDict({
            k[len(prefix_in_ckpt) + 1:]: v
            for k, v in state_dict.items() if k.startswith(f'{prefix_in_ckpt}.')
        })
        model.load_state_dict(state_dict, strict=True)
        print(f'| load \'{prefix_in_ckpt}\' from \'{self.model_path}\'.')
        return model


class MelSpectrogram_ONNX(nn.Module):
    def __init__(
            self,
            n_mel_channels,
            sampling_rate,
            win_length,
            hop_length,
            n_fft=None,
            mel_fmin=0,
            mel_fmax=None,
            clamp=1e-5
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        mel_basis = mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def forward(self, audio, center=True):
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=audio.device),
            center=center,
            return_complex=False
        )
        magnitude = torch.sqrt(torch.sum(fft ** 2, dim=-1))
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec
