import torch
from fairseq import checkpoint_utils


class ContentVec768L12(torch.nn.Module):
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        super().__init__()
        self.device = device
        models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="")
        self.hubert = models[0].to(self.device).eval()

    def forward(self, waveform):  # B, T
        feats = waveform.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
            "source": feats.to(waveform.device),
            "padding_mask": padding_mask.to(waveform.device),
            "output_layer": 9,  # layer 9
        }
        with torch.no_grad():
            logits = self.hubert.extract_features(**inputs)
            feats = logits[0]
        units = feats  # .transpose(2, 1)
        return units
