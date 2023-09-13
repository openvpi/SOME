import torch.nn


class BinaryEMDLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.L1Loss()

    def forward(self, pred, gt):
        # pred, gt: [B, T]
        return self.loss(pred.cumsum(dim=1), gt.cumsum(dim=1))
