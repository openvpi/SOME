import torch.nn


class BinaryEMDLoss(torch.nn.Module):
    def __init__(self, bidirectional=False):
        super().__init__()
        self.loss = torch.nn.L1Loss()
        self.bidirectional = bidirectional

    def forward(self, pred, gt):
        # pred, gt: [B, T]
        loss = self.loss(pred.cumsum(dim=1), gt.cumsum(dim=1))
        if self.bidirectional:
            loss += self.loss(pred.flip(dim=1).cumsum(dim=1), gt.flip(dim=1).cumsum(dim=1))
            loss /= 2
        return loss
