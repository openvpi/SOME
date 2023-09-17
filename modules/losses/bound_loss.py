from math import sqrt

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
            loss += self.loss(pred.flip(1).cumsum(dim=1), gt.flip(1).cumsum(dim=1))
            loss /= 2
        loss = loss / (sqrt(len(gt[0])))
        return loss


class BoundaryLoss(torch.nn.Module):
    def __init__(self, lambda_bce=0.1):
        super().__init__()
        self.emd = BinaryEMDLoss(bidirectional=False)
        self.bce = torch.nn.BCELoss()
        self.lambda_bce = lambda_bce

    def forward(self, pred, gt):
        # pred, gt: [B, T]
        emd_loss = self.emd(pred, gt)
        bce_loss = self.bce(pred, gt)
        return emd_loss + self.lambda_bce * bce_loss
