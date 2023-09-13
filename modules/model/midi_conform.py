import torch
import torch.nn as nn

class midi_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss=nn.BCELoss()


    def forward(self,x,target):
        midiout, cutp=x
        midi_target, cutp_target = target

        cutploss=self.loss(cutp,cutp_target)
        midiloss = self.loss(midiout, midi_target)
        return midiloss,cutploss





class midi_conform(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.model=midi_conform(**config['midi_extractor_args'])

    def forward(self,x,f0,mask=None):
        return self.model(x,f0,mask)
    def get_loss(self):
        return midi_loss()
