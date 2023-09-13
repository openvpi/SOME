import torch
import torch.nn as nn

class midi_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cut_prob_loss=nn.BCELoss()
        # self.midiloss
        #todo
    def forward(self,x):
        midiout, cutp=x
        cutploss=self.cut_prob_loss(x)





class midi_conform(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.model=midi_conform(**config['midi_extractor_args'])

    def forward(self,x,f0,mask=None):
        return self.model(x,f0,mask)
    def get_loss(self):
        pass
    #todo