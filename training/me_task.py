from torch import nn

from utils import build_object_from_class_name
from .base_task import BaseTask

#todo
class MIDIExtractionTask(BaseTask):

    def __init__(self,config:dict):
        super().__init__(config)



    def build_model(self):


        model=build_object_from_class_name(self.config['model_cls'],nn.Module,config=self.config)


        return model

    def build_losses_and_metrics(self):
        self.midiloss=self.model.get_loss()

    def run_model(self, sample, infer=False):
        """
        steps:
            1. run the full model
            2. calculate losses if not infer
        """
        spec = sample['spec']  # [B, T_ph]
        target = sample['target']  # [B, T_s, M]
        mask=sample['mask']

        f0 = sample['f0']
        output=self.model(x=spec,f0=f0,mask=mask)

        if infer:
            return output
        else:
            losses = {}
            midi_loss,board_loss=self.midiloss(output,target)



            losses['board_loss'] = board_loss


            losses['midi_loss'] = midi_loss

            return losses


        # raise NotImplementedError()


