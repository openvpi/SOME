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


