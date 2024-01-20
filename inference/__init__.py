from .base_infer import BaseInference
from .me_infer import MIDIExtractionInference

task_inference_mapping = {
    'training.MIDIExtractionTask': 'inference.MIDIExtractionInference',
}
