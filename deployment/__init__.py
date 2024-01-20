from .base_onnx_module import BaseONNXModule
from .me_onnx_module import MIDIExtractionONNXModule

task_module_mapping = {
    'training.MIDIExtractionTask': 'deployment.MIDIExtractionONNXModule',
}
