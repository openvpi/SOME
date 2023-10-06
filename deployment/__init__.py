from .base_onnx_module import BaseONNXModule
from .me_onnx_module import MIDIExtractionONNXModule
from .me_quant_onnx_module import QuantizedMIDIExtractionONNXModule

task_module_mapping = {
    'training.MIDIExtractionTask': 'deployment.MIDIExtractionONNXModule',
    'training.QuantizedMIDIExtractionTask': 'deployment.QuantizedMIDIExtractionONNXModule',
}
