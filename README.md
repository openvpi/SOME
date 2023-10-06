# SOME
SOME: Singing-Oriented MIDI Extractor.

> WARNING
> 
> This project is under beta version now. No backward compatibility is guaranteed.

## Overview

SOME is a MIDI extractor that can convert singing voice to MIDI sequence, with the following advantages:

1. Speed: 9x faster than real-time on an i5 12400 CPU, and 300x on a 3080Ti GPU.
2. Low resource dependency: SOME can be trained on custom dataset, and can achieve good results with only 3 hours of training data.
3. Functionality: SOME can produce non-integer MIDI values, which is specially suitable for DiffSinger variance labeling.

## Getting Started

### Installation

SOME requires Python 3.8 or later. We strongly recommend you create a virtual environment via Conda or venv before installing dependencies.

1. Install PyTorch 2.1 or later following the [official instructions](https://pytorch.org/get-started/locally/) according to your OS and hardware.

2. Install other dependencies via the following command:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) For better pitch extraction results, please download the RMVPE pretrained model from [here](https://github.com/yxlllc/RMVPE/releases) and extract it into `pretrained/` directory.

### Inference via pretrained model

Download pretrained model of SOME from [releases](https://github.com/openvpi/SOME/releases) and extract them somewhere.

To infer with CLI, run the following command:

```bash
python infer.py --model CKPT_PATH --wav WAV_PATH
```

This will load model at CKPT_PATH, extract MIDI from audio file at WAV_PATH and save a MIDI file. For more useful options, run

```bash
python infer.py --help
```

To infer with Web UI, run the following command:

```bash
python webui.py --work_dir WORK_DIR
```

Then you can open the gradio interface through your browser and use the models under WORK_DIR following the instructions on the web page. For more useful options, run

```bash
python webui.py --help
```

### Training from scratch

_Training scripts are uploaded but may not be well-organized yet. For the best compatibility, we suggest training your own model after a stable release in the future._


## Disclaimer

Any organization or individual is prohibited from using any recordings obtained without consent from the provider as training data. If you do not comply with this item, you could be in violation of copyright laws or software EULAs.

## License

SOME is licensed under the [MIT License](LICENSE).

