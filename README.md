## Requirements
* 64-bit Python 3.9 and PyTorch 2.1 (or later). See https://pytorch.org for PyTorch install instructions.
* Python libraries: `pip3 install click Pillow psutil requests scipy tqdm accelerate diffusers tensorboard`

## Data and pretrained EDM2 models
Follow the instructions in [EDM2](https://github.com/NVlabs/edm2?tab=readme-ov-file#preparing-datasets) to prepare the following zip files into the `../data/edm2` folder:
- `img64.zip`
- `img512.zip`
- `img512-sd.zip`

Download pretrained EDM2 models into `../model_zoo/edm2` folder from the links listed in `quick_start/edm2.txt`.

## Training
Start training by running
```bash
bash quick_start/train-{MODEL_NAME}.sh {NUM_GPUS} {BATCH_PER_GPU}
```
Note that the effective batch size 2048 should be divided by `{NUM_GPUS}` x `{BATCH_PER_GPU}`.

The `{MODEL_NAME}` can be one of the following:
 - `img64-{s|m|l}`     
 - `img512-{s|m|l|xl}`
