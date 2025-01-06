## Requirements
* 64-bit Python 3.9 and PyTorch 2.1 (or later). See https://pytorch.org for PyTorch install instructions.
* Python libraries: `pip install click Pillow psutil requests scipy tqdm diffusers accelerate==0.27.2 tensorboard`


## Training
Start training by running
```bash
bash quick_start/train-ffhq256-xs.sh {NUM_GPUS} {BATCH_PER_GPU}
```
Note that the effective batch size 128 should be divided by `{NUM_GPUS}` x `{BATCH_PER_GPU}`.

## Evaluation
```bash
torchrun --standalone --nproc_per_node={NUM_GPUS} calculate_metrics.py gen \
    --net='<your-snapshot.pkl>' \
    --ref='path/to/ffhq256.zip' \
    --seed=123456789
```
