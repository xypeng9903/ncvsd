## Requirements
* 64-bit Python 3.9 and PyTorch 2.1 (or later). See https://pytorch.org for PyTorch install instructions.
* Python libraries: `pip install click Pillow psutil requests scipy tqdm diffusers accelerate==0.27.2 tensorboard`


## Training

## Evaluation
```bash
# ImageNet 64x64
torchrun --standalone --nproc_per_node=8 calculate_metrics.py gen \
    --net='<your-snapshot.pkl>' \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img64.pkl \
    --seed=123456789

# ImageNet 512x512
torchrun --standalone --nproc_per_node=8 calculate_metrics.py gen \
    --net='<your-snapshot.pkl>' \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl \
    --seed=123456789
```
