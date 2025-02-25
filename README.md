## Noise Conditional Variational Score Distillation <br><sub>Official PyTorch implementation</sub>


**Noise Conditional Variational Score Distillation** <br>
Xinyu Peng, Ziyang Zheng, Yaoming Wang, Han Li, Nuowen Kan, Wenrui Dai, Chenglin Li, Junni Zou, Hongkai Xiong <br>
https://arxiv.org/abs/2402.02149v2

**Abstract:** *We propose Noise Conditional Variational Score Distillation (NCVSD), a novel method for distilling a pretrained diffusion model into generative denoiser. We achieve this by revealing that the unconditional score function implicitly characterizes the score function of denoising posterior distributions. By integrating this insight into the Variational Score Distillation (VSD) framework, we enable scalable learning of generative denoisers capable of approximating samples from the denoising posterior distribution across a wide range of noise levels. The proposed generative denoisers exhibit desirable properties that allow fast generation while preserve the benefit of iterative refinement: (1) fast one-step generation through sampling from pure Gaussian noise at high noise levels; (2) improved sample quality by scaling the test-time compute with multi-step sampling; and (3) zero-shot probabilistic inference for flexible and controllable sampling. We evaluate NCVSD through extensive experiments, including class-conditional image generation and inverse problem solving. By scaling the test-time compute, our method outperforms teacher diffusion models and is on par with consistency models of larger sizes. Additionally, with significantly fewer NFEs than diffusion-based methods, we achieve record-breaking LPIPS on inverse problems.*


## Requirements

### Environment
* 64-bit Python 3.9 and PyTorch 2.1 (or later, see https://pytorch.org for install instructions).
* Python libraries: `pip3 install click Pillow psutil requests scipy tqdm accelerate diffusers tensorboard`

### Data and pretrained EDM2 models
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

The `{MODEL_NAME}` can be one of the following:
 - `img64-{s|m|l}`     
 - `img512-{s|m|l}`
 - `ffhq256-xs`

Note that the effective batch size (2048 for ImageNet, 128 for FFHQ) should be divided by `{NUM_GPUS}` x `{BATCH_PER_GPU}`.

## Class-conditional image generation
### ImageNet-64x64
*Table 1. FID on ImageNet-64x64*
| Model | 1-step FID | 2-step FID | 4-step FID |
| - | - | - | - |
NCVSD-S | 2.94 | 2.30 | 2.06 |
NCVSD-M |  |  |  |
NCVSD-L | 2.93 | 2.02 | 1.84 |

To reproduce the results in Table 1, run
```bash
bash quick_start/eval-img64.sh {NUM_GPUS} {SNAPSHOT.pkl} 12,39         # 1-step FID
bash quick_start/eval-img64.sh {NUM_GPUS} {SNAPSHOT.pkl} 10,22,39      # 2-step FID
bash quick_start/eval-img64.sh {NUM_GPUS} {SNAPSHOT.pkl} 0,10,20,30,39 # 4-step FID
```

### ImageNet-512x512
*Table 2. FID on ImageNet-512x512*
| Model | 1-step FID | 2-step FID | 4-step FID |
| - | - | - | - |
NCVSD-S | 3.93 | 3.01 | 2.31 |
NCVSD-M | 3.18 | 2.32 | 1.98 |
NCVSD-L | 2.92 | 2.18 | 1.57 |

To reproduce the results in Table 2, run
```bash
bash quick_start/eval-img512.sh {NUM_GPUS} {SNAPSHOT.pkl} 12,39         # 1-step FID
bash quick_start/eval-img512.sh {NUM_GPUS} {SNAPSHOT.pkl} 10,22,39      # 2-step FID
bash quick_start/eval-img512.sh {NUM_GPUS} {SNAPSHOT.pkl} 0,10,20,30,39 # 4-step FID
```

## Inverse problem solving
Solving inverse problems with PnP-NCVSD by running

```bash
bash quick_start/posterior-sample.sh {TASK_NAME}
```

The `{TASK_NAME}` can be one of the following:
- `gaussian_deblur_circ`
- `motion_deblur_circ`
- `super_resolution`
- `phase_retrieval`

## Citation

## Acknowledgments



