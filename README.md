## Noise Conditional Variational Score Distillation <br><sub>Official PyTorch implementation of the ICML 2025 paper</sub>

![Overview](assets/overview.jpg "Overview")

**Noise Conditional Variational Score Distillation**  
Xinyu Peng, Ziyang Zheng, Yaoming Wang, Han Li, Nuowen Kan, Wenrui Dai, Chenglin Li, Junni Zou, Hongkai Xiong <br>
https://arxiv.org/abs/2506.09416

**Abstract:**  
*We propose Noise Conditional Variational Score Distillation (NCVSD), a novel method for distilling pretrained diffusion models into generative denoisers. We achieve this by revealing that the unconditional score function implicitly characterizes the score function of denoising posterior distributions. By integrating this insight into the Variational Score Distillation (VSD) framework, we enable scalable learning of generative denoisers capable of approximating samples from the denoising posterior distribution across a wide range of noise levels. The proposed generative denoisers exhibit desirable properties that allow fast generation while preserving the benefits of iterative refinement: (1) fast one-step generation through sampling from pure Gaussian noise at high noise levels; (2) improved sample quality by scaling the test-time compute with multi-step sampling; and (3) zero-shot probabilistic inference for flexible and controllable sampling. We evaluate NCVSD through extensive experiments, including class-conditional image generation and inverse problem solving. By scaling the test-time compute, our method outperforms teacher diffusion models and is on par with consistency models of larger sizes. Additionally, with significantly fewer NFEs than diffusion-based methods, we achieve record-breaking LPIPS on inverse problems.*

## Requirements

### Environment
- 64-bit Python 3.9 and PyTorch 2.1 ([PyTorch installation guide](https://pytorch.org))
- Python libraries:  
  ```bash
  pip3 install -r requirements.txt
  ```

### Data
Follow the instructions in [EDM2](https://github.com/NVlabs/edm2?tab=readme-ov-file#preparing-datasets) to prepare the following zip files in the `../data/edm2` folder:
- `img64.zip`
- `img512.zip`
- `img512-sd.zip`

In addition, prepare `ffhq256.zip` with the following command:
```
python dataset_tool.py convert \
    --source=path/to/ffhq \
    --dest=../data/edm2/ffhq256.zip \
    --resolution=256x256 --transform=center-crop-dhariwal 
```

### Teacher EDM2 models
Download EDM2 models into the `../model_zoo/edm2` folder from the links listed in `quick_start/edm2.txt`.

## Training

Start training by running:
```bash
bash quick_start/train-{MODEL_NAME}.sh
```

The `{MODEL_NAME}` can be one of the following:
 - `img64-{s|m|l}`     
 - `img512-{s|m|l}`
 - `ffhq256-xs`

\* Please modify the `DISTRIBUTED_ARGS` in the scripts according to your training environment.

\* The batch size per GPU (controlled by `--batch-gpu` in the bash scripts) has been optimized for training under NVIDIA A100-80G GPUs. If you run out of GPU memory, please consider modifying `--batch-gpu` to reduce memory cost. Note that, similar to the [EDM2 training](https://github.com/NVlabs/edm2?tab=readme-ov-file#training-new-models), modifying `--batch-gpu` is safe in the sense that it has no interaction with the other hyperparameters.  

## Class-conditional image generation

NCVSD achieves the following image generation performance on ImageNet-64x64 and ImageNet-512x512 datasets:

*Image Generation on ImageNet-64x64:*
| Model | Checkpoint | 1-step FID | 2-step FID | 4-step FID |
| - | - | - | - | - |
NCVSD-S | [ncvsd-img64-s.pkl]() | 3.13 | 2.66 | 2.14 |
NCVSD-M | [ncvsd-img64-m.pkl]() | 3.04 | 2.47 | 1.92 |
NCVSD-L | [ncvsd-img64-l.pkl]() | 2.96 | 2.35 | 1.53 |

*Image Generation on ImageNet-512x512:*
| Model | Checkpoint | 1-step FID | 2-step FID | 4-step FID |
| - | - | - | - | - |
NCVSD-S | [ncvsd-img512-s.pkl]() |2.95 | 2.60 | 2.00 |
NCVSD-M | [ncvsd-img512-m.pkl]() |2.85 | 2.08 | 1.92 |
NCVSD-L | [ncvsd-img512-l.pkl]() |2.56 | 2.03 | 1.76 |

\* The checkpoints are not yet publicly available, pending review under Meituan's open-source protocol.

### Generating images

To generate images using provided pkl checkpoints, run `generate_images.py` and specifying
- `--net`, the pkl checkpoint.
- `--ts`, the sampling timesteps, please refer to Table 3 in the paper. * You should add 39 at the last of the timesteps. 
- `--outdir`, where to save the generation images.
- `--class`, (optional), the class of generation images.
- `--seeds`, random seed.

For example, to generate 4 images using `NCVSD-S` model trained on ImageNet-512x512 dataset with 2 NFEs (sampling timesteps set to 10,22), run

```bash
python generate_images.py \
    --net=path/to/ncvsd-img512-s.pkl \
    --ts=10,22,39 \
    --outdir=out \
    --seeds=0-3
```

### Evaluating FID

To reproduce the FID scores, run `calculate_metrics.py` and specifying
- `--net`, the pkl checkpoint.
- `--ts`, the sampling timesteps, please refer to Table 3 in the paper. * You should add 39 at the last of the timesteps.
- `--ref`, pre-computed reference statistics for the dataset.
- `--seed`, random seed, 123456789 by default.

For `--ref`, we use available reference statistics provided in https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/:
- Set `--ref=https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img64.pkl` for computing FID on ImageNet-64x64 dataset.
- Set `--ref=https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl` for computing FID on ImageNet-512x512 dataset.

For example, to evaluate 2-step FID of `NCVSD-S` on ImageNet-512x512 dataset, run

```bash
torchrun --standalone --nproc_per_node=8 calculate_metrics.py gen \
    --net=path/to/ncvsd-img512-s.pkl \
    --ts=10,22,39 \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl \
    --seed=123456789
```

## Inverse problem solving
1. Download the FFHQ checkpoint [ncvsd-ffhq256.pkl](https://huggingface.co/xypeng9903/ncvsd/resolve/main/edm2-ffhq256-xs.pkl?download=true) to `../model_zoo/ncvsd`.
2. Download the test data [test.zip](https://drive.google.com/file/d/1I8at4Y1MPrKV8yPHq_6sn6Et7Elyxavx/view?usp=drive_link) and unzip to `../data`.
3. Solving inverse problems with PnP-GD by running

```bash
bash quick_start/posterior-sample.sh {TASK_NAME}
```

The `{TASK_NAME}` can be one of the following:
- `box_inpainting`
- `gaussian_deblur_circ`
- `motion_deblur_circ`
- `super_resolution`
- `phase_retrieval`

## Citation
If you find this repo helpful, please cite:
```
@inproceedings{
  peng2025noise,
  title={Noise Conditional Variational Score Distillation},
  author={Xinyu Peng and Ziyang Zheng and Yaoming Wang and Han Li and Nuowen Kan and Wenrui Dai and Chenglin Li and Junni Zou and Hongkai Xiong},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=UYUqCPCZCw}
}
```

## Acknowledgments

This code is based on: 

- [EDM2](https://github.com/NVlabs/edm2): Provide the code structure.

- [DAPS](https://github.com/zhangbingliang2019/DAPS): Provide the evaluation code for inverse problem solving.

- [PnP-DM](https://github.com/zihuiwu/PnP-DM-public): Provide the code for closed-form solutions of likelihood steps.


