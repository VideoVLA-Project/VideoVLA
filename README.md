# VideoVLA[NeurIPS 2025]

**VideoVLA** is a simple approach that explores the potential of directly transforming large video generation models into robotic VLA manipulators..


This repository contains the **official implementation** of the paper:

> **VideoVLA: Video Generators Can Be Generalizable Robot Manipulators**
> *NeurIPS 2025*



🔗 **Project Page:**  [Project Website](https://videovla-nips2025.github.io/)

📄 **Paper:**  [Paper Link](https://arxiv.org/pdf/2512.06963)

<!-- ## 🚧 Code Release Status

The official code for **VideoVLA** will be **released very soon**. -->

## 1. Quick Start

First, prepare the runtime environment and install all required dependencies by running:

```bash
bash build.sh
```

## 2. Downloading the Pretrained Checkpoint


Our method relies on pretrained components from **CogVideo**. You can follow the official CogVideo instructions to obtain the pretrained checkpoints:
[CogVideo](https://github.com/zai-org/CogVideo?tab=readme-ov-file#sat)

Specifically, download:

* **T5** checkpoint
* **VAE** checkpoint



After downloading, update the checkpoint paths in the following configuration file:

```text
config_use/action_config/videovla_config.yaml
```

Make sure the paths correctly point to the downloaded T5 and VAE checkpoints before starting training or evaluation.

---

## 3. Inference

This section describes how to run inference with a trained model checkpoint to generate video and action.

```bash
python sample_video_action.py \
  --base config_use/action_config/videovla_config.yaml config_use/action_config/inference_config/inference.yaml
```


## Citations
```bibtex
@article{
    videovla,
    title={VideoVLA: Video Generators Can Be Generalizable Robot Manipulators},
    author={Yichao Shen and Fangyun Wei and Zhiying Du and Yaobo Liang and Yan Lu and Jiaolong Yang and Nanning Zheng and Baining Guo},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems(NeurIPS2025)},
    year={2025},
    url={https://openreview.net/forum?id=UPHlqbZFZB}
    }
  
```


<!-- 
*Code coming soon.* 🚀 -->
