# YES: You should Examine Suspect cues for low-light object detection

This repository provides the official implementation of **YES**, a low-light object detection framework proposed in our paper published in *Computer Vision and Image Understanding (CVIU)*.

> Ye, S., Huang, W., Liu, W., Chen, L., Wang, X., & Zhong, X. (2025).
> **YES: You should Examine Suspect cues for low-light object detection**.
> *Computer Vision and Image Understanding*, 251, 104271.
> [DOI: 10.1016/j.cviu.2024.104271](https://doi.org/10.1016/j.cviu.2024.104271)

## 📌 Abstract

Object detection in low-light conditions presents substantial challenges, particularly the issue we define as **"low-light object-background cheating."** This phenomenon arises from uneven lighting, leading to blurred and inaccurate object edges. Most existing methods focus on basic feature enhancement and addressing the gap between normal-light and synthetic low-light conditions. However, they often overlook the complexities introduced by uneven lighting in real-world environments.

To address this, we propose a novel low-light object detection framework, **You Examine Suspect (YES)**, comprising two key components:

* The **Optical Balance Enhancer (OBE)** emphasizes *balance* by employing techniques such as inverse tone mapping, white balance, and gamma correction to recover details in dark regions while adjusting brightness and contrast without introducing noise.
* The **Entanglement Attenuation Module (EAM)** focuses on *disentanglement* by analyzing both object regions and surrounding areas affected by lighting variations and integrating multi-scale contextual information to clarify ambiguous features.

Extensive experiments on **ExDark** and **Dark Face** datasets demonstrate the superior performance of the proposed YES framework, validating its effectiveness in low-light object detection tasks.

## 🛠️ Getting Started

### Installation

```bash
git clone https://github.com/Regina971/YES.git
cd YES
pip install -r requirements.txt
```

## 📂 Dataset Preparation

Download and prepare the datasets:

* [ExDark Dataset](https://drive.google.com/file/d/1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx/view?usp=sharing)
* [Dark Face Dataset](https://flyywh.github.io/CVPRW2019LowLight/)

Directory structure:

```
datasets/
├── ExDark/
└── DarkFace/
```

## 🏃‍♂️ Training

To train YES on ExDark:

```bash
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/yes/yes_exdark.yml -t rtdetr_r50vd_6x_coco_from_paddle.pth
```


## 📈 Evaluation

To evaluate the trained model:

```bash
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/yes/yes_exdark.yml -t yes_exdark.pth --test-only
```

## 🙏 Acknowledgements

Our framework is built upon [RT-DETR](https://github.com/lyuwenyu/RT-DETR).
We sincerely thank the RT-DETR authors for their outstanding work and open-source contributions.

## 📄 Citation

If you use this repository in your research, please cite:

```bibtex
@article{YE2025104271,
title = {YES: You should Examine Suspect cues for low-light object detection},
journal = {Computer Vision and Image Understanding},
volume = {251},
pages = {104271},
year = {2025},
issn = {1077-3142},
doi = {https://doi.org/10.1016/j.cviu.2024.104271},
url = {https://www.sciencedirect.com/science/article/pii/S1077314224003527},
author = {Shu Ye and Wenxin Huang and Wenxuan Liu and Liang Chen and Xiao Wang and Xian Zhong}
}
```