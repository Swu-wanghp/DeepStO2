# OxyRIG-Net: Physiology-Guided Region–Relation Graph Network for Deception Detection

This repository contains the official implementation of **OxyRIG-Net**,  
a physiology-guided region–relation graph framework for facial deception detection  
based on **tissue oxygen saturation (StO₂)** imaging.

---

## 🧩 Environment

| Component | Version |
|------------|----------|
| OS | Ubuntu 20.04.4 LTS |
| Python | 3.8 |
| PyTorch | 1.10.1 |
| CUDA | 10.2 |
| cuDNN | 7.6.5 |
| GPU | NVIDIA GeForce RTX 2080 Ti |

---

## 🚀 Getting Started

### 1. Clone this repository
bash
git clone git@github.com:Swu-wanghp/DeepStO2.git
cd OxyRIG-Net

2. Prepare environment        
conda create -n env_name python=3.8
conda activate env_name
pip install -r requirements.txt

3. Upload StO2 data to the specified folder
The StO2-Deception-Detection dataset is publicly available for academic research. Researchers can obtain access by contacting the corresponding author via email (ctong\@swu.edu.cn; chentong\@psych.ac.cn) and signing a License Agreement.  
4. Download the weights obtained from the first phase of training.
```bash
$ tar -xf features.tar.gz -C dir_to_save_feature
5. Training
``` python ./train.py
6. Inference
``` python ./test.py
We also provide ckpts, logs, etc. to reproduce the results in the paper, please download ckpt.tar.gz.
You may open an issue or email me at wanghp568@gmail.com if you have any inquiries or issues.
