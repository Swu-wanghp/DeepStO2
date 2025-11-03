name: "The Face of Deception: StO₂ Dataset, Pattern Analysis, and Detection Model"
description: |
  This repository contains the official implementation of **OxyRIG-Net**, a physiology-guided region–relation graph framework for facial deception detection based on **tissue oxygen saturation (StO₂)** imaging.

sections:
  - title: "🧩 Environment"
    content: |
      | Component | Version |
      |------------|----------|
      | OS | Ubuntu 20.04.4 LTS |
      | Python | 3.8 |
      | PyTorch | 1.10.1 |
      | CUDA | 10.2 |
      | cuDNN | 7.6.5 |
      | GPU | NVIDIA GeForce RTX 2080 Ti |

  - title: "🚀 Getting Started"
    steps:
      - step: "1. Clone this repository"
        code: |
          ```bash
          git clone git@github.com:Swu-wanghp/DeepStO2.git
          cd OxyRIG-Net
          ```
      - step: "2. Prepare environment"
        code: |
          ```bash
          conda create -n env_name python=3.8
          conda activate env_name
          pip install -r requirements.txt
          ```
      - step: "3. Upload StO₂ data to the specified folder"
        details: |
          The StO₂-Deception-Detection dataset is publicly available for academic research.
          Researchers can obtain access by contacting the corresponding author via email (ctong@swu.edu.cn; chentong@psych.ac.cn) and signing a License Agreement.
      - step: "4. Download the weights obtained from the first phase of training"
        code: |
          ```bash
          tar -xf features.tar.gz -C dir_to_save_feature
          ```
      - step: "5. Training"
        code: |
          ```bash
          python train.py
          ```
      - step: "6. Inference"
        code: |
          ```bash
          python test.py
          ```
        details: |
          We also provide ckpts, logs, etc. to reproduce the results in the paper. Please download `ckpt.tar.gz`.

  - title: "📊 Dataset Overview"
    content: |
      The **Deception-Detection-StO₂** dataset includes three experimental paradigms with increasing stress intensity:
      1. Personal Information Description — low stress  
      2. Factual Statement — moderate stress  
      3. Mock Crime — high stress  

      All hyperspectral data were captured under calibrated **halogen illumination** to ensure uniform facial lighting and stable reflectance estimation.

  - title: "🧠 Model Overview"
    content: |
      OxyRIG-Net adopts a two-stage “attribute first, reason later” framework:
      - Stage I: Physiology-guided regional attribution using ACCA and soft facial priors  
      - Stage II: Multi-relation graph reasoning over ROIs (spatial, bilateral, functional)  

      This enables interpretable and robust deception detection through both localized oxygenation variation and cross-regional coupling.

  - title: "📬 Contact"
    content: |
      If you have any questions or encounter any issues, please feel free to open an issue or contact:
      - **Hanpu Wang** — wanghp568@gmail.com  
      - **Tong Chen** — ctong@swu.edu.cn / chentong@psych.ac.cn  

  - title: "🙏 Acknowledgement"
    content: |
      This research was supported by the **Key Laboratory of Cognitive Neuroscience, Southwest University**, and the **Institute of Psychology, Chinese Academy of Sciences**.
