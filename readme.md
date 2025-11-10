# 🌌 HyperMapper: Hyperbolic Structure-Aware Mapping for Dual-Task Scene Parsing

Official PyTorch implementation of  
**"Beyond Euclidean Tokens: Hyperbolic Structure-Aware Mapping for Dual-Task Scene Parsing with Only 0.6% Additional Parameters"**

---

## 🔍 Overview

**HyperMapper** is a **hyperbolic structure-aware framework** for unified scene parsing — jointly performing semantic segmentation and depth estimation **across unseen domains** without retraining.  
It builds on **DepthAnythingV2** and introduces **hyperbolic token-to-feature interactions**, achieving:

- **26.8% reduction** in hierarchical bias  
- **+4.0% mIoU** gain on cross-domain benchmarks  
- **Only 0.6% extra parameters (1.99M)**

---

## 🎥 Zero-Shot Multi-Domain Scene Parsing

**Rainy**
<video src="https://github.com/user-attachments/assets/6c630330-bd32-4f7a-96ef-e6342ac8d7d9" width="480" autoplay loop muted></video>

**Night**
<video src="https://github.com/user-attachments/assets/11fcc21f-f7e5-4393-8bff-49b84afec680" width="480" autoplay loop muted></video>

**Snowy**
<video src="https://github.com/user-attachments/assets/0afe46c1-8b8d-4917-8c5e-692cf5205c14" width="480" autoplay loop muted></video>

**Foggy**
<video src="https://github.com/user-attachments/assets/b7d26088-ec4c-4ff2-8a88-748b5f306803" width="480" autoplay loop muted></video>

**Overcast**
<video src="https://github.com/user-attachments/assets/ac175212-3f9e-47ca-8c43-81f0e150a3f6" width="480" autoplay loop muted></video>

> Zero-shot dual-task parsing (semantic segmentation + depth estimation) across unseen weather domains.


> Zero-shot dual-task parsing (segmentation + depth) on unseen weather domains.

---

## ⚙️ Quick Start

```bash
git clone https://github.com/yourusername/HyperMapper.git
cd HyperMapper
pip install -r requirements.txt
python demo_infer.py --image_path ./samples/demo.png
