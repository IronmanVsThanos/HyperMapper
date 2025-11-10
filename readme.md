# 🌌 HyperMapper: Hyperbolic Structure-Aware Mapping for Dual-Task Scene Parsing with Only 0.6% Additional Parameters

> Official PyTorch Implementation of the paper  
> **"Beyond Euclidean Tokens: Hyperbolic Structure-Aware Mapping for Dual-Task Scene Parsing with Only 0.6% Additional Parameters"**  
>  
> 🏛️ *Southwest Jiaotong University* · 🚀 *2025 IEEE Transactions on Intelligent Transportation Systems (under review)*  
>  
> 🔗 Paper | 📘 Project Page | 📦 [Dataset Links](#datasets) | 🧠 [Model Weights](#pretrained-weights)

---

## 🧭 Abstract

Achieving unified scene parsing that jointly performs cross-domain **semantic segmentation** and **depth estimation** without scene-specific retraining is crucial for robust perception in complex real-world environments, yet remains a challenging goal.

We identify a persistent **Hierarchical Bias**, where fine-grained categories degrade more severely than coarse-grained ones under domain or viewpoint shifts.  
To address this, we propose **HyperMapper**, a **hyperbolic structure-aware mapping framework** that unifies semantic and geometric understanding via **hyperbolic token-to-feature interactions** within the Poincaré ball.

By exploiting the negative curvature of hyperbolic space, HyperMapper naturally encodes hierarchical relationships and maintains geometric consistency across domains.  
With only **0.6% additional parameters (1.99M)**, it inherits the depth generalization of **DepthAnythingV2** while significantly improving cross-domain segmentation.

**Results:**  
- 🔺 **26.8% reduction** in hierarchical bias  
- 📈 **+4.0% mIoU** over state-of-the-art baselines  
- ⚡ Zero retraining for depth estimation  

---

## 🏗️ Framework Overview

<p align="center">
  <img src="assets/hypermapper_framework.png" width="85%">
</p>

**Key components:**
- **Hyperbolic Token Mapping** — Projects semantic tokens into the Poincaré ball using exponential and logarithmic maps.  
- **Geometry-Aware Modulation** — Ensures task alignment between semantic and geometric branches.  
- **PEFT Integration** — Parameter-efficient adaptation on frozen foundation backbones (e.g., DepthAnythingV2, DINOv2).  

---

## 🚀 Highlights

| Feature | Description |
|----------|-------------|
| 🧩 **Unified Scene Parsing** | Joint semantic segmentation & depth estimation |
| 🌍 **Cross-Domain Generalization** | Tested on Cityscapes → ACDC, BDD100K, WildDash, etc. |
| ⚙️ **Parameter-Efficient** | Only 0.6% additional parameters (≈1.99M) |
| 🌀 **Hyperbolic Geometry** | Structure-aware semantic embedding using Poincaré ball |
| 🧠 **DepthAnything Integration** | Retains pretrained geometric consistency |
| 🔍 **Hierarchical Bias Reduction** | Up to 26.8% improvement over Euclidean baselines |

---

## 🧪 Results

### Cross-Domain Scene Parsing (Cityscapes → ACDC)

| Method | Add. Params | mIoU ↑ | Hier. Bias ↓ | Depth Transfer |
|--------|--------------|--------|---------------|----------------|
| Baseline (DepthAnythingV2) | – | 62.1 | 1.00× | ✅ |
| Adapter-Only (LoRA) | +0.4% | 63.5 | 0.91× | ✅ |
| **HyperMapper (Ours)** | **+0.6%** | **66.1** | **0.74×** | ✅ |

---

## 🧩 Installation

```bash
git clone https://github.com/yourusername/HyperMapper.git
cd HyperMapper
conda create -n hypermapper python=3.10
conda activate hypermapper
pip install -r requirements.txt
