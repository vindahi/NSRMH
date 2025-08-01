# NSRMH 

## Noise-Aware Semantic Robust Multi-modal Hashing for Multi-modal Retrieval

## Overview
**NSRMH** (Noise-aware Semantic Robust Multi-modal Hashing) is a novel framework designed to address the challenge of noisy supervision in supervised multi-modal hashing. This method improves retrieval performance by refining noisy labels and enhancing semantic alignment through adaptive and robust mechanisms. 

Supervised multi-modal hashing often suffers from degraded performance due to label noise in real-world datasets. NSRMH mitigates this issue with:
- **Dynamic Discriminative Loss:** To evaluate sample reliability and adaptively mitigate the influence of noisy labels.
- **Pseudo-label Refinement:** A probabilistic model generates soft pseudo-labels, which are more robust to noise.
- **Prototype-based Semantic Alignment Module:** To enhance the discriminability of hash codes.

Extensive experiments on three widely-used datasets (MIRFlickr, NUS-WIDE, and MS COCO) demonstrate that NSRMH achieves state-of-the-art performance across various noise levels.

---

## Abstract
Supervised multi-modal hashing relies on semantic labels to learn compact binary codes for efficient retrieval. However, its performance degrades significantly under label noise—a common issue in real-world datasets. In this paper, we propose **Noise-aware Semantic Robust Multi-modal Hashing (NSRMH)**, a framework that addresses noisy supervision through adaptive pseudo-label refinement and enhanced semantic alignment. Specifically, we design a dynamic discriminative loss to assess sample reliability and leverage a probabilistic model to generate soft pseudo-labels. These refined labels guide a prototype-based semantic alignment module to improve hash code discriminability. Experiments on three benchmark datasets show that **NSRMH consistently outperforms state-of-the-art methods under various noise levels**, demonstrating robustness and effectiveness.

---

## Datasets
We evaluate NSRMH on three popular benchmark datasets: **MIRFlickr**, **MS COCO**, and **NUS-WIDE**. Their details are as follows:

| Datasets   | Categories | Training Samples | Retrieval Samples | Query Samples |
|------------|------------|------------------|-------------------|---------------|
| MIRFlickr  | 24         | 5,000            | 18,015            | 2,000         |
| MS COCO    | 80         | 5,000            | 121,287           | 2,000         |
| NUS-WIDE   | 10         | 5,000            | 184,577           | 2,000         |

Datasets are available for download from **[https://pan.baidu.com/s/1-_XwzUb8w-UMupa_U6aWnw]**  Code：u7gu.

---

## Experimental Setup
### Hardware and Software
- **Python Version:** 3.8.18  
- **PyTorch Version:** 1.10.1  
- **GPU:** NVIDIA RTX 3090  

### Key Steps:
1. Use `generate.py` to simulate noisy data for experiments.
2. Execute the training script for a specific dataset, e.g.,:
   ```bash
   bash flickr.sh
