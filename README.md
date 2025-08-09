# Lightweight Skin Lesion Classification with GAN-based Augmentation (HAM10000)

## Motivation
In rural and tribal parts of India, access to dermatologists is limited and connectivity can be unreliable. To make AI-assisted screening useful there, models must be:
- **Lightweight** — run on mobile/edge devices  
- **Offline-capable** — no constant internet  
- **Fast** — results in seconds

I first built a **MobileNetV2 feasibility prototype** (separate repo) to test this idea. It ran fast on low-resource devices, but severe **class imbalance** in HAM10000 caused overfitting and poor recall on rare, clinically important classes.  

**Question:** can we use **GAN-generated images** to improve representation of minority classes, *without* abandoning the lightweight/portable constraint?

> **Baseline (separate repo):** https://github.com/ShridharPol/Skin-Lesion-Classifier

---

## Approach (what I did)
The pipeline follows the Self-Transfer GAN (STGAN) idea and is implemented with a lightweight backbone:

1) **Baseline classifier**  
   - **Model:** MobileNetV2 (ImageNet pretrained)  
   - **Outcome:** Fast & small, but overfits due to class imbalance.

2) **Stage 1 — Global GAN training**  
   - **Backbone:** **FastGAN** (unconditional) trained on *all* HAM10000 classes to learn universal lesion features.

3) **Stage 2 — Classwise fine-tuning**  
   - **One GAN per minority class.**  
   - Initialize from Stage 1 weights.  
   - **Freeze-D** (freeze early discriminator layers) + **Barlow Twins** self-supervised loss to stabilize/improve diversity with few samples.

4) **Post-generation deduplication (quality control)**  
   After generating all class-specific images in Stage 2, I applied a **two-step perceptual hashing filter** to improve dataset diversity:  
   - **pHash** — removes near-duplicates via **global** perceptual similarity (threshold = **10**)  
   - **wHash** — removes near-duplicates via **local/texture** similarity (threshold = **10**)  
   This curation step reduced redundancy in the final synthetic dataset used to train the classifier.

5) **Classifier with GAN-augmented data**  
   - Train MobileNetV2 on **real + deduplicated synthetic** images.  
   - Evaluate on mixed validation (real+synthetic) and on a **purely real** held-out test set.

---

## Why FastGAN (vs heavier GANs like StyleGAN2)?
- **Efficient:** Trains quickly on limited GPUs and smaller datasets.  
- **Lightweight:** Better aligned with mobile/edge constraints and the overall project goal.  
- **Flexible:** Easy to integrate **Freeze-D** and **Barlow Twins**; clean to customize.  
- **Controllable:** Simpler training loop and modules made iteration/debugging faster.

> The STGAN paper compares against conditional StyleGAN2+ADA as a baseline; it does not require a specific backbone for Stage 1. I chose **FastGAN** deliberately for compute and portability reasons.

---

## Results & Lessons

**Final metrics**
- **Mixed (real + synthetic) validation:** > **90%** accuracy  
- **Purely real test set:** ~ **11%** accuracy  

**Key learnings**
1. **Lightweight models are viable** for low-resource healthcare (MobileNetV2 is fast and small), but real-world performance is limited more by **data quality/imbalance** than model choice alone.  
2. **GANs need quality control.** Raw outputs contained duplicates/low-variance samples; **pHash + wHash (threshold=10)** removed near-duplicates and improved diversity.  
3. **Transfer inside GANs helps.** Stage-1→Stage-2 transfer stabilized minority-class training and reduced time to useful samples.  
4. **Evaluate like you’ll deploy.** Mixed validation can be overly optimistic; **real-only test** is essential for true generalization in medical AI.  
5. **Pipelines > single models.** A staged design (baseline → GAN → dedup → classifier) plus hosting large assets on Kaggle made the work reproducible and maintainable.

---

## Datasets & Pretrained Weights (Kaggle)
Large assets are hosted on Kaggle to keep this repo lean.

1. **Stage 1 — Global FastGAN Weights (70K iters)**  
   `[Kaggle Link](https://www.kaggle.com/datasets/shridharspol/fastgan-weights-70k-iter/data)`

2. **Stage 2 — Classwise Fine-Tuned Weights (all 6 classes)**  
   `[Kaggle Link](https://www.kaggle.com/datasets/shridharspol/stgan-stage2-finetuned-weights/settings)`

3. **Synthetic Dataset (deduplicated) + Real Test Set (with labels)**  
   `[Kaggle Link](https://www.kaggle.com/datasets/shridharspol/synthetic-data-ham10000/data)`

> You can reproduce results *without* retraining by downloading the weights, or regenerate everything from the notebooks.

---
## Repository Structure

```plaintext
fastgan-ham10000-phase1-unconditional-gan.ipynb
# Stage 1: train global FastGAN on all HAM10000 classes

# Stage 2: one notebook per class (all call into the code in the folder below)
STGAN-HAM10000-AKIEC-BT.ipynb
STGAN-HAM10000-BCC-BT.ipynb
STGAN-HAM10000-BKL-BT.ipynb
STGAN-HAM10000-DF-BT.ipynb
STGAN-HAM10000-MEL-BT.ipynb
STGAN-HAM10000-VASC-BT.ipynb

# Post-generation deduplication (run after Stage 2 for the final synthetic set)
Generated_Images_Deduplication_pHash_wHash.ipynb

# Train classifier with deduplicated synthetic + real data
SkinLesion_Classifier_MobileNetv2_SyntheticData.ipynb

STGAN-Finetune-BarlowTwins/
├── stgan_models_finetune.py         # Stage 2 G/D architectures
├── stgan_operations_finetune.py     # forward ops, Freeze-D hooks, feature taps
├── stgan_train_finetune.py          # training loop (Freeze-D + Barlow Twins)
├── eval.py                          # FID/KID/MS-SSIM/Precision-Recall helpers
├── train_utils.py                   # logging, checkpointing, seeding
└── args                             # shared config (paths, lrs, betas, etc.)

scripts/
└── download_from_kaggle.md          # CLI commands to pull datasets/weights
```
---

## Quickstart

## 1️ Environment Setup
```bash
pip install -r requirements.txt
```

## 2️ Download datasets & pretrained weights from Kaggle
See scripts/download_from_kaggle.md and replace <owner> and <slug> with your own:
```bash
# examples
kaggle datasets download -d <owner>/<stage1-slug> -p datasets/ --unzip
kaggle datasets download -d <owner>/<stage2-slug> -p datasets/ --unzip
kaggle datasets download -d <owner>/<synthetic-slug> -p datasets/ --unzip
```

## 3️ Run notebooks in order
Stage 1
fastgan-ham10000-phase1-unconditional-gan.ipynb
(or load Stage-1 weights from Kaggle)

Stage 2 (pick a class)
STGAN-HAM10000-<CLASS>-BT.ipynb

Deduplicate synthetic set
Generated_Images_Deduplication_pHash_wHash.ipynb

Train classifier
Use real + deduplicated synthetic images in your classifier notebook.

---
## Dataset Links (Kaggle)
Stage 1 (Global FastGAN Weights):  [Kaggle Link](https://www.kaggle.com/datasets/shridharspol/fastgan-weights-70k-iter/data)

Stage 2 (Classwise Fine-tuned Weights): [Kaggle Link](https://www.kaggle.com/datasets/shridharspol/stgan-stage2-finetuned-weights/settings)

Synthetic Dataset: [Kaggle Link](https://www.kaggle.com/datasets/shridharspol/synthetic-data-ham10000/data)

---
---

## References

[1] Su, Y., Zhang, H., Chen, Y., Xu, W., Zhao, Y., & Xu, W. (2024).  
**A GAN-Based Data Augmentation Method for Imbalanced Multi-Class Skin Lesion Classification.**  
*IEEE Access*, vol. 12, pp. 18049–18060, 2024.  
doi:[10.1109/ACCESS.2024.3350764](https://doi.org/10.1109/ACCESS.2024.3350764)

If you use this repository or adapt the methods, please cite both the above paper and this GitHub repo.

---

## Author

**Shridhar Sunilkumar Pol**  
MS in Electrical and Computer Engineering  
Northeastern University  
Passionate about using AI for social good
