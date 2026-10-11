# Team conclusions — Tree Species Classification (NNCv 2026)

Each group member first built and trained their own notebook independently (see the `Tree project - …` folders).
All three notebooks share the same base structure (Lecture 6 notebook + the PyTorch transfer learning tutorial):
pretrained **ResNet18** and **ResNet50**, full fine-tuning with Adam (lr 1e-4, weight decay 1e-4), StepLR,
best-validation-epoch checkpointing, and evaluation with accuracy, macro precision / recall / F1, confusion
matrices and n-best (top-n) accuracy. The main differences are the data split, the hardware and the experiments
each member ran on top of the baseline.

---

## Jaakko Ruhanen — `Tree project - Jakeksii/Tree-Recognition-Project.ipynb`

**Setup**
- Used the **train / val / test split that ships with the dataset** (80 / 10 / 10, stratified):
  3 850 / 482 / 472 images. This is the split used in the final notebook.
- Local RTX 4070 Laptop GPU with mixed precision (bf16), `channels_last` and 8 DataLoader workers
  (the `Dataset` class was moved to `tree_data.py` so workers run on Windows).
  Training time for 15 epochs: ResNet18 ~1.4 min, ResNet50 ~3.1 min.
- Three training setups per architecture: **no augmentation** ("cleaned": resize + center crop),
  **tutorial augmentation** (RandomResizedCrop + horizontal flip) and **stronger augmentation round 2**
  (crop scale 0.5–1, rotation 10°, ColorJitter, label smoothing 0.1, 20 epochs).

**Results (test set, 472 images)**

| Model | Training data | Accuracy | Macro F1 |
|---|---|---|---|
| ResNet18 | no augmentation | 88.77 % | 89.29 % |
| ResNet50 | no augmentation | **90.89 %** | **91.47 %** |
| ResNet18 | tutorial augmentation | 88.14 % | 89.22 % |
| ResNet50 | tutorial augmentation | 90.47 % | 91.22 % |
| ResNet18 | stronger augmentation (round 2) | 85.17 % | 86.15 % |
| ResNet50 | stronger augmentation (round 2) | 89.83 % | 90.43 % |

**Conclusions**
- ResNet50 beat ResNet18 in every setup (+2.1 … +4.7 %-points) — model size mattered more than augmentation.
- Without augmentation the models memorise the training set (100 % train accuracy by epoch 3–5), but test
  accuracy was the same as with augmentation. Augmentation reduced overfitting but did not raise test accuracy.
- The round-2 "stronger augmentation" made the results **worse**, especially for ResNet18.
- Validation accuracy overestimated test accuracy by 2–7 %-points: the best epoch is picked on a small
  validation set (482 images), so it is an optimistic estimate.
- **N-best:** ResNet50 was 2.3 %-points better at n = 1; the gap shrank to ~1–1.7 %-points for n ≥ 6 but stayed
  above 1 %-point up to n = 10, so within n ≤ 10 there was no clear point where the models stop separating.
- **Hardest species:** *Sapindus saponaria*, mostly confused with *Koelreuteria paniculata* (both in the
  soapberry family). *Liriodendron chinense* was a frequent wrong prediction. Visually distinctive species
  (*Cedrus deodara*, *Salix babylonica*, *Prunus cerasifera* f. *atropurpurea*) were recognised almost perfectly.
- **Dataset problems found:**
  - most photos are stored **sideways** (tree top pointing left, no EXIF rotation tag) → `RandomHorizontalFlip`
    turns these trees upside down instead of mirroring them;
  - 10 validation images have an exact duplicate in the training set;
  - ~2 % of the photos are truncated (large flat grey area at the bottom).
- Suggested next steps: rotate the sideways photos upright, remove damaged and duplicate images, train longer,
  repeat runs with several seeds.

---

## Timofey Krylov — `Tree project - Timofey.Krylov/Tree-Recognition-Project.ipynb`

**Setup**
- Collected all 4 804 images and made an **own random stratified 70 / 15 / 15 split**
  (3 362 / 721 / 721 images) with `train_test_split` (seed 42), instead of the dataset's own split.
- Trained locally on an AMD Radeon GPU (ROCm). The ResNet18 round-1 training crashed with
  `miopenStatusUnknownError`; disabling cuDNN/MIOpen (`torch.backends.cudnn.enabled = False`) was tried as a
  workaround. ResNet50 15 epochs took ~13 min, round 2 ~11–19 min per model.
- The backup notebook also contains extra experiments: a "round 3" and a **round 4** where ResNet50 was
  fine-tuned 5 more epochs from the round-3 weights with a 10× smaller learning rate (1e-5); best val accuracy 94.04 %.

**Results (test set, 721 images)**

| Model | Accuracy | Macro F1 |
|---|---|---|
| ResNet18 round 1 | 88.35 % | 89.14 % |
| ResNet50 round 1 | **93.20 %** | **93.70 %** |
| ResNet18 round 2 | 88.90 % | 89.14 % |
| ResNet50 round 2 | 92.79 % | 93.18 % |

**Conclusions**
- ResNet50 round 1 was the best model (93.20 % accuracy, 93.70 % F1), 4.85 %-points better than ResNet18.
- ResNet50 had the correct species in its top 2 / top 3 / top 5 for 97.1 % / 98.1 % / 98.9 % of test images.
- Round 2 (stronger augmentation + label smoothing + 20 epochs) improved ResNet18 slightly (+0.55 %-points) but
  ResNet50 dropped slightly (−0.41 %-points) → the changes did not improve the best model.
- Most common ResNet50 errors: *Magnolia liliflora* → *Liriodendron chinense*, *Koelreuteria paniculata* →
  *Sapindus saponaria*, *Acer palmatum* → *Ginkgo biloba*.
- Ideas for improvement: EfficientNet, larger image size, cropping the tree better, other augmentation settings.
- **Notes for interpreting these numbers:** because the ResNet18 round-1 run crashed, the ResNet18 confusion
  matrix / per-class report / n-best table in that notebook come from a different (broken) checkpoint than the
  accuracy line (e.g. 5.8 % top-1 in the n-best table vs. 88.35 % in the test line). Also, a random split can put
  the duplicate photos of the dataset into both training and test sets, and the 721-image test set is different
  from the official one, so these scores are not directly comparable with the other notebooks.

---

## Tomi Kitola — `Tree project Tomi/tree_reco_tomi.ipynb`

**Setup**
- Ran the notebook in **Google Colab** (T4 GPU), models saved to Google Drive.
- Own random stratified **70 / 15 / 15 split** (3 362 / 721 / 721 images, seed 42) — same split procedure
  as Timofey's notebook.
- Resizing all photos once to 256 px (shorter side) took ~4 min in Colab, after that training was fast:
  ResNet18 15 epochs ~4.6 min, ResNet50 ~10 min; round 2 (20 epochs) 13 / 16 min.

**Results (test set, 721 images)**

| Model | Accuracy | Macro F1 |
|---|---|---|
| ResNet18 round 1 | 88.90 % | 89.37 % |
| ResNet50 round 1 | 90.57 % | 90.72 % |
| ResNet18 round 2 | 86.69 % | 86.87 % |
| ResNet50 round 2 | **90.98 %** | **91.22 %** |

**Conclusions**
- ResNet50 was better than ResNet18 in both rounds, but the difference at n = 1 was only ~1.7 %-points.
- Training/validation gap was small (a few %-points) → no big overfitting with the tutorial augmentation.
- **N-best:** the models differ by ≥ 1 %-point only for n = 1 and n = 2. From n = 3 on the difference is below
  1 %-point and both models are above 97 %, so **n-best stops separating the models beyond n = 2**.
  For the round-2 models the separation lasted up to n = 3.
- **Repeating error:** *Liquidambar formosana* → *Ginkgo biloba* was the most common mistake for **both**
  ResNet18 and ResNet50 (7 images for ResNet50). *Acer palmatum* → *Ginkgo biloba* and
  *Photinia serratifolia* → *Magnolia grandiflora* also repeated.
- Round 2 changed ResNet50 test accuracy by ~0 %-points and lowered ResNet18 by ~2 %-points.
- Ideas not tried: larger input size (e.g. 320 px), class weights for the smallest classes, cropping the tree
  with an object detector before classification, test-time augmentation, EfficientNet.
- Trained models (individual work): https://drive.google.com/drive/folders/1m3ppYQ8aP8y_xxPuwrz74WMe8Uz7n0Nk?usp=drive_link

---

## What we take from comparing the three notebooks

- **ResNet50 > ResNet18 in every notebook**, but the size of the gap depends a lot on the split and on the run
  (1.7 – 4.9 %-points at n = 1). Single runs on small validation/test sets are noisy: one test image is
  0.14 – 0.21 %-points.
- **The same kind of round-2 changes (bigger crop scale, rotation, ColorJitter, label smoothing) did not help in
  any of the three notebooks.** This pointed us to look at the data itself instead of only the training recipe —
  which is where Jaakko found the sideways photos, duplicates and truncated images.
- The **n-best answer depends on the run**: in Tomi's run the models stopped separating after n = 2, in Jaakko's
  run the gap stayed above 1 %-point up to n = 10. In all runs both models reach ~97–98 % already at n = 3.
- Systematic confusions are between species with similar crowns/leaves (*Sapindus* ↔ *Koelreuteria*,
  *Liquidambar* / *Acer* → *Ginkgo*, *Magnolia liliflora* ↔ *Liriodendron* / *Michelia*).
- For the final notebook we use **Jaakko's version as the base**, because it uses the dataset's official split
  (no hand-made split, comparable with other work on this dataset), and add the data-cleaning round
  (orientation fix, duplicate removal, damaged-image handling) as the model-improvement step.
