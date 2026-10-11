# Final project files

| File | What it is |
|---|---|
| `Tree-Recognition-Final.ipynb` | Final cleaned notebook for the teacher (based on Jaakko's notebook, runs in Google Colab) |
| `team-conclusions.md` | Each member's own results and conclusions, plus a comparison of the three notebooks |

The individual work notebooks are in the `Tree project - …` folders in the repository root.

## Running the final notebook in Colab

1. Open `Tree-Recognition-Final.ipynb` in Colab (File → Upload notebook, or open it from GitHub).
2. `Runtime → Change runtime type → T4 GPU`.
3. `Runtime → Run all` and allow Google Drive access when asked.
   - The dataset is downloaded with `kagglehub`; it works without a Kaggle login.
   - Six models are trained (ResNet18 / ResNet50 × no augmentation, round 1, round 2), about 45–60 min on a T4.
   - Weights and training histories are saved to `MyDrive/nncv_tree_models_final/`. If the runtime disconnects,
     just "Run all" again: models that are already saved are loaded instead of trained again.
     This is a new folder, so the older models in `nncv_tree_models/` (trained on a different split) are not used.

## Before submitting (Deadline Sunday 11.10.2026)

- [ ] After the Colab run, compare the numbers in these markdown cells with the printed results and update them
      if needed (a GPU run is never exactly reproducible): **Training observations** (section 7),
      **No augmentation vs. augmented** (8.1), **N-best finding** (9), **What we noticed** (10.1) and
      **Conclusions** (end). They currently quote the reference run from Jaakko's notebook.
- [ ] Share the `nncv_tree_models_final` Drive folder ("anyone with the link") and paste the link at the end of the
      notebook (`_(Google Drive link here)_`).
- [ ] Word report: results tables, confusion-matrix findings, n-best answer, repeating errors and how to fix them
      (see the conclusions in the notebook and `team-conclusions.md`).
- [ ] Zip: final notebook + the three individual notebooks + Word report. **No `.pth` model files.**
