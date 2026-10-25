# BT4012 Bitcoin Fraud Detection

This repository contains the reproducible notebook for the final V24 model. The selected Kaggle submissions are V24 (public ROC-AUC 0.94893) and V23 (0.94888).

## Files

- `E1121721.ipynb`: the required student-number notebook. It includes the model source, validation notes, and code to generate the V24 submission and V23 fallback.
- `submission_v24.csv` and `submission_v23_fallback.csv`: the selected submission files.
- `requirements.txt`: Python dependencies.
- `.gitignore`: ignores local data and generated outputs.

## Reproduce

1. Download the official competition data from Kaggle. Keep it outside this repository.
2. Install the dependencies with `pip install -r requirements.txt`.
3. Open and run all cells in `E1121721.ipynb`.

The notebook expects the official `train.csv`, `test.csv`, `sample_submission.csv`, and `txs_edgelist.csv` files, either in the Kaggle input directory or in a data folder. It generates the V24 submission and a V23 fallback in its output directory.

The report discusses the validation and leaderboard results. The chronological validation windows overlap and were reused during model selection, so they are screening evidence rather than independent final-test estimates. Private leaderboard performance may differ from the public score.
