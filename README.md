# BT4012 Bitcoin Fraud Detection

Final selected public leaderboard submissions:
- V24: ROC-AUC 0.94893
- V23: ROC-AUC 0.94888

V24 is the top-submission model documented in E1121721.ipynb. The notebook embeds the pipeline source and describes the data inputs and reproduction steps. Supporting source modules and dependency ranges are included here.

V24 uses a rank blend of the LightGBM-based V12 anchor and two recent-window, eight-category CatBoost experts, followed by a small predecessor-rank graph correction. Unknown training labels are excluded from supervised fitting. No external data or pseudo-labels are used.

The report uses V1 (public AUC 0.94133) as the project's initial public-score benchmark. The V23/V24 validation windows are reused and overlapping, so they are screening evidence rather than independent test estimates. The final private leaderboard score may differ from the public score.

## Reproduce

1. Download the official competition data from Kaggle. Keep the dataset out of this repository.
2. Install dependencies from requirements.txt.
3. Run the notebook or use train_v24.py --data /path/to/data --out ./V24_run.

The data input must include train.csv, test.csv, sample_submission.csv, and txs_edgelist.csv. The repository does not include the competition data or cached fitted models.

## Files

- BT4012_Report_Draft.docx: editable report draft.
- E1121721.ipynb: reproducible notebook named with the NUS student number.

- train_v*.py, graph_postprocess.py: reproducible training and graph-ranking source.
- submission_v24.csv, submission_v23_fallback.csv: selected final outputs.
