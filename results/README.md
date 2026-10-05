# Results

This folder intentionally starts without claimed performance numbers.

Run:
`python -m src.evaluate --config configs/config.yaml --checkpoint models/best_model.pth`

Then use `metrics.json` as the source for the final report.

For the submission, include:
- Top-1 / Top-5 / Top-10 retrieval
- Verification accuracy, precision, recall, F1, ROC-AUC
- Threshold selected on validation data
- Gallery/query counts
- Training configuration
- Efficiency metrics if claiming the bonus

Never insert made-up metrics.
