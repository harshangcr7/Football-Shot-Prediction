## Contextual Feature Engineering for Probabilistic Football Shot Outcome Prediction

- Built a **leakage-free expected goals (xG) model** using StatsBomb event data
- Engineered **contextual, spatial, and temporal features** from raw JSON events
- Trained and evaluated **CatBoost and XGBoost** models for probabilistic shot prediction
- Achieved **Goal F1 ≈ 0.68** using only pre-shot information
- Conducted **feature ablation**, showing spatial-only models fail without context
- Verified **probability reliability** via calibration curves and a low **Brier score (0.053)**
- Focused on **model validity, calibration, and interpretability**, not metric gaming

## Architecture & Workflow

The project follows a modular, end-to-end machine learning workflow:

1. **Data Ingestion**
   - Fetch event-level football data using the `statsbombpy` API
   - Parse nested JSON structures into tabular format

2. **Data Filtering**
   - Select a subset of matches to reduce runtime
   - Extract only shot-related events

3. **Feature Engineering**
   - Spatial features (e.g., shot distance, shot angle)
   - Contextual features (game state, pressure, shot type)
   - Temporal features (match period, time since last event)
   - Positional and execution-based features (one-hot encoded)

4. **Modeling**
   - Binary probabilistic prediction (goal vs. no goal)
   - Gradient-boosted decision trees (CatBoost as final baseline)
   - Early stopping to prevent overfitting

5. **Evaluation**
   - Class-wise Precision, Recall, and F1-score
   - Confusion matrices
   - Feature ablation (contextual vs. spatial-only)
   - Calibration analysis (reliability curves, Brier score)

6. **Analysis & Interpretation**
   - Emphasis on probabilistic validity
   - Explicit avoidance of target leakage
   - Clear separation between prediction and causality

## Results Summary - CatBoost (no leakage, no SMOTE)

| Experiment | Precision (Goal) | Recall (Goal) | F1 (Goal) | Notes |
|----------|------------------|---------------|-----------|------|
| Final CatBoost (no leakage, no SMOTE) | 0.74 | 0.63 | **0.68** | Selected baseline |
| Spatial-only Ablation | 0.00 | 0.00 | **0.00** | Model collapses without context |
| XGBoost (default threshold) | ~0.71 | ~0.65 | ~0.68 | Comparable robustness |
| XGBoost (recall-oriented threshold) | 0.63 | **0.82** | 0.71 | Decision-level trade-off |
| Calibration (CatBoost) | — | — | — | Brier score = **0.053** |

