# Fraud-Probability Reinsurance with a Soft-Voting Ensemble

Undergraduate thesis (*Tugas Akhir*). The code comments are in Indonesian.

The question: can an insurer cut its total claims cost by buying reinsurance against **fraudulent claims**, priced from a machine-learning fraud probability rather than a flat rate?

1. A **soft-voting (SV) ensemble** of AdaBoost, Gradient Boosting and XGBoost predicts the probability that an auto insurance claim is fraudulent.
2. Those probabilities price a reinsurance premium. Four reinsurance designs are then compared against having no reinsurance at all, on claims the model never saw.

## Data

`insurance_claims.csv`: 1,000 auto insurance claims with 39 columns covering policy, insured person, incident, vehicle and claim amounts. Incidents run from 1 Jan to 1 Mar 2015, and 247 claims (24.7%) are fraudulent.

The data is split by date to mimic real use:

| Set | Incident dates | Claims | Purpose |
|-----|----------------|--------|---------|
| Model period | ≤ 2015-02-20 | 846 | Train the classifier and price the reinsurance premium |
| Actual period | > 2015-02-20 (9 days) | 154 | Out-of-sample test: score the classifier and compare losses |

## Method

**Preprocessing**
- `?` values imputed with the mode.
- Location and policy-ID style columns dropped.
- Age, tenure, premium and deductible binned.
- Hobbies collapsed to chess / cross-fit / other.
- `injury_claim`, `property_claim` and `vehicle_claim` dropped because they are highly correlated with `total_claim_amount`.
- Label encoding applied.

**Classifier.** Each base model was tuned with a grid search, then combined by soft voting:

| Model | Tuned parameters |
|-------|------------------|
| AdaBoost | `DecisionTree(max_depth=2)`, `learning_rate=0.23`, `n_estimators=8` |
| GBDT | `learning_rate=0.01`, `max_depth=1`, `n_estimators=471` |
| XGBoost | `learning_rate=0.01`, `max_depth=1`, `n_estimators=525` |
| **Soft voting** | weights AB : GBDT : XGB = **45 : 16 : 39**, found by grid search over weights summing to 100 |

**Reinsurance cases.** Every case is evaluated over the 9 days of the actual period. The daily premium is the expected fraud loss per day in the model period:

| Case | Insurer retains | Reinsurance premium per day |
|------|-----------------|-----------------------------|
| 1. No reinsurance | every claim | — |
| 2. Quota share | `Σ claim × (1 − p_fraud)` | `Σ claim × p_fraud` / days |
| 3. Threshold | claims with `p_fraud < t` | `Σ claim × 1[p_fraud ≥ t]` / days |
| 4. Aggregate probability | `Σ claim × (1 − p̄)` | `Σ claim × p̄` / days, where p̄ is the mean, median or mode of `p_fraud` over claims with `p_fraud > 0.4` |

## Results

**Classifier.** The tuning notebook uses a 70/30 split of all 1,000 claims (300 in the test set):

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| AdaBoost (tuned) | 0.847 | 0.643 | 0.851 | 0.733 |
| GBDT (tuned) | 0.823 | 0.613 | 0.770 | 0.683 |
| XGBoost (tuned) | 0.830 | 0.621 | 0.797 | 0.698 |
| **Soft voting** | **0.853** | **0.647** | **0.892** | **0.750** |

On the held-out actual period (154 claims), the SV model trained only on the model period scores:
- accuracy 0.792
- precision 0.458
- recall 0.786
- F1 0.579

It catches most fraud but flags many legitimate claims along the way.

**Total loss over the actual period:**

| Case | Total loss | vs. no reinsurance |
|------|-----------:|-------------------:|
| 1. No reinsurance | 7,604,330 | — |
| 2. Quota share | 7,724,222 | +1.6% |
| 3. Threshold t = 0.6 | **7,390,236** | **−2.8%** |
| 3. Threshold t = 0.8 | 7,559,632 | −0.6% |
| 4. Aggregate (mean / median / mode) | 7,759,280 / 7,723,910 / 7,721,251 | +1.5% to +2.0% |

- **Threshold reinsurance at t = 0.6** is the only design that clearly beats having no reinsurance.
- Pricing on a continuous probability (cases 2 and 4) overpays, because the premium charges for every claim's small fraud probability.
- **Caveats:**
  - The threshold is chosen on the same 9-day period it is scored on.
  - The model period is only 7 weeks long.
  - Treat these results as a proof of concept, not a pricing recommendation.

## Files

| File | Contents |
|------|----------|
| `01_hyperparameter_tuning.ipynb` | EDA, preprocessing, grid searches, base-model and ensemble evaluation, threshold sweeps |
| `02_final_prediction.ipynb` | Date split, final SV model, fraud probabilities written to the two `insurance_claims_with_probs` CSVs |
| `03_reinsurance_model.ipynb` | The four reinsurance cases |
| `insurance_claims.csv` | Raw claims data |
| `insurance_claims_with_probs(846).csv` / `(154).csv` | Model period and actual period claims, each with `fraud_prob` |
| `requirements.txt` | Pinned environment that reproduces the saved outputs |

## Running

```bash
uv venv -p 3.10 && source .venv/bin/activate   # or any Python 3.10 environment
pip install -r requirements.txt
jupyter lab
```

- `03_reinsurance_model.ipynb` runs on its own from the committed CSVs.
- Re-running `02_final_prediction.ipynb` regenerates the CSVs. With the pinned versions it gives the same values (checked in 2026).
- Use the pinned versions. Current pandas, NumPy and scikit-learn break the notebooks, and current pandas silently changes the data by reading `None` in `authorities_contacted` as missing.
- The written thesis is not in this repo.
