# Detecting Patronizing & Condescending Language (PCL) with RoBERTa-large

A binary classifier for **SemEval-2022 Task 4, Subtask 1**: given a paragraph from a news article, predict whether it contains patronizing or condescending language towards vulnerable communities ([Pérez-Almendros et al., 2020](https://aclanthology.org/2020.coling-main.518/)).

The dataset is small (~10k paragraphs) and heavily imbalanced (~9.5 : 1 negative-to-positive), and PCL is often subtle: the same vulnerability keywords ("homeless", "refugee", ...) appear in both classes, and what differs is the framing around them. The pipeline tackles that with RoBERTa-large, contextual metadata, an $\alpha$-balanced focal loss combined with layer-wise learning-rate decay, and a 5-seed soft-voting ensemble.

**Result:** F1 = **0.610** on the official test set, which would have placed **13th of 77 teams** in the original competition (the organizers' baseline scored 0.491).

📄 **Full write-up:** [`docs/report.pdf`](docs/report.pdf), covering the EDA, method rationale, ablations and error analysis.

---

## Results

### Ablation (internal held-out test split)

| Model                                                      | Threshold $\tau$ | AUPRC     | Precision (PCL) | Recall (PCL) | F1 (PCL)  |
|------------------------------------------------------------|------------------|-----------|-----------------|--------------|-----------|
| Baseline (RoBERTa-large + cleaning + metadata + bias init) | 0.10             | 0.615     | 0.558           | 0.677        | 0.612     |
| + LLRD only                                                | 0.02             | 0.605     | 0.547           | 0.707        | 0.617     |
| + Focal loss only                                          | 0.48             | 0.650     | 0.726           | 0.455        | 0.559     |
| + LLRD + Focal loss                                        | 0.59             | 0.622     | 0.647           | 0.647        | 0.647     |
| **5-seed ensemble (soft voting)**                          | 0.49             | **0.670** | 0.618           | 0.687        | **0.651** |

Focal loss and LLRD fail in opposite ways when used alone. Focal loss fixes calibration and precision but overfits, while LLRD recovers recall but leaves the model miscalibrated. Used together, they give a balanced, well-calibrated model, and ensembling over seeds reduces variance on top of that.

### Official test set

These are the ensemble's results on the official SemEval test set (3,832 paragraphs, labels hidden during development):

| TP  | FP  | TN   | FN  | Precision | Recall | F1 (PCL)  |
|-----|-----|------|-----|-----------|--------|-----------|
| 208 | 157 | 3358 | 109 | 0.570     | 0.656  | **0.610** |

Here is how that score compares with the original [SemEval-2022 Subtask 1 leaderboard](https://aclanthology.org/2022.semeval-1.38/) (77 teams):

| Rank   | Team                                | F1 (PCL)  |
|--------|-------------------------------------|-----------|
| 1      | PALI-NLP                            | 0.651     |
| 2      | STCE                                | 0.650     |
| 3      | ymf924                              | 0.647     |
| ⋮      | ⋮                                   | ⋮         |
| 11     | Leo_team                            | 0.620     |
| 12     | PAI-Team                            | 0.617     |
| **13** | **This project (5-seed ensemble)**  | **0.610** |
| 13     | Anonymus                            | 0.608     |
| 14     | BLING                               | 0.593     |
| ⋮      | ⋮                                   | ⋮         |
| 43     | *Organizers' RoBERTa-base baseline* | *0.491*   |

The ensemble would have placed **13th of 77 teams**, within 0.041 F1 of the winner and +0.119 above the official baseline, using a single architecture without external data, extra pre-training or large hyperparameter searches.

*This project was built after the competition closed, so it was never an official entry. Its score was computed on the same official test set.*

---

## Method

**Data & preprocessing** (`preprocessing.py`)
- Text normalization: fixes encoding errors, strips HTML, masks URLs and user handles as `[URL]` / `[USER]`, repairs tokenization artifacts (e.g. `ca n't` $\to$ `can't`), and drops fragments shorter than 5 words.
- **Contextual metadata injection:** EDA showed that vulnerability keywords appear in both classes, and that PCL skews towards particular countries (a long tail of Global South nations). The targeted keyword and country code are therefore prepended to every input: `[KEYWORD] homeless [COUNTRY] jm [SEP] <text>`.

**Model & training** (`train.py`)
- **RoBERTa-large** (355M parameters), max length 256 tokens.
- **Prior-probability bias initialization:** the classifier bias starts at $−\log((1 − \pi) / \pi) \approx −2.25$ ($\pi \approx 0.095$), so early epochs aren't spent learning the base rate.
- **α-balanced focal loss** ($\gamma = 1$, $\alpha = 0.75$). $\gamma$ is lowered from the usual 2 so that hard negatives (empathetic but *not* condescending text) still give a useful learning signal.
- **Layer-wise learning-rate decay** ($\xi = 0.95$ per layer): the top layers adapt at the full LR, while the embeddings keep about 29% of it, which protects pre-trained syntax.
- Early stopping on validation **AUPRC**. The decision threshold is tuned for F1 on the validation split.
- **5-seed ensemble** with soft voting: probabilities are averaged before the threshold is applied.

<details>
<summary>Full hyperparameters</summary>

| Hyperparameter                  | Value                                         |
|---------------------------------|-----------------------------------------------|
| Batch size × grad. accumulation | 8 × 4                                         |
| Precision                       | BF16                                          |
| Optimizer                       | AdamW, LR 1e-5, weight decay 0.01, warmup 0.1 |
| LLRD decay $\xi$                | 0.95                                          |
| Focal loss $\gamma$ / $\alpha$  | 1.0 / 0.75                                    |
| Bias init                       | −2.25                                         |
| Max epochs / patience           | 10 / 3                                        |

</details>

**Evaluation protocol.** The official test labels are hidden, and reusing the official dev set for both model selection and final evaluation would leak information. To maximize data usability, the official train and dev sets were merged and re-split **80 / 10 / 10 (stratified)**:
- Train: used for training only.
- Validation: used for early stopping, model selection and threshold calibration.
- Test: never touched until the final evaluation. All numbers in the ablation table above come from this split.

Because of this split, scores on the *official* dev set are not meaningful: the training data includes it.

---

## Repository structure

```text
.
├── data/                   # Raw dataset goes here (not tracked, see Reproducing); splits go to data/processed/
├── docs/
│   └── report.pdf          # Full write-up
├── experiments/            # Checkpoints (not tracked)
├── figures/                # EDA plots, PR curves, confusion matrices
├── models/                 # Base roberta-large weights (not tracked)
├── eda.ipynb               # Exploratory data analysis
├── evaluation.ipynb        # Metrics, figures, error analysis, prediction export
├── preprocessing.py        # Cleaning + stratified 80/10/10 split
├── train.py                # Training (focal loss, LLRD, bias init)
├── run_experiment.sh       # Single training run
├── run_ablation.sh         # Baseline / LLRD / focal / LLRD+focal
├── train_ensemble.sh       # 5-seed ensemble
└── requirements.txt
```

---

## Reproducing

```bash
pip install -r requirements.txt

# Base model
hf download FacebookAI/roberta-large --local-dir models/roberta-large
```

**Data.** The dataset is not redistributed here. Request it through the [official repository](https://github.com/Perez-AlmendrosC/dontpatronizeme); the official test set is released separately on request. Place these files in `data/`:

```text
data/
├── dontpatronizeme_pcl.tsv
├── dontpatronizeme_categories.tsv
├── train_semeval_parids-labels.csv
├── dev_semeval_parids-labels.csv
└── task4_test.tsv
```

Then run:

```bash
python preprocessing.py        # cleans text and writes the stratified splits to data/processed/
```

**Pre-trained ensemble (optional).** The 5 fine-tuned checkpoints are too large for git. Download [`ensemble_llrd_focal`](https://drive.google.com/drive/folders/19gNvMRX2Q69O-oynGhGyQ38x26tCTX70) and put it at `experiments/ensemble_llrd_focal/`.

**Train.**

```bash
bash run_ablation.sh           # the 4 ablation models
bash train_ensemble.sh         # optional: retrain the 5-seed ensemble
```

**Evaluate.** Run `evaluation.ipynb` to compute metrics on the internal splits, generate PR curves and confusion matrices, and export predictions. It needs both the ablation checkpoints and the ensemble.

### Hardware

Everything was trained on Apple Silicon (MPS), and the code falls back to CPU if MPS is unavailable. To run on an NVIDIA GPU:
1. `run_experiment.sh`: add `--dataloader_pin_memory True` to the `train.py` call.
2. `evaluation.ipynb` and the t-SNE cell in `eda.ipynb`: change the device line to
   `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.

---

## Error analysis

- **Wins:** the ensemble picks up "savior" framing that is buried in long, diplomatic sentences, where the baseline missed it (8.6% $\to$ 90%).
- **Misses:** with a higher, calibrated threshold, implicit PCL that strips people of agency but uses no savior vocabulary can slip through.
- **Shared false positives:** sincere political advocacy ("do better for the most vulnerable") fools both models. They respond to the *vocabulary* of vulnerability more than to pragmatic intent.

See Section 5 of the [report](docs/report.pdf) for details.

---

## References

- C. Pérez-Almendros, L. Espinosa Anke, S. Schockaert. *Don't Patronize Me! An Annotated Dataset with Patronizing and Condescending Language towards Vulnerable Communities.* COLING 2020.
- C. Pérez-Almendros, L. Espinosa Anke, S. Schockaert. *SemEval-2022 Task 4: Patronizing and Condescending Language Detection.* SemEval 2022.
- D. Mildenberger et al. *A Tale of Two Classes: Adapting Supervised Contrastive Learning to Binary Imbalanced Datasets.* CVPR 2025.
