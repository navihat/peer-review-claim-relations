# ReviewSynth NLI — Relation Classifier for Peer Review Claims

Fine-tuned **6-class relation classifier** that decides how two atomic claims from different peer reviewers of the same paper relate to each other.

Weights are published at **[navihat/reviewsynth-nli](https://huggingface.co/navihat/reviewsynth-nli)** (mDeBERTa-v3-base, 0.3 B, safetensors).

---

## Label space

```
AGREEMENT — PARTIAL_AGREEMENT — COMPLEMENTARY — PARTIAL_CONTRADICTION — CONTRADICTION
                                      ⊥
                                  UNRELATED
```

| Label | One-line definition |
|---|---|
| `AGREEMENT` | Same specific point, same direction, same strength |
| `PARTIAL_AGREEMENT` | Same direction, different scope or certainty |
| `COMPLEMENTARY` | Different specific points, but both target **the same underlying issue** |
| `PARTIAL_CONTRADICTION` | Same issue, one hedges where the other is firm |
| `CONTRADICTION` | Same specific point, logically incompatible — both cannot be true |
| `UNRELATED` | No shared issue |

The full decision tree is in [`docs/RUBRIC_GOLD.md`](docs/RUBRIC_GOLD.md).

---

## Quick start

### Load from HuggingFace

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

LABELS = ["AGREEMENT", "PARTIAL_AGREEMENT", "COMPLEMENTARY",
          "PARTIAL_CONTRADICTION", "CONTRADICTION", "UNRELATED"]

tok   = AutoTokenizer.from_pretrained("navihat/reviewsynth-nli")
model = AutoModelForSequenceClassification.from_pretrained(
            "navihat/reviewsynth-nli").eval()

def predict(left: str, right: str) -> str:
    """Symmetric TTA: average logits over both orderings before argmax."""
    with torch.no_grad():
        logits = sum(
            model(**tok(a, b, return_tensors="pt",
                        truncation=True, max_length=160)).logits
            for a, b in [(left, right), (right, left)]
        ) / 2
    return LABELS[logits.argmax().item()]

print(predict(
    "The method is well-motivated and addresses a real gap.",
    "I fail to see why this approach is needed over prior work."
))
# → CONTRADICTION
```

> **Always pass both orderings and average the logits.** Skipping symmetric TTA degrades accuracy on order-sensitive examples (this was the root failure of the LLM ensemble used to build the dataset).

### Inference server

```bash
pip install fastapi uvicorn torch transformers

# point MODEL_DIR at the local checkpoint or the HF repo id
MODEL_DIR=navihat/reviewsynth-nli uvicorn src.serve.app:app --port 8000
```

**Single pair:**
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"left": "The experiments are thorough.", "right": "Key baselines are missing."}'
```

**Batch (contract v1.0):**
```bash
curl -s -X POST http://localhost:8000/v1/relations:predict \
  -H "Content-Type: application/json" \
  -d '{
    "schema_version": "1.0",
    "request_id": "req-001",
    "pairs": [
      {"pair_id": "p1",
       "left":  {"text": "The method is well-motivated."},
       "right": {"text": "I fail to see why this is needed."}}
    ]
  }'
```

---

## Training data

**989 claim pairs** extracted from ICLR / NeurIPS peer reviews and labeled with Claude using a fixed rubric ([`docs/RUBRIC_GOLD.md`](docs/RUBRIC_GOLD.md)).

| Source | n | How |
|---|---:|---|
| `full_batch_manual` | 800 | Claude reads each pair against rubric |
| `mined_stance_opposition` | 150 | Mined POSITIVE-vs-NEGATIVE pairs, then labeled |
| `fewshot_human_pairs` | 39 | Human-curated examples, labels by Claude |
| **Total** | **989** | |

Split by **paper group** to prevent claim leakage between train and eval:

| Split | Pairs | Papers |
|---|---:|---:|
| train | 788 | 177 |
| val | 100 | 15 |
| test | 101 | 18 |

Label distribution in train:

| Label | n | % |
|---|---:|---:|
| UNRELATED | 273 | 34.6 |
| PARTIAL_AGREEMENT | 173 | 22.0 |
| PARTIAL_CONTRADICTION | 134 | 17.0 |
| COMPLEMENTARY | 122 | 15.5 |
| AGREEMENT | 61 | 7.7 |
| CONTRADICTION | 25 | 3.2 |

The high UNRELATED share reflects how the pairs were mined (same paper, same broad aspect → many claims that share a topic but not a specific issue).

---

## Performance

All silver labels are Claude-generated. Numbers measure agreement with those labels, not absolute ground truth.

| | macro-F1 |
|---|---:|
| Majority baseline (always predict most-common class) | 0.10 |
| Stance rule (3-line heuristic) | 0.19 |
| **Fine-tune — 5-fold CV on silver (989 pairs)** | **~0.28–0.36** |
| **Gold test — 129 pairs, Vietnamese, out-of-domain** | **0.25** |

The gold test set (`pipeline_data/golden_set/gold_test.jsonl`) contains 129 **human-verified** pairs from a different domain (Vietnamese university grant reviews) and a different language. The cross-lingual / cross-domain drop from the English test split to this gold set is ~0.04, which indicates strong multilingual transfer from the mDeBERTa-v3 backbone.

**Flip-rate** (how often the model changes its prediction when the two claims are swapped) on raw logits: **~10%**, down from 44–50% for the raw LLM labelers used during data construction.

Per-class F1 (5-fold CV, silver):

| Label | F1 |
|---|---:|
| COMPLEMENTARY | 0.58 |
| PARTIAL_AGREEMENT | 0.48 |
| PARTIAL_CONTRADICTION | 0.35 |
| AGREEMENT | 0.29 |
| UNRELATED | 0.13 † |
| CONTRADICTION | 0.03 † |

† UNRELATED F1 is suppressed by prior mismatch (see Limitations). CONTRADICTION has only 25 train / 2 test samples — read its F1 from CV, not from the test split.

---

## Design decisions

**Symmetric training + inference.** The relation between claim A and claim B must not depend on which is listed first. Three mechanisms enforce this:

| Mechanism | Where |
|---|---|
| Augment both orderings | Training: each pair appears as `(A,B)` and `(B,A)` with the same label |
| Symmetric TTA | Inference: logits from both orderings are **averaged** before argmax |
| Flip-rate metric | Measured on **raw logits** (TTA disabled) to verify the model learned symmetry, not just masked by TTA |

**Class weights.** `CrossEntropyLoss(weight=inverse_frequency)` so the model does not ignore rare classes. CONTRADICTION receives ~5–6× weight.

**Paper-grouped split.** Five folds stratified by label, groups defined at the paper level. The 39 few-shot examples are pinned to train in every fold (`fold = -1`) because their labels defined the rest of the dataset.

---

## Limitations

- **Labels are LLM-generated.** No independent annotator → no human ceiling. A macro-F1 of 0.25 cannot be interpreted without knowing inter-annotator agreement on the same task.
- **CONTRADICTION is rare** (25/989 train samples). F1 on the 2-sample test split is not informative; use 5-fold CV results.
- **Prior mismatch on deployment data.** UNRELATED is 35% of training data but only 14% of the gold test set. If your deployment distribution differs, consider applying a prior correction to the logits.
- **Training domain:** English ML conference reviews. Transfer to other peer-review venues or languages will vary; the gold test (Vietnamese grant reviews, 0.25 macro-F1) gives one data point.
- **No second annotator.** The only human annotation is the 129-pair gold set. All 989 training labels are from Claude.

---

## Repository layout

```
src/
  serve/app.py              FastAPI inference server — /predict and /v1/relations:predict
  train/
    split.py                Paper-grouped stratified split + 5-fold assignment
    train.py                Fine-tune loop (symmetric aug, class weights, early stopping)
  data/
    relabel_complementary.py  Re-labeling COMPLEMENTARY → UNRELATED per gold contract
  fewshot/                  Few-shot set construction

notebooks/
  01_data_pipeline.ipynb          Data pipeline (claim extraction + LLM labeling, Colab)
  02_train_relation_classifier.ipynb  Training + 7-metric evaluation suite (Colab)

pipeline_data/
  processed/trackB_silver.jsonl     989 labeled pairs (current silver set)
  processed/splits/                 train / val / test splits + folds.json + class_weights.json
  reports/split_report.md           Label distribution and leakage audit

docs/
  PLAN.md          Full data-pipeline design and labeling strategy
  TRAIN.md         Training decisions, results, known issues (sections 9–12)
  RUBRIC_GOLD.md   Gold-set label contract — the decision tree used for annotation
  FEWSHOT.md       How the 39 few-shot examples were built
  CHECKLIST.md     Pipeline execution checklist
```

---

## Citation

```bibtex
@misc{reviewsynth-nli-2026,
  author = {Truong Van Thai},
  title  = {ReviewSynth NLI: 6-class Relation Classifier for Peer Review Claims},
  year   = {2026},
  url    = {https://huggingface.co/navihat/reviewsynth-nli}
}
```
