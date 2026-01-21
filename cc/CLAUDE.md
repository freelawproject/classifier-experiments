# Claude Experiment: Automated Annotation Workflow

## Overview

This experiment explores automating the human annotation workflow for the docket-entry classifier project. The goal is to have Claude perform the iterative steps that humans currently do—creating synthetic annotations, grounding decision boundaries, generating training sets, and preparing data for BERT fine-tuning.

## Project Context

- **Target Project**: `docket-entry` - classifying docket entries (motions, orders, etc.)
- **Pipeline**: Synthetic annotations → Human grounding → AI annotation of training sets → BERT fine-tuning

## Note on Dictation

The user often uses dictation, so be forgiving of typos and odd formatting in spoken instructions. Defer to what's written in the code for canonical names and spellings (e.g., "Claude" may be transcribed as "Cloud").

## Important: Import Pattern

Always import models using the shim pattern:
```python
from clx.models import Label, LabelHeuristic, LabelDecision, Project
```
NOT `from clx.app.models import ...`. The shim at `clx/models.py` auto-initializes Django.

---

## Step 1: Creating Heuristics for a Label

### Purpose

Heuristics partition the corpus into three buckets for efficient annotation:
- **Excluded**: High-confidence negatives (don't meet minimal conditions)
- **Neutral**: Uncertain cases (meet minimal but not likely conditions)
- **Likely**: High-confidence positives (meet both minimal and likely conditions)

### Two Types of Heuristics

#### 1. Query String Heuristics

Simple keyword conditions using a mini-language:

| Operator | Meaning | Example |
|----------|---------|---------|
| `,` | AND (all must match) | `motion, court` |
| `\|` | OR (any can match) | `motion\|filing` |
| `~` | NOT (negation) | `~denied` |
| `^` | Starts with | `^Summary` |

**Precedence**: ORs are nested within ANDs. So `a, b|c` means `(a) AND (b OR c)`.

**All matching is case-insensitive.**

Examples:
- `complaint` - contains "complaint" anywhere
- `^motion` - starts with "motion"
- `motion, ~denied` - contains "motion" AND does not contain "denied"
- `motion|application|request` - contains any of these terms
- `^motion, court|judge` - starts with "motion" AND contains "court" or "judge"

#### 2. Custom Function Heuristics

For complex logic, define a decorated function in `clx/app/custom_heuristics.py`:

```python
from clx.app.custom_heuristics import custom_heuristic

def within_first(text, term, n):
    """Helper: check if term appears in first n words."""
    first_n = " ".join(text.split()[:n])
    return term in first_n

@custom_heuristic("docket-entry", "Motion")
def first_3_motion(text, **kwargs):
    """Matches if 'motion' appears in the first 3 words."""
    return within_first(text.lower(), "motion", 3)
```

The decorator registers the function with:
- `project_id`: Which project this applies to
- `label_name`: Which label this is a heuristic for
- The function receives `text` and must return `True`/`False`

After adding custom heuristics, sync them:
```python
LabelHeuristic.sync_custom_heuristics()
```

### Minimal vs Likely Conditions

**Minimal Conditions** (`is_minimal=True`):
- Define what MUST be true for a positive example
- Used to exclude obvious negatives
- Should be conservative—avoid false exclusions
- Example: A "Complaint" should contain "complaint" (or common misspellings)

**Likely Conditions** (`is_likely=True`):
- Define patterns that strongly suggest a positive
- Used to identify easy positive cases
- Can be more aggressive
- Example: Text starting with "Complaint" is very likely a complaint

### The Three Buckets Logic

```
EXCLUDED = does not match ANY minimal heuristic
NEUTRAL  = matches at least one minimal BUT no likely heuristics
LIKELY   = matches at least one minimal AND at least one likely heuristic
```

### Creating and Managing Heuristics

```python
from clx.models import Label, LabelHeuristic

# Get the label
label = Label.objects.get(project_id="docket-entry", name="Motion")

# View existing heuristics
for h in label.heuristics.all():
    print(f"ID: {h.id}")
    print(f"  Query: {h.querystring or h.custom}")
    print(f"  is_minimal: {h.is_minimal}, is_likely: {h.is_likely}")
    print(f"  Matches: {h.num_examples}, Applied: {h.applied_at}")

# Create a querystring heuristic
heuristic = LabelHeuristic.objects.create(
    label=label,
    querystring="motion|application|request",
    is_minimal=True,
)

# Apply the heuristic (computes across corpus)
heuristic.apply()

# Create a likely heuristic
likely_heuristic = LabelHeuristic.objects.create(
    label=label,
    querystring="^motion",
    is_likely=True,
)
likely_heuristic.apply()

# Check bucket counts
label.refresh_from_db()
print(f"Excluded: {label.num_excluded}")
print(f"Neutral: {label.num_neutral}")
print(f"Likely: {label.num_likely}")
```

### Guidelines for Claude

1. **Start simple** - Don't overthink. Simple keyword matches work well.

2. **Minimal conditions should be conservative**:
   - Ask: "Could there ever be a positive example that doesn't match this?"
   - If yes, broaden the condition or add alternatives with `|`
   - Include common misspellings, abbreviations, synonyms

3. **Likely conditions can be aggressive**:
   - These just identify easy cases, not all cases
   - Prefix matches (`^term`) are often good likely conditions

4. **Iterate based on counterexamples**:
   - If you find a positive example in the "excluded" bucket → expand minimal condition
   - If you find obvious positives in "neutral" → add likely conditions

5. **Multiple heuristics combine with OR**:
   - Multiple minimal heuristics: excluded if matches NONE of them
   - Multiple likely heuristics: likely if matches ANY of them

---

## Step 2: Create Annotation Decisions

### Purpose

Decisions are reason-annotated examples that define decision boundaries. They serve two purposes:
1. Document where we're drawing the line on edge cases
2. Provide training examples for the GEPA predictor optimization

### What Makes a Good Decision

- **Keep it minimal**: Humans should be able to review all decisions and understand the labeling policy
- **Include obvious examples**: At least one clear positive example ("This is obviously a complaint")
- **Focus on edge cases**: Where the boundary isn't obvious
- **Short reasons**: 1-2 sentences explaining why

### Examples of Good Decisions

For a "Complaint" label:
- **Positive**: "Complaint for Damages" → `True`, "This is clearly a complaint filing"
- **Negative**: "Submission of Complaint as Exhibit" → `False`, "This references a complaint but is not the complaint itself"
- **Negative**: "Response to Complaint" → `False`, "This is a response document, not the complaint"

For a "Motion" label:
- **Positive**: "Motion for Summary Judgment" → `True`, "Standard motion filing"
- **Positive**: "Application for Extension of Time" → `True`, "Applications that request court action are functionally motions"
- **Negative**: "Opposition to Motion" → `False`, "This opposes a motion but is not itself a motion"

### Creating Decisions

```python
from clx.models import Label, LabelDecision
from clx import generate_hash

label = Label.objects.get(project_id="docket-entry", name="Motion")

# View existing decisions
for d in label.decisions.all():
    print(f"Value: {d.value}")
    print(f"Text: {d.text[:100]}...")
    print(f"Reason: {d.reason}")
    print()

# Create a decision from text
text = "Motion for Summary Judgment filed by Defendant"
decision = LabelDecision.objects.create(
    label=label,
    text_hash=generate_hash(text),
    text=text,
    value=True,
    reason="Standard motion filing requesting summary judgment"
)

# Or create from a search result (has text_hash already)
# See Search section for how to find examples
```

### Guidelines for Claude

1. **Start with 1-2 obvious decisions** per label
2. **Add edge case decisions as you encounter them** during review
3. **Keep reasons brief but clear** - they'll be used for predictor training
4. **Update decisions if needed** - the same text_hash will update the existing decision

---

## Step 3: Sample the Training Set

### Purpose

The training set is a diverse sample of examples used for:
- Running predictor inference
- Training fine-tuned BERT models
- Evaluating model performance

### How Sampling Works

The trainset samples from multiple sources to ensure diversity:

1. **Heuristic buckets**: Random samples from excluded, neutral, and likely buckets
2. **Decision neighbors**: Semantic neighbors of each decision (finds similar edge cases)

Default configuration (configurable per label):
- `trainset_num_excluded`: 1000 examples from excluded bucket
- `trainset_num_neutral`: 1000 examples from neutral bucket
- `trainset_num_likely`: 1000 examples from likely bucket
- `trainset_num_decision_neighbors`: 50 neighbors per decision

The sampling uses "mesh sort" to select diverse examples (not just random).

### Train vs Eval Split

- **Train split**: Main sample (ratio=1.0)
- **Eval split**: Smaller sample (ratio=0.2) for evaluation

### Updating the Trainset

```python
from clx.models import Label

label = Label.objects.get(project_id="docket-entry", name="Motion")

# Configure sampling parameters (optional)
label.trainset_num_excluded = 1000
label.trainset_num_neutral = 1000
label.trainset_num_likely = 1000
label.trainset_num_decision_neighbors = 50
label.save()

# Sample the trainset
label.update_trainset()

# Check what was sampled
print(f"Train examples: {label.trainset_examples.filter(split='train').count()}")
print(f"Eval examples: {label.trainset_examples.filter(split='eval').count()}")
```

### When to Resample

Resample the trainset when:
- You add new decisions (to include their neighbors)
- You change heuristics significantly
- You want different sampling parameters

**Note**: Resampling will require re-running predictions (Step 4), which costs money.

---

## Step 4: Fit and Run Predictor

### Purpose

The predictor is a small LLM (GPT-mini, Gemini Flash, etc.) that classifies examples. It uses GEPA (a DSPY optimization algorithm) to generate an optimized classification prompt based on your decisions.

### Cost Warning

Running predictions costs money (~$2-3 per full trainset run). Plan your workflow to minimize re-runs:
- Batch multiple decisions before resampling
- Fix as many issues as possible before re-running predictions
- The iteration loop is: decisions → resample → fit predictor → run predictions → review → repeat

### Fitting the Predictor

Fitting uses your decisions to optimize a classification prompt:

```python
from clx.models import Label

label = Label.objects.get(project_id="docket-entry", name="Motion")

# Configure models (optional)
label.inference_model = "openai/gpt-5-mini"  # For predictions
label.teacher_model = "openai/gpt-5"         # For GEPA optimization
label.save()

# Fit the predictor (uses decisions as training examples)
label.fit_predictor()
# This will print the cost when done
```

### Running Predictions

After fitting, run predictions across the trainset:

```python
label.update_trainset_preds(num_threads=128)

# Check prediction counts
label.refresh_from_db()
print(f"Positive predictions: {label.trainset_num_positive_preds}")
print(f"Negative predictions: {label.trainset_num_negative_preds}")
```

### Viewing Predictions with Reasons

The predictor outputs both a value and a reason for each prediction:

```python
# View trainset examples with predictions
for ex in label.trainset_examples.filter(pred__isnull=False)[:10]:
    print(f"Pred: {ex.pred}")
    print(f"Text: {ex.text[:100]}...")
    print(f"Reason: {ex.reason}")
    print()
```

---

## Step 5: Train Fine-tuned Models

### Purpose

Fine-tuned BERT models are the production output. They're fast and cheap to run at scale. We train them on the predictor's outputs.

### Two Configs

- **`main`**: Full training (10 epochs) - the production model
- **`underfit`**: Light training (1 epoch) - useful for finding different failure modes

Training both configs helps identify disagreements between models.

### Training Process

Training can be done via CLI or programmatically:

```bash
# CLI: Train main model (10 epochs)
clx train docket-entry "Motion" main

# CLI: Train underfit model (1 epoch)
clx train docket-entry "Motion" underfit
```

```python
# Programmatic: Train a specific config
label = Label.objects.get(project_id="docket-entry", name="Motion")
label.train_finetune("main")
```

The training process:
1. Prepares training data from the trainset
2. Runs training remotely in the cloud
3. Runs predictions on the trainset using the trained model
4. Updates the finetune tags and saves eval results

### Update All (Recommended)

The `update_all` method runs the full pipeline, but **only steps that are out of date**:

```python
label = Label.objects.get(project_id="docket-entry", name="Motion")

# Run only what's needed based on timestamps
label.update_all()

# Force run everything regardless of timestamps
label.update_all(force=True)
```

This checks timestamps and runs:
1. **Resample trainset** - if decisions are newer than trainset
2. **Fit predictor** - if trainset is newer than predictor
3. **Run predictions** - if predictor is newer than predictions
4. **Train finetunes** - if predictions are newer than finetunes
5. **Run global corpus predictions** - only if `predict=True` and main finetune is newer than global predictions

```python
# Also run global corpus predictions (step 5)
label.update_all(predict=True)
```

### Programmatic Access

```python
from clx.models import Label

label = Label.objects.get(project_id="docket-entry", name="Motion")

# Prepare finetune data (for inspection)
train_data, eval_data, config = label.prepare_finetune("main")

# Get the trained model pipeline (runs remotely)
pipe = label.get_finetune_run_pipe("main")
predictions = pipe(["some text to classify"], batch_size=16)

# View finetune results
for ft in label.fintunes.all():
    print(f"Config: {ft.config_name}")
    print(f"Results: {ft.eval_results}")
```

### Global Corpus Predictions

After training a finetune, you can run predictions across the **entire corpus** (not just the trainset). This is a separate step because it's more expensive and not always needed during development.

Global predictions only run for the **main finetune config** (defined on the search model as `main_finetune_config`).

```python
# Run predictions across entire corpus using the main finetune config
label.predict_finetune()

# Force restart (clears cache and starts fresh)
label.predict_finetune(force=True)
```

The `predict_finetune` method:
1. **Uses the main finetune config** - defined on the search model (e.g., `DocketEntry.main_finetune_config = "main"`)
2. **Is idempotent** - caches progress and picks up where it left off if interrupted
3. **Uses a CSV cache** in the label's data directory (`data_dir/finetune_predictions_cache.csv`)
4. **Updates the global finetune tag** (`ft`) when complete
5. **Sets `predicted_at` timestamp** on the LabelFinetune object
6. **Deletes the cache** after successful completion

**Tags**:
- `trainset:ft:{config}` - Predictions on trainset only (set by `train_finetune`)
- `ft` - Predictions on entire corpus for the main config (set by `predict_finetune`)

**Timestamps** on LabelFinetune:
- `finetuned_at` - When the model was last trained
- `predicted_at` - When global corpus predictions were last run

---

## Step 6: Review and Iterate

### Finding Issues

Use search to find examples where models might be wrong:

1. **Review disagreements**: Examples where predictor and fine-tunes disagree
2. **Search by heuristic bucket**: Look in neutral bucket for edge cases
3. **Keyword search**: Find specific patterns
4. **Semantic search**: Find examples similar to a known problem

### Fast Annotations

For quick fixes without full decision reasons, use fast annotations:

```python
from clx.models import Label

label = Label.objects.get(project_id="docket-entry", name="Motion")
model = label.project.get_search_model()

# Get an example
example = model.objects.get(id=12345)

# Set annotation (no reason needed)
example.set_annotation(label, True)   # Mark as positive
example.set_annotation(label, False)  # Mark as negative
example.set_annotation(label, "flag") # Flag for exclusion from trainset
example.set_annotation(label, None)   # Clear annotation
```

Flagged examples are excluded from the trainset entirely.

### The Iteration Loop

1. **Search** for potential issues (disagreements, specific patterns, etc.)
2. **Review** examples and identify errors
3. **Fix** via decisions (for edge cases needing reasons) or fast annotations (for quick fixes)
4. **Batch fixes** - do as many as possible before re-running
5. **Resample trainset** (`label.update_trainset()`)
6. **Refit predictor** (`label.fit_predictor()`)
7. **Re-run predictions** (`label.update_trainset_preds()`)
8. **Retrain models** (CLI train commands)
9. **Repeat**

---

## Search Reference

The search system is the primary way to find and review examples.

### Basic Search

```python
from clx.models import Project

project = Project.objects.get(id="docket-entry")
model = project.get_search_model()

# Simple search - returns dict with 'data' key
results = model.objects.search(page=1, page_size=100)
for item in results["data"]:
    print(item["id"], item["text"][:80])
```

### Search Parameters

All parameters go in a `params` dict:

```python
results = model.objects.search(
    active_label_id=label.id,  # Required for most filters
    params={
        # Heuristic bucket filter
        "heuristic_bucket": "excluded" | "neutral" | "likely",

        # Trainset filter
        "trainset_split": "train" | "eval" | "both",

        # Predictor prediction filter
        "predictor_value": "true" | "false",

        # Manual annotation filter
        "annotation_value": "true" | "false" | "flag" | "any" | "none",

        # Find disagreements between models
        "review_disagreements": True,

        # Keyword search (uses query string syntax)
        "querystring": "motion, ~denied",
    },
    page=1,
    page_size=100,
)
```

### Semantic Search

Find examples similar to a query or embedding:

```python
# Search by text similarity
results = model.objects.search(
    semantic_sort="motion for summary judgment",
    page_size=50,
)

# Or use an embedding directly
embedding = [0.1, 0.2, ...]  # 96-dim vector
results = model.objects.search(semantic_sort=embedding)
```

### Search Result Format

Each result includes:

```python
{
    "id": 12345,
    "text_hash": "abc123...",
    "text": "Full text of the example",
    "tags": [1, 5, 12],  # Tag IDs
    # If in trainset:
    "split": "train" | "eval",
    "pred": True | False | None,
    "reason": "Predictor's reasoning...",
}
```

### Count Only

```python
result = model.objects.search(
    active_label_id=label.id,
    params={"heuristic_bucket": "neutral"},
    count=True,
)
print(f"Total: {result['total']}")
```

### Query String Syntax (Review)

| Operator | Meaning | Example |
|----------|---------|---------|
| `,` | AND | `motion, court` |
| `\|` | OR | `motion\|filing` |
| `~` | NOT | `~denied` |
| `^` | Starts with | `^Summary` |

---

## Key Files Reference

| Component | File | Key Lines |
|-----------|------|-----------|
| Models | `clx/app/models.py` | Full file |
| Search | `clx/app/search_utils.py` | `SearchQuerySet.search` |
| Heuristics | `clx/app/models.py` | `LabelHeuristic` class |
| Custom Heuristics | `clx/app/custom_heuristics.py` | Decorator pattern |
| Train CLI | `clx/cli/train.py` | Export/train/import |
| Views | `clx/app/views.py` | All endpoints |
| **Helpers** | `experiment/helpers.py` | Claude Code utilities |

---

## Helper Scripts for Claude Code

The `experiment/helpers.py` module provides convenient functions for the annotation workflow:

### Quick Status Check

```python
from experiment.helpers import print_label_status

print_label_status("Motion")
```

### Searching and Viewing Examples

```python
from experiment.helpers import (
    search_examples,
    print_examples,
    disagreements,
    neutral_examples,
    similar_to,
)

# Find disagreements between models
examples = disagreements("Motion")
print_examples(examples)

# Look at edge cases (neutral bucket)
examples = neutral_examples("Motion", page_size=10)
print_examples(examples)

# Find similar examples
examples = similar_to("Motion", "application for extension of time")
print_examples(examples, show_full_text=True)

# Complex search
examples = search_examples(
    "Motion",
    heuristic_bucket="neutral",
    querystring="application",
    page_size=20,
)
print_examples(examples)
```

### Creating Decisions

```python
from experiment.helpers import (
    create_decision,
    create_decision_from_id,
    view_decisions,
)

# View existing decisions
view_decisions("Motion")

# Create from text
create_decision(
    "Motion",
    text="Application for Extension of Time",
    value=True,
    reason="Applications requesting court action are functionally motions"
)

# Create from search result ID
create_decision_from_id(
    "Motion",
    example_id=12345,
    value=False,
    reason="This is a response to a motion, not a motion itself"
)
```

### Fast Annotations

```python
from experiment.helpers import annotate

annotate("Motion", example_id=12345, value=True)   # Positive
annotate("Motion", example_id=12346, value=False)  # Negative
annotate("Motion", example_id=12347, value="flag") # Exclude from trainset
```

### Creating Heuristics

```python
from experiment.helpers import create_heuristic

create_heuristic(
    "Motion",
    querystring="motion|application|request",
    is_minimal=True,
    apply=True,  # Immediately computes across corpus
)
```

---

## Scales OKN Integration (docket-entry only)

For the docket-entry project, we have predictions from Scales OKN—a similar classification project with pre-trained models for many of the same labels.

### Available Scales Labels

The following labels have Scales OKN predictions imported:

| Scales Label | Our Label |
|--------------|-----------|
| summons | Summons |
| waiver | Waiver |
| brief | Brief / Memorandum |
| arrest | Arrest |
| warrant | Warrant |
| verdict | Verdict |
| answer | Answer |
| complaint | Complaint |
| indictment | Indictment |
| information | Information |
| petition | Petition |
| notice | Notice |
| response | Reply / Response |
| minute entry | Minute Entry |
| plea agreement | Plea Agreement |
| judgment | Judgment |
| stipulation | Stipulation |
| motion | Motion |
| order | Order |

### How Scales Tags Work

- Each label has a `LabelTag` with `name="scales"`
- Positive Scales predictions (score > 0.5) are tagged
- Absence of tag means Scales predicted negative (or no prediction)

### Using Scales for Review

Scales predictions are another source of feedback when reviewing. You can compare:
- Examples where our models predict TRUE but Scales predicts FALSE
- Examples where our models predict FALSE but Scales predicts TRUE

**Important caveats:**
1. **Scales is not ground truth** - it has errors and may make different annotation decisions
2. **Scope to trainset** - we only compute our predictions on the trainset, so compare within trainset
3. **Check against decisions** - if our models disagree with Scales but are consistent with our documented decisions, that's fine

### Searching with Scales

```python
from experiment.helpers import search_examples

# Find examples where we predict TRUE but Scales predicts FALSE
# (These might be cases Scales missed, or cases we're wrong about)
label = get_label("Motion")
scales_tag = label.labeltag_set.filter(name="scales").first()

# Search for trainset examples our predictor says TRUE
examples = search_examples(
    "Motion",
    trainset_split="train",
    predictor_value="true",
)

# Filter to those without scales tag (Scales said FALSE)
# This requires checking tags manually or using raw search
```

### When to Use Scales Feedback

- **After initial model training** - to find potential blind spots
- **When reviewing disagreements** - as an additional signal
- **NOT as automatic corrections** - always review why there's a disagreement

---

## Notes

- The docket-entry project uses `DocketEntry` as the search model
- Heuristics create `LabelTag` entries attached to documents via PostgreSQL array fields
- The `apply()` step processes documents in batches of 1M for efficiency
- Predictions cost money - batch your changes before re-running
- The main fine-tune model is the production output; underfit helps find different errors
