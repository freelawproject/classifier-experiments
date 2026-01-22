"""
Helper functions for Claude Code to interact with the CLX annotation workflow.

These utilities make it easier to:
- Search and view examples with predictions
- Create decisions and annotations
- Check label status
"""

from clx import generate_hash
from clx.models import Label, LabelDecision, LabelHeuristic, Project


def get_label(label_name: str, project_id: str = "docket-entry") -> Label:
    """Get a label by name."""
    return Label.objects.get(project_id=project_id, name=label_name)


def get_project(project_id: str = "docket-entry") -> Project:
    """Get a project by ID."""
    return Project.objects.get(id=project_id)


def label_status(label_name: str, project_id: str = "docket-entry") -> dict:
    """Get comprehensive status of a label including warnings and decisions."""
    label = get_label(label_name, project_id)

    # Get all decisions with full info
    decisions = [
        {
            "id": d.id,
            "value": d.value,
            "reason": d.reason,
            "text": d.text,
            "created_at": d.created_at,
            "updated_at": d.updated_at,
        }
        for d in label.decisions.all().order_by("-updated_at")
    ]

    # Get finetunes with timestamps
    finetunes = [
        {
            "config": ft.config_name,
            "results": ft.eval_results,
            "created_at": ft.created_at,
            "updated_at": ft.updated_at,
            "finetuned_at": ft.finetuned_at,
            "predicted_at": ft.predicted_at,
            "is_main": ft.config_name
            == label.project.get_search_model().main_finetune_config,
        }
        for ft in label.fintunes.all()
    ]

    # Generate warnings based on timestamp comparisons
    warnings = []

    # Get latest decision timestamp
    latest_decision_at = None
    if decisions:
        latest_decision_at = max(d["updated_at"] for d in decisions)

    # Warning: decisions newer than trainset
    if latest_decision_at and label.trainset_updated_at:
        if latest_decision_at > label.trainset_updated_at:
            warnings.append(
                "Decisions updated since last trainset sampling - consider resampling"
            )
    elif latest_decision_at and not label.trainset_updated_at:
        warnings.append("Trainset has never been sampled")

    # Warning: trainset newer than predictor
    if label.trainset_updated_at and label.predictor_updated_at:
        if label.trainset_updated_at > label.predictor_updated_at:
            warnings.append(
                "Trainset updated since last predictor fit - consider refitting"
            )
    elif label.trainset_updated_at and not label.predictor_updated_at:
        warnings.append("Predictor has never been fit")

    # Warning: predictor newer than predictions
    if label.predictor_updated_at and label.trainset_predictions_updated_at:
        if label.predictor_updated_at > label.trainset_predictions_updated_at:
            warnings.append(
                "Predictor updated since last prediction run - consider rerunning predictions"
            )
    elif (
        label.predictor_updated_at
        and not label.trainset_predictions_updated_at
    ):
        warnings.append("Predictions have never been run")

    # Warning: predictions newer than finetunes
    if label.trainset_predictions_updated_at and finetunes:
        for ft in finetunes:
            finetuned_at = ft.get("finetuned_at")
            if (
                not finetuned_at
                or label.trainset_predictions_updated_at > finetuned_at
            ):
                warnings.append(
                    f"Predictions updated since '{ft['config']}' finetune - consider retraining"
                )

    # Warning: finetunes newer than global predictions
    for ft in finetunes:
        finetuned_at = ft.get("finetuned_at")
        predicted_at = ft.get("predicted_at")
        if finetuned_at and (not predicted_at or finetuned_at > predicted_at):
            warnings.append(
                f"'{ft['config']}' finetune updated since global predictions - consider running predict_finetune"
            )

    return {
        "name": label.name,
        "id": label.id,
        "warnings": warnings,
        "heuristic_buckets": {
            "excluded": label.num_excluded,
            "neutral": label.num_neutral,
            "likely": label.num_likely,
        },
        "heuristics": [
            {
                "id": h.id,
                "query": h.querystring or f"[custom: {h.custom}]",
                "is_minimal": h.is_minimal,
                "is_likely": h.is_likely,
                "num_examples": h.num_examples,
                "applied_at": h.applied_at,
            }
            for h in label.heuristics.all()
        ],
        "decisions": decisions,
        "trainset": {
            "train": label.trainset_examples.filter(split="train").count(),
            "eval": label.trainset_examples.filter(split="eval").count(),
            "updated_at": label.trainset_updated_at,
        },
        "predictor": {
            "positive_preds": label.trainset_num_positive_preds,
            "negative_preds": label.trainset_num_negative_preds,
            "updated_at": label.trainset_predictions_updated_at,
            "fitted_at": label.predictor_updated_at,
            "inference_model": label.inference_model,
            "teacher_model": label.teacher_model,
        },
        "finetunes": finetunes,
    }


def print_label_status(label_name: str, project_id: str = "docket-entry"):
    """Print a formatted label status report."""
    status = label_status(label_name, project_id)

    print(f"=== Label: {status['name']} (ID: {status['id']}) ===\n")

    # Show warnings prominently at the top
    if status["warnings"]:
        print("WARNINGS:")
        for warning in status["warnings"]:
            print(f"  ! {warning}")
        print()

    print("Heuristic Buckets:")
    for bucket, count in status["heuristic_buckets"].items():
        print(f"  {bucket}: {count:,}")

    print(f"\nHeuristics ({len(status['heuristics'])}):")
    for h in status["heuristics"]:
        flags = []
        if h["is_minimal"]:
            flags.append("minimal")
        if h["is_likely"]:
            flags.append("likely")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        print(f"  {h['query']}{flag_str} → {h['num_examples']:,} matches")

    print(f"\nDecisions ({len(status['decisions'])}):")
    for d in status["decisions"]:
        value_str = "TRUE" if d["value"] else "FALSE"
        text_preview = (
            d["text"][:80] + "..."
            if d["text"] and len(d["text"]) > 80
            else d["text"]
        )
        print(f"  [{value_str}] {text_preview}")
        print(f"    Reason: {d['reason']}")

    print("\nTrainset:")
    print(f"  Train: {status['trainset']['train']:,}")
    print(f"  Eval: {status['trainset']['eval']:,}")
    print(f"  Updated: {status['trainset']['updated_at']}")

    print("\nPredictor:")
    print(f"  Positive preds: {status['predictor']['positive_preds']:,}")
    print(f"  Negative preds: {status['predictor']['negative_preds']:,}")
    print(f"  Fitted: {status['predictor']['fitted_at']}")
    print(f"  Predictions updated: {status['predictor']['updated_at']}")

    if status["finetunes"]:
        print("\nFinetunes:")
        for ft in status["finetunes"]:
            print(f"  {ft['config']}:")
            print(f"    Results: {ft['results']}")
            print(f"    Finetuned at: {ft['finetuned_at']}")
            print(f"    Global predictions at: {ft['predicted_at']}")


def search_examples(
    label_name: str,
    project_id: str = "docket-entry",
    heuristic_bucket: str | None = None,
    trainset_split: str | None = None,
    predictor_value: str | None = None,
    annotation_value: str | None = None,
    review_disagreements: bool = False,
    querystring: str | None = None,
    semantic_sort: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> list[dict]:
    """
    Search for examples with full context.

    Returns examples with:
    - text and metadata
    - predictor prediction and reason (if in trainset)
    - finetune predictions
    - annotation status
    """
    label = get_label(label_name, project_id)
    model = label.project.get_search_model()

    params = {}
    if heuristic_bucket:
        params["heuristic_bucket"] = heuristic_bucket
    if trainset_split:
        params["trainset_split"] = trainset_split
    if predictor_value:
        params["predictor_value"] = predictor_value
    if annotation_value:
        params["annotation_value"] = annotation_value
    if review_disagreements:
        params["review_disagreements"] = True
    if querystring:
        params["querystring"] = querystring

    search_kwargs = {
        "active_label_id": label.id,
        "params": params,
        "page": page,
        "page_size": page_size,
    }
    if semantic_sort:
        search_kwargs["semantic_sort"] = semantic_sort

    results = model.objects.search(**search_kwargs)

    # Enrich with tag information
    enriched = []
    for item in results.get("data", []):
        tags = item.get("tags", [])

        # Check annotation status
        anno_status = None
        if label.anno_true_tag.id in tags:
            anno_status = "true"
        elif label.anno_false_tag.id in tags:
            anno_status = "false"
        elif label.anno_flag_tag.id in tags:
            anno_status = "flag"

        # Check finetune predictions
        finetune_preds = {}
        for ft in label.fintunes.all():
            ft_tag = label.get_trainset_finetune_tag(ft.config_name)
            finetune_preds[ft.config_name] = ft_tag.id in tags

        # Check predictor prediction
        predictor_pred = label.trainset_pred_tag.id in tags

        enriched.append(
            {
                "id": item["id"],
                "text_hash": item["text_hash"],
                "text": item["text"],
                "annotation": anno_status,
                "predictor_pred": predictor_pred
                if item.get("split")
                else None,
                "predictor_reason": item.get("reason"),
                "finetune_preds": finetune_preds if finetune_preds else None,
                "trainset_split": item.get("split"),
            }
        )

    return enriched


def print_examples(
    examples: list[dict],
    show_full_text: bool = False,
    max_text_len: int = 120,
):
    """Print examples in a readable format."""
    for i, ex in enumerate(examples, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}] ID: {ex['id']}")

        text = ex["text"]
        if not show_full_text and len(text) > max_text_len:
            text = text[:max_text_len] + "..."
        print(f"Text: {text}")

        # Predictions
        preds = []
        if ex.get("predictor_pred") is not None:
            preds.append(f"predictor={ex['predictor_pred']}")
        if ex.get("finetune_preds"):
            for config, pred in ex["finetune_preds"].items():
                preds.append(f"{config}={pred}")
        if preds:
            print(f"Predictions: {', '.join(preds)}")

        if ex.get("predictor_reason"):
            print(f"Reason: {ex['predictor_reason']}")

        if ex.get("annotation"):
            print(f"Annotation: {ex['annotation']}")

        if ex.get("trainset_split"):
            print(f"Split: {ex['trainset_split']}")


def view_decisions(label_name: str, project_id: str = "docket-entry"):
    """View all decisions for a label."""
    label = get_label(label_name, project_id)

    print(f"=== Decisions for {label_name} ===\n")

    for d in label.decisions.all().order_by("-updated_at"):
        value_str = "TRUE" if d.value else "FALSE"
        print(f"[{value_str}] {d.text[:100]}...")
        print(f"  Reason: {d.reason}")
        print()


def create_decision(
    label_name: str,
    example_id: int,
    value: bool,
    reason: str,
    project_id: str = "docket-entry",
) -> LabelDecision:
    """
    Create a decision from an example ID.

    IMPORTANT: Always use example IDs from search results, not raw text.
    This ensures the text_hash matches documents in the search table.
    """
    label = get_label(label_name, project_id)
    model = label.project.get_search_model()
    example = model.objects.get(id=example_id)

    text = example.text
    text_hash = generate_hash(text)

    decision, created = LabelDecision.objects.update_or_create(
        label=label,
        text_hash=text_hash,
        defaults={
            "text": text,
            "value": value,
            "reason": reason,
        },
    )

    action = "Created" if created else "Updated"
    print(f"{action} decision: {value} - {reason}")
    return decision


def annotate(
    label_name: str,
    example_id: int,
    value: bool | str | None,
    project_id: str = "docket-entry",
):
    """
    Set a fast annotation on an example.

    value can be:
    - True: positive
    - False: negative
    - "flag": exclude from trainset
    - None: clear annotation
    """
    label = get_label(label_name, project_id)
    model = label.project.get_search_model()
    example = model.objects.get(id=example_id)
    example.set_annotation(label, value)
    print(f"Set annotation {value} on example {example_id}")


def create_heuristic(
    label_name: str,
    querystring: str,
    is_minimal: bool = False,
    is_likely: bool = False,
    apply: bool = False,
    project_id: str = "docket-entry",
) -> LabelHeuristic:
    """Create and optionally apply a heuristic. Defaults to NOT applying."""
    label = get_label(label_name, project_id)

    heuristic = LabelHeuristic.objects.create(
        label=label,
        querystring=querystring,
        is_minimal=is_minimal,
        is_likely=is_likely,
    )

    if apply:
        print(f"Applying heuristic: {querystring}")
        heuristic.apply()
        label.refresh_from_db()
        print(f"Matches: {heuristic.num_examples:,}")
        print(
            f"Buckets - Excluded: {label.num_excluded:,}, Neutral: {label.num_neutral:,}, Likely: {label.num_likely:,}"
        )

    return heuristic


# Quick aliases for common operations
def disagreements(label_name: str, page_size: int = 20, **kwargs):
    """Find examples where models disagree."""
    return search_examples(
        label_name,
        review_disagreements=True,
        page_size=page_size,
        **kwargs,
    )


def neutral_examples(label_name: str, page_size: int = 20, **kwargs):
    """Get examples from the neutral bucket (edge cases)."""
    return search_examples(
        label_name,
        heuristic_bucket="neutral",
        page_size=page_size,
        **kwargs,
    )


def likely_examples(label_name: str, page_size: int = 20, **kwargs):
    """Get examples from the likely bucket (probable positives)."""
    return search_examples(
        label_name,
        heuristic_bucket="likely",
        page_size=page_size,
        **kwargs,
    )


def excluded_examples(label_name: str, page_size: int = 20, **kwargs):
    """Get examples from the excluded bucket (probable negatives)."""
    return search_examples(
        label_name,
        heuristic_bucket="excluded",
        page_size=page_size,
        **kwargs,
    )


def similar_to(label_name: str, text: str, page_size: int = 20, **kwargs):
    """Find examples semantically similar to the given text."""
    return search_examples(
        label_name,
        semantic_sort=text,
        page_size=page_size,
        **kwargs,
    )


def set_instructions(
    label_name: str, instructions: str, project_id: str = "docket-entry"
):
    """Set the instructions for a label."""
    label = get_label(label_name, project_id)
    label.instructions = instructions
    label.save()
    print(f"Instructions saved for {label_name}")


def view_heuristics(label_name: str, project_id: str = "docket-entry"):
    """View all heuristics for a label."""
    label = get_label(label_name, project_id)
    print(f"=== Heuristics for {label_name} ===\n")
    for h in label.heuristics.all():
        flags = []
        if h.is_minimal:
            flags.append("minimal")
        if h.is_likely:
            flags.append("likely")
        flag_str = f"[{', '.join(flags)}]" if flags else "[none]"
        print(
            f"  {flag_str} {h.querystring or f'[custom: {h.custom}]'} -> {h.num_examples:,} matches"
        )


def count_matches(querystring: str, project_id: str = "docket-entry") -> int:
    """Count how many examples match a querystring without creating a heuristic."""
    project = get_project(project_id)
    model = project.get_search_model()
    result = model.objects.search(
        params={"querystring": querystring},
        count=True,
    )
    return result.get("total", 0)


def search_by_query(
    querystring: str,
    project_id: str = "docket-entry",
    page: int = 1,
    page_size: int = 20,
) -> list[dict]:
    """Search examples by querystring (without needing a label)."""
    project = get_project(project_id)
    model = project.get_search_model()
    results = model.objects.search(
        params={"querystring": querystring},
        page=page,
        page_size=page_size,
    )
    return [
        {
            "id": item["id"],
            "text": item["text"],
            "text_hash": item["text_hash"],
        }
        for item in results.get("data", [])
    ]


def print_search_results(results: list[dict], max_text_len: int = 120):
    """Print search results in a compact format."""
    for i, r in enumerate(results, 1):
        text = r["text"]
        if len(text) > max_text_len:
            text = text[:max_text_len] + "..."
        print(f"[{i}] (id={r['id']}) {text}")


def setup_label(
    label_name: str,
    minimal_query: str,
    likely_query: str,
    instructions: str,
    project_id: str = "docket-entry",
    apply: bool = False,
):
    """Set up a label with minimal heuristic, likely heuristic, and instructions. Defaults to NOT applying."""
    label = get_label(label_name, project_id)

    # Set instructions
    label.instructions = instructions
    label.save()
    print(f"Instructions saved for {label_name}")

    # Create minimal heuristic
    minimal_h = LabelHeuristic.objects.create(
        label=label,
        querystring=minimal_query,
        is_minimal=True,
    )
    print(f"Created minimal heuristic: {minimal_query}")

    # Create likely heuristic
    likely_h = LabelHeuristic.objects.create(
        label=label,
        querystring=likely_query,
        is_likely=True,
    )
    print(f"Created likely heuristic: {likely_query}")

    if apply:
        print("Applying heuristics...")
        minimal_h.apply()
        likely_h.apply()
        label.refresh_from_db()
        print(f"Minimal matches: {minimal_h.num_examples:,}")
        print(f"Likely matches: {likely_h.num_examples:,}")
        print(
            f"Buckets - Excluded: {label.num_excluded:,}, Neutral: {label.num_neutral:,}, Likely: {label.num_likely:,}"
        )

    return label
