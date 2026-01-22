# Review First Cut

Audit CLX labels to verify they have proper first-cut setup and identify potential issues for human review.

## Usage
```
/review-first-cut [label_name or pattern]
```

If no label name is provided, review all labels. If a label name/pattern is provided, review matching labels.

## What This Skill Does

1. Pulls the current status of each label (heuristics, decisions, instructions, bucket counts)
2. Checks for completeness (has minimal heuristic, likely heuristic, instructions, decisions)
3. **Validates heuristic syntax** - especially checking for the AND/OR precedence mistake
4. Identifies potential issues or edge cases that may need human review
5. Returns a structured report with findings

## Execution

### Step 1: Pull label status

For a single label:
```python
uv run python -c "
from cc.helpers import print_label_status
print_label_status('$ARGUMENTS')
"
```

For all labels or a pattern:
```python
uv run python -c "
from clx.models import Label

project_id = 'docket-entry'
labels = Label.objects.filter(project_id=project_id).order_by('name')

# Optional: filter by pattern
# labels = labels.filter(name__startswith='Motion')

for label in labels:
    print(f'========== {label.name} ==========')
    print()

    # Instructions
    print('INSTRUCTIONS:')
    if label.instructions:
        print(label.instructions[:500] + '...' if len(label.instructions) > 500 else label.instructions)
    else:
        print('  (none)')
    print()

    # Heuristics
    print('HEURISTICS:')
    for h in label.heuristics.all():
        flags = []
        if h.is_minimal: flags.append('minimal')
        if h.is_likely: flags.append('likely')
        flag_str = f'[{\", \".join(flags)}]' if flags else '[other]'
        query = h.querystring or f'[custom: {h.custom}]'
        print(f'  {flag_str:12} {query} -> {h.num_examples:,}')
    print()

    # Decisions
    print(f'DECISIONS ({label.decisions.count()}):')
    for d in label.decisions.all().order_by('-value', 'text')[:10]:
        val = 'TRUE' if d.value else 'FALSE'
        text = d.text[:100] + '...' if len(d.text) > 100 else d.text
        print(f'  [{val}] {text}')
        print(f'         Reason: {d.reason}')
    if label.decisions.count() > 10:
        print(f'  ... and {label.decisions.count() - 10} more')
    print()

    # Bucket counts
    print(f'BUCKETS: excluded={label.num_excluded:,}, neutral={label.num_neutral:,}, likely={label.num_likely:,}')
    print()
    print()
"
```

### Step 2: Automated syntax checks

Run this to catch common heuristic mistakes:

```python
uv run python -c "
from clx.models import Label, LabelHeuristic

project_id = 'docket-entry'
labels = Label.objects.filter(project_id=project_id)

# Optional: filter by pattern
# labels = labels.filter(name__startswith='Motion')

print('=== HEURISTIC SYNTAX CHECK ===')
print()

# Check 1: Likely heuristics with commas (CRITICAL - usually broken)
print('1. LIKELY HEURISTICS WITH COMMAS (usually broken):')
print('-' * 50)
likely_with_commas = LabelHeuristic.objects.filter(
    label__in=labels,
    is_likely=True,
    querystring__contains=','
)
if likely_with_commas.exists():
    for h in likely_with_commas:
        print(f'  {h.label.name}:')
        print(f'    {h.querystring}')
        print(f'    ^ Commas in likely heuristics create AND requirements')
        print(f'      across ALL OR alternatives. This is almost always wrong.')
        print()
else:
    print('  None found - good!')
print()

# Check 2: Minimal heuristics that might be too narrow
print('2. MINIMAL HEURISTICS WITH 3+ COMMA-SEPARATED TERMS:')
print('-' * 50)
for h in LabelHeuristic.objects.filter(label__in=labels, is_minimal=True):
    # Count comma-separated required groups
    comma_count = h.querystring.count(',') if h.querystring else 0
    if comma_count >= 2:  # 3+ required terms
        print(f'  {h.label.name}:')
        print(f'    {h.querystring}')
        print(f'    ^ {comma_count + 1} required terms - may be too narrow')
        print()
print()

# Check 3: Minimal heuristics with ^ (starts with) - usually too restrictive
print('3. MINIMAL HEURISTICS WITH ^ (starts with):')
print('-' * 50)
for h in LabelHeuristic.objects.filter(label__in=labels, is_minimal=True):
    if h.querystring and '^' in h.querystring:
        # Check if ALL alternatives have ^
        parts = h.querystring.replace(',', '|').split('|')
        if all(p.strip().startswith('^') for p in parts if p.strip()):
            print(f'  {h.label.name}:')
            print(f'    {h.querystring}')
            print(f'    ^ Minimal with only ^patterns may miss valid entries')
            print()
print()

# Check 4: Likely heuristics without ^ (might be too broad for likely)
print('4. LIKELY HEURISTICS WITHOUT ^ (might be broad):')
print('-' * 50)
for h in LabelHeuristic.objects.filter(label__in=labels, is_likely=True):
    if h.querystring and '^' not in h.querystring:
        print(f'  {h.label.name}:')
        print(f'    {h.querystring}')
        print(f'    ^ Likely without ^ prefix - verify this is intentional')
        print()
print()

print('=== COMPLETENESS CHECK ===')
print()
incomplete = []
for label in labels:
    heuristics = label.heuristics.all()
    minimal = [h for h in heuristics if h.is_minimal]
    likely = [h for h in heuristics if h.is_likely]
    decisions = label.decisions.count()
    has_instructions = bool(label.instructions and label.instructions.strip())

    missing = []
    if not minimal:
        missing.append('minimal heuristic')
    if not likely:
        missing.append('likely heuristic')
    if not has_instructions:
        missing.append('instructions')
    if decisions < 4:
        missing.append(f'decisions ({decisions}/4)')

    if missing:
        incomplete.append((label.name, missing))

if incomplete:
    for name, missing in incomplete:
        print(f'  {name}: missing {', '.join(missing)}')
else:
    print('  All labels complete!')
"
```

### Step 3: Analyze and report

After pulling the data, analyze each label for:

#### Completeness Checks
- [ ] Has at least 1 minimal heuristic
- [ ] Has at least 1 likely heuristic
- [ ] Has instructions (not empty)
- [ ] Has at least 4 decisions (2+ positive, 2+ negative)

#### CRITICAL: Heuristic Syntax Issues

**Likely heuristics with commas (ALMOST ALWAYS WRONG):**

The most common mistake is putting commas in likely heuristics. Remember:
- Commas (AND) are ALWAYS the outer scope
- Pipes (OR) are ALWAYS the inner scope

So `^motion to dismiss|^motion for summary judgment,motion` parses as:
- `(^motion to dismiss OR ^motion for summary judgment) AND motion`
- The `,motion` applies to EVERYTHING, not just the last alternative!

**Fix:** Remove all commas from likely heuristics. Use only OR alternatives:
```
^motion to dismiss|^motion for summary judgment
```

**Minimal heuristics that are too narrow:**
- 3+ comma-separated terms usually means too narrow
- `^` (starts with) on ALL alternatives is usually too restrictive for minimal
- Exact phrases like `motion,summary judgment` may miss variants

**Good minimal patterns:**
- `motion,<keyword>` - simple and broad
- `<keyword>` alone for distinctive terms
- `keyword1|keyword2` for synonyms

#### Other Issues to Flag

**Instruction Issues:**
- Instructions that contradict decisions
- Instructions that are too vague
- Instructions that incorrectly state labels are mutually exclusive (labels CAN and WILL overlap)
- Missing mention of combined/alternative filings

**Decision Issues:**
- Decisions that seem inconsistent with instructions
- Missing obvious positive or negative examples
- Missing combined motion examples (e.g., "Motion to Dismiss or for Summary Judgment")
- Decisions where the reasoning seems wrong

**Label Boundary Issues:**
- Overlap with other labels that may need clarification
- Edge cases that could go either way

## Report Format

Structure your report as:

```
## Summary
- X labels reviewed
- X labels complete
- X labels with syntax issues
- X labels with other issues

## Critical: Heuristic Syntax Issues

### [Label Name]
**Issue:** Likely heuristic has commas
**Current:** `^pattern1|^pattern2,extra_term`
**Problem:** The `,extra_term` applies to ALL alternatives, not just the last one
**Fix:** `^pattern1|^pattern2` (remove the comma and everything after)

## Completeness Issues

### [Label Name]
**Missing:** minimal heuristic, instructions

## Other Issues

### [Label Name]
**Issue Type:** [Instruction / Decision / Boundary]
**Description:** [What the issue is]
**Recommendation:** [Your suggested fix, if any]
```

## Key Principles to Remember

### What's a REAL problem:

1. **Likely heuristics with commas** - Almost always broken. Commas create AND requirements across ALL OR alternatives.

2. **Minimal heuristics that are too narrow** - 3+ required terms, or only `^` patterns. The minimal should catch lots of things.

3. **Instructions contradicting decisions** - If instructions say X but decisions show Y, that needs fixing.

4. **Missing completeness** - No heuristics, no instructions, fewer than 4 decisions.

### What's NOT a problem:

1. **Labels that overlap** - This is expected and fine. A "Minute Order" is both a Minute Entry AND an Order. Both labels apply.

2. **Broad minimal heuristics** - A minimal that catches 100K entries out of 7M is still very narrow (1.4%). Broad is good.

3. **Likely that doesn't catch everything** - Likely just needs to flag SOME obvious positives. It's not meant to be comprehensive.

4. **Different bucket sizes** - Having lots of "neutral" entries is fine. That's what the neutral bucket is for.

## Example: Fixing Broken Heuristics

If you find a likely heuristic with commas:

```python
uv run python -c "
from clx.models import LabelHeuristic
from cc.helpers import get_label

label = get_label('Motion for Summary Judgment')

# Delete the broken likely heuristic
old = LabelHeuristic.objects.filter(label=label, is_likely=True)
print(f'Deleting: {[h.querystring for h in old]}')
old.delete()

# Create fixed heuristic - NO COMMAS
LabelHeuristic.objects.create(
    label=label,
    querystring='^motion for summary judgment|^motion for partial summary judgment|^cross motion for summary judgment',
    is_likely=True
)
print('Created fixed likely heuristic')
"
```

If you find a minimal heuristic that's too narrow:

```python
uv run python -c "
from clx.models import LabelHeuristic
from cc.helpers import get_label, count_matches

label = get_label('Motion for Bench Trial')

# Check current vs proposed
print(f'Current: motion,bench trial -> {count_matches(\"motion,bench trial\"):,}')
print(f'Broader: motion,bench -> {count_matches(\"motion,bench\"):,}')

# If broader is better, fix it
old = LabelHeuristic.objects.filter(label=label, is_minimal=True)
print(f'Deleting: {[h.querystring for h in old]}')
old.delete()

LabelHeuristic.objects.create(
    label=label,
    querystring='motion,bench',  # Simpler, broader
    is_minimal=True
)
print('Created fixed minimal heuristic')
"
```

## Common Patterns to Watch For

### Broken likely patterns (have commas):
```
# BROKEN - comma makes ALL alternatives require the extra term
motion to compel|motion,compel discovery
^motion for leave|motion,leave to
motion,relief from judgment|motion,relief from order|motion,rule 60
```

### Fixed likely patterns (no commas):
```
# FIXED - pure OR alternatives
^motion to compel
^motion for leave|^motion seeking leave
^motion for relief from judgment|^motion for relief from order|^rule 60 motion
```

### Overly narrow minimal patterns:
```
# TOO NARROW - exact phrase required
motion,bench trial
motion,summary judgment
motion,correct sentence

# BETTER - simpler term matching
motion,bench
motion,summar
motion,sentence
```

### Good minimal patterns:
```
motion,venue
motion,compel
motion,amend
motion,relief|rule 60
substantive consolidation
certificate of appealability
```
