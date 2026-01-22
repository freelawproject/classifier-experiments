# First Cut Label Setup

Create a "first cut" for a CLX label. This sets up minimal heuristics, likely heuristics, instructions, and initial decisions.

## Usage
```
/first-cut <label_name>
```

## What "First Cut" Means

A first cut gets a label to a usable initial state with:
1. **Minimal heuristic**: ONE broad keyword query for potential positives (HIGH RECALL - err on side of inclusion)
2. **Likely heuristic**: ONE narrower query for some obvious positives (does NOT need to be perfect or complete)
3. **Instructions**: Clear guidelines on what the label represents
4. **Decisions**: 4-6 examples (at least 2 positive, 2 negative) with reasons

Do NOT run expensive operations (sampling, fitting, predictions, finetunes).

## CRITICAL Rules

1. **Always import from `clx.models`** - Never import from `clx.app.models`. The `clx.models` module initializes Django properly.

2. **Always check existing status first** - The label may already be complete or partially set up. Don't create duplicate heuristics.

3. **Only ONE minimal heuristic, ONE likely heuristic** - Multiple heuristics of the same type are treated as disjunctions (OR). This is rarely needed since the query syntax already supports `|` for OR. Only create multiple if you truly cannot express the logic in one query string.

4. **Never modify heuristics in place** - Heuristics have associated tags and status data (num_examples, applied_at) that won't update if you just change the querystring. To change a heuristic, DELETE it and CREATE a new one.

5. **Don't apply heuristics during first cut** - Creating heuristics is fast; applying them is slow. A cleanup script will apply all heuristics after first cuts are done. Use `apply=False` (now the default).

6. **Use `cc.helpers` functions** - All helpers handle imports correctly. Don't import models directly.

7. **Labels are NOT mutually exclusive** - Documents can and will have multiple labels. A "Motion to Dismiss or in the Alternative for Summary Judgment" is BOTH a Motion to Dismiss AND a Motion for Summary Judgment. Instructions should reflect this.

8. **Avoid redundant terms** - Heuristics use substring matching, so `memo` already matches `memorandum`. Don't include both.

9. **Use example IDs for decisions** - ALWAYS use the `id` from search results when creating decisions. Never paste raw text. This ensures text hashes match the document table.

## Query String Syntax

- `|` = OR (e.g., `warrant|summons`)
- `,` = AND (e.g., `arrest,warrant`)
- `~` = NOT (e.g., `order,~minute`)
- `^` = starts with (e.g., `^order`)

**Important:** These are substring matches. `memo` will match "memo", "memorandum", "memo endorsed", etc.

### CRITICAL: Operator Precedence (READ THIS CAREFULLY)

**Commas (AND) are ALWAYS the outer scope. Pipes (OR) are ALWAYS the inner scope.**

Think of it this way: The query is split by commas into REQUIRED GROUPS. Within each group, pipes create alternatives.

```
query = "group1,group2,group3"
       = (group1) AND (group2) AND (group3)

group1 = "a|b|c"
       = a OR b OR c
```

**Examples:**

| Query | Meaning |
|-------|---------|
| `motion,bench` | contains "motion" AND contains "bench" |
| `motion,bench trial` | contains "motion" AND contains "bench trial" (exact phrase) |
| `motion,bench,trial` | contains "motion" AND contains "bench" AND contains "trial" (all 3 anywhere) |
| `^motion for bench\|^motion for jury` | starts with "motion for bench" OR starts with "motion for jury" |
| `motion,bench\|jury` | contains "motion" AND (contains "bench" OR contains "jury") |
| `a\|b,c\|d` | (contains "a" OR "b") AND (contains "c" OR "d") - BOTH groups required |

**COMMON MISTAKE - Commas in Likely Heuristics:**

```
# WRONG - This requires ALL alternatives to match!
^motion to dismiss|^motion for summary judgment,motion

# This parses as:
# (starts with "motion to dismiss" OR starts with "motion for summary judgment")
# AND (contains "motion")
#
# The ",motion" applies to EVERYTHING before it!
```

```
# CORRECT - No commas, just OR alternatives
^motion to dismiss|^motion for summary judgment

# This parses as:
# starts with "motion to dismiss" OR starts with "motion for summary judgment"
```

**Rule of thumb for likely heuristics: NEVER use commas.** Likely heuristics should be pure OR alternatives of specific patterns.

### Phrase Matching vs Term Matching

| Query | What it matches |
|-------|-----------------|
| `motion,bench trial` | "motion" anywhere AND the exact phrase "bench trial" anywhere |
| `motion,bench,trial` | "motion" anywhere AND "bench" anywhere AND "trial" anywhere |

Use **phrase matching** only for legal terms of art where the phrase itself is meaningful:
- `matter of law` (legal standard)
- `certificate of appealability` (specific document type)
- `jury trial` / `jury demand` (specific legal concepts)
- `new trial` (specific remedy)
- `rule 60` (specific rule reference)

Use **term matching** for most other cases - it's broader and catches more variants.

## Three-Bucket Logic

- **Excluded**: Does not match minimal heuristic (probable negatives)
- **Neutral**: Matches minimal but not likely (edge cases for later review)
- **Likely**: Matches both minimal and likely (obvious positives)

## Heuristic Philosophy

### Minimal Heuristic: BE TRULY MINIMAL

The minimal heuristic's job is to "boil the ocean" - to reduce 7 million entries down to a manageable subset that MIGHT contain positives.

**CRITICAL:** Never worry about a minimal being "too broad." A minimal that catches 100K entries out of 7M is already very narrow (1.4%). The only failure mode is being TOO NARROW and missing things we want.

**Good minimal patterns:**
- `motion,<keyword>` - e.g., `motion,venue`, `motion,compel`, `motion,amend`
- `<keyword>` alone for distinctive terms - e.g., `habeas`, `limine`, `notwithstanding`
- `keyword1|keyword2` for synonyms - e.g., `judgment|judgement`, `settlement|compromise`

**Bad minimal patterns (too narrow):**
- `motion,bench trial` - the phrase "bench trial" is too specific, use `motion,bench`
- `motion,correct,record` - three required terms is usually too narrow
- `^motion to compel` - the `^` makes it too restrictive for a minimal

**Test your minimal:** If it catches fewer than 1,000 entries, it's probably too narrow. Consider broadening.

### Likely Heuristic: CAPTURE OBVIOUS CASES (NO COMMAS!)

The likely heuristic identifies obvious/salient positive examples. It does NOT need to:
- Be perfect or precise
- Catch every true positive
- Have high precision

It just needs to flag some obvious cases.

**CRITICAL: Likely heuristics should NEVER contain commas.** Commas create AND requirements that apply across ALL alternatives, which breaks the pattern matching.

**Good likely patterns:**
```
^motion for summary judgment|^motion for partial summary judgment|^cross motion for summary judgment
^motion to compel
^motion in limine|^motions in limine
^motion to dismiss|^motion for dismissal
```

**Bad likely patterns (broken by commas):**
```
# WRONG - comma breaks the OR logic
^motion to compel|motion,compel discovery

# WRONG - comma makes ALL alternatives require "motion"
motion,^summary judgment|^partial summary judgment
```

## Execution Steps

### Step 1: Check existing status (ALWAYS DO THIS FIRST)

```python
uv run python -c "
from cc.helpers import print_label_status

print_label_status('$ARGUMENTS')
"
```

Review the output:
- If heuristics already exist, DO NOT create new ones (unless they're broken/wrong)
- If decisions exist, you may add more but don't duplicate
- If instructions exist, only update if they're empty or inadequate
- If the label looks complete, report that and stop

### Step 2: Explore patterns and find example IDs

Search results include `id` fields - note these for creating decisions:

```python
uv run python -c "
from cc.helpers import count_matches, search_by_query, print_search_results

LABEL = '$ARGUMENTS'

# Try obvious patterns - start broad
patterns = [
    LABEL.lower().split()[-1],  # Just the key term
    f'motion,{LABEL.lower().split()[-1]}',  # motion + key term
    f'^{LABEL.lower()}',  # starts with full name
]

print(f'=== Pattern counts for {LABEL} ===')
for p in patterns:
    print(f'{p}: {count_matches(p):,}')

print(f'\\n=== Examples starting with {LABEL.lower()} ===')
print('(Note the IDs for creating decisions)')
results = search_by_query(f'^{LABEL.lower()}', page_size=12)
print_search_results(results)

print(f'\\n=== Examples containing key term (edge cases) ===')
results2 = search_by_query(f'{LABEL.lower().split()[-1]},~^{LABEL.lower()}', page_size=12)
print_search_results(results2)
"
```

### Step 3: Setup label with heuristics, instructions, and decisions

Use `setup_label` for heuristics/instructions, then `create_decision` with **example IDs** (not text):

```python
uv run python -c "
from cc.helpers import setup_label, create_decision

# Setup label with heuristics and instructions
setup_label(
    '<LABEL_NAME>',
    minimal_query='<broad_pattern>',  # Usually: motion,<keyword> or just <keyword>
    likely_query='<precise_pattern>',  # NO COMMAS - just ^pattern1|^pattern2|^pattern3
    instructions='''<Clear description of what this label represents.

Include:
- <what to include>

Exclude:
- <what to exclude>

Note: This label may overlap with other labels (e.g., X and Y). A document can have multiple labels.'''
)

print()
print('Adding decisions...')

# Use example IDs from search results - NEVER paste raw text
create_decision('<LABEL_NAME>', <example_id>, True, '<reason>')
create_decision('<LABEL_NAME>', <example_id>, True, '<reason>')
create_decision('<LABEL_NAME>', <example_id>, False, '<reason>')
create_decision('<LABEL_NAME>', <example_id>, False, '<reason>')

print('Done!')
"
```

### If fixing a broken heuristic:

**NEVER modify a heuristic in place. DELETE and CREATE new.**

```python
uv run python -c "
from clx.models import LabelHeuristic
from cc.helpers import get_label, create_heuristic

label = get_label('<LABEL_NAME>')

# Delete the broken heuristic
old = LabelHeuristic.objects.filter(label=label, is_likely=True)  # or is_minimal=True
print(f'Deleting: {[h.querystring for h in old]}')
old.delete()

# Create the fixed heuristic
create_heuristic('<LABEL_NAME>', '<fixed_query>', is_likely=True)  # or is_minimal=True
"
```

### If only adding decisions (heuristics already exist):

```python
uv run python -c "
from cc.helpers import create_decision

# Use example IDs from search results
create_decision('<LABEL_NAME>', <example_id>, True, '<reason>')
create_decision('<LABEL_NAME>', <example_id>, False, '<reason>')
"
```

### If only setting instructions (heuristics already exist):

```python
uv run python -c "
from cc.helpers import set_instructions

set_instructions('<LABEL_NAME>', '''<instructions text>''')
"
```

## Instruction Guidelines

When writing instructions:

1. **Acknowledge overlap** - If this label naturally overlaps with others, say so explicitly. E.g., "A 'Motion to Dismiss or in the Alternative for Summary Judgment' should receive BOTH labels."

2. **Don't require specific formatting** - Avoid rules like "must start with X" unless truly definitional. Documents come in many formats.

3. **Focus on function, not form** - Define labels by what the document DOES or IS, not how it's formatted.

4. **Include examples of edge cases** - Especially for overlapping labels or ambiguous cases.

5. **Mention combined/alternative filings** - Many motions are filed as alternatives (e.g., "Motion to Dismiss or in the Alternative Motion for Summary Judgment"). Make clear these get BOTH labels.

## Decision Guidelines

### CRITICAL: Use Example IDs

When creating decisions, ALWAYS use the `id` from search results:
- Search results show: `[1] (id=12345) WARRANT for Arrest...`
- Use: `create_decision('Warrant', 12345, True, 'reason')`
- NEVER paste the text directly

### Good Positive Examples
- Clear, unambiguous instances of the label
- Different variants/phrasings
- **Combined filings** - e.g., "Motion to Dismiss or for Summary Judgment" is positive for BOTH labels
- Use IDs from search results

### Good Negative Examples
- Documents that contain the keyword but are NOT the label
- Related but distinct document types
- Responses/oppositions TO the motion type (not the motion itself)
- Orders RULING ON the motion (not the motion itself)
- Explain WHY it's not a match

## Example: Motion Subtype Workflow

```python
# Step 1: Check status first
uv run python -c "
from cc.helpers import print_label_status
print_label_status('Motion for Summary Judgment')
"

# Output shows no heuristics, so proceed...

# Step 2: Explore patterns (note the IDs!)
uv run python -c "
from cc.helpers import count_matches, search_by_query, print_search_results

# Test minimal patterns - start broad
print('=== Minimal pattern options ===')
print(f'motion,summar: {count_matches(\"motion,summar\"):,}')  # Catches summary, summarize, etc.
print(f'motion,summary judgment: {count_matches(\"motion,summary judgment\"):,}')  # Phrase - narrower

# Test likely patterns - NO COMMAS
print('\\n=== Likely pattern options ===')
print(f'^motion for summary judgment: {count_matches(\"^motion for summary judgment\"):,}')
print(f'^motion for partial summary judgment: {count_matches(\"^motion for partial summary judgment\"):,}')
print(f'^cross motion for summary judgment: {count_matches(\"^cross motion for summary judgment\"):,}')

# Find examples
print('\\n=== Examples ===')
results = search_by_query('^motion for summary judgment', page_size=10)
print_search_results(results)

# Find combined/alternative motions (important edge cases!)
print('\\n=== Combined motions (should be positive!) ===')
results = search_by_query('^motion to dismiss,summary judgment', page_size=5)
print_search_results(results)
"

# Step 3: Setup with example IDs
uv run python -c "
from cc.helpers import setup_label, create_decision

setup_label(
    'Motion for Summary Judgment',
    minimal_query='motion,summar',  # Broad - catches summary, summarize, etc.
    likely_query='^motion for summary judgment|^motion for partial summary judgment|^cross motion for summary judgment',  # NO COMMAS
    instructions='''A Motion for Summary Judgment requests the court decide a case (or part of it) without trial.

Include:
- Motions for summary judgment
- Motions for partial summary judgment
- Cross-motions for summary judgment
- Combined motions (e.g., \"Motion to Dismiss or in the Alternative Motion for Summary Judgment\")

Exclude:
- Orders granting or denying summary judgment
- Oppositions to summary judgment motions
- Memoranda in support (unless combined with the motion)

Note: A \"Motion to Dismiss or for Summary Judgment\" should receive BOTH the Motion to Dismiss AND Motion for Summary Judgment labels.'''
)

print()
print('Adding decisions...')

# Clear positives
create_decision('Motion for Summary Judgment', 12345, True, 'Standard MSJ filing')
create_decision('Motion for Summary Judgment', 12346, True, 'Partial summary judgment motion')

# Combined motion - POSITIVE for this label
create_decision('Motion for Summary Judgment', 236813066, True,
    'Combined motion to dismiss OR summary judgment. This IS a motion for summary judgment (positive) even though it also contains a motion to dismiss.')

# Negatives
create_decision('Motion for Summary Judgment', 12347, False, 'Order ruling on MSJ, not the motion itself')
create_decision('Motion for Summary Judgment', 12348, False, 'Opposition to MSJ, not the motion')

print('Done!')
"
```

## Checklist Before Finishing

- [ ] Checked existing status first
- [ ] Minimal heuristic is BROAD (usually `motion,<keyword>` or just `<keyword>`)
- [ ] Likely heuristic has NO COMMAS (just OR alternatives)
- [ ] Instructions acknowledge label overlap where appropriate
- [ ] Instructions mention combined/alternative filings
- [ ] At least 4 decisions with mix of positive and negative
- [ ] Decisions use example IDs (not pasted text)
- [ ] Decisions include edge cases (combined motions, etc.)
