PROJECT_UNDERSTANDING = """
# Step 1: Understand the data

Search the project data to understand what this project is about. Start with \
a broad sample (empty query, ~20 results), but if documents are short, pull \
more to get a representative picture.

# Step 2: Clarify your understanding of the project

Using the project name and data, form your best hypothesis about the project's \
goals and domain. Then ask the user a series of targeted questions to validate \
and deepen your understanding. Keep asking follow-ups until you have a thorough \
grasp of the project's aims, scope, and any nuances.

# Step 3: Update the project instructions

Once you have a solid understanding, update the project instructions with a \
detailed overview covering the project's purpose, domain, data characteristics, \
and any important context.

> Note: the active label in your system prompt may be one of many. Focus your \
questions on the project as a whole, not the currently active label.
"""

LABEL_UNDERSTANDING = """
# Step 1: Understand the data

Search the project data to understand what this label should capture. Use the \
label name, project instructions, and the data itself to form a hypothesis \
about what documents should be included and excluded.

Be thorough in your search — look for clear positives, clear negatives, and \
edge cases. Try different query patterns to surface tricky examples that might \
be ambiguous.

# Step 2: Clarify your understanding of the label

Then ask the user targeted questions to validate your understanding of the \
label's criteria. Keep asking follow-ups until you can confidently distinguish \
what belongs under this label and what doesn't. You can ask multiple questions \
during each turn. Be very thorough.

# Step 3: Update the label instructions

Once ready, update the label instructions with detailed, specific annotation \
guidelines. Include clear inclusion/exclusion criteria and address any edge \
cases you identified.
"""


SAMPLING_STRATEGY = """
# Step 1: Get a sense of training set size

The expected size for the initial training set should be mentioned in the \
project instructions. If it isn't, ask the user and update the project instructions \
to reflect their answer.

# Step 2: Come up with minimal and likely heuristics

Search through the data an try to come up with two types of queries:

- A minimal heuristic: This should be broad enough such that no positive example would \
ever plausibly be excluded by the query. If the language of positive examples is \
such that it would not be possible to scope examples with keyword conditions, \
then you might leave this blank.

- A likely heuristic: This should be narrow enough such that it catches some obvious \
positive examples. It does not need to be perfect or complete, but it should catch \
many easy positives.

# Step 3: Store the heuristics in the label instructions

Add a note detailing the function of the heuristics and their queries to the label \
instructions.

# Step 4: Create the initial sample

Sample approximately 1/3 of the expected training set size from three buckets:

- Things that do not satisfy the minimal heuristic.
- Things that satisfy the minimal heuristic but not the likely heuristic.
- Things that satisfy the likely heuristic.

Note: When sampling based on a search, you can set num_examples to a much higher \
number than the num_results used for the search tool. For example, if you wanted to \
sample in 1000 examples, you might do a search with num_results=5 and then sample 1000 \
examples from that search. This is encouraged so that you don't need to pull literally \
every example into context.

# Step 5: Target specific language

You should perform additional queries to sample in any specific language that is discussed \
in the label instructions. How many examples will depend on dataset size, but feel free to \
grow the dataset size by ~20% with the stuff you pull in. Be thorough to capture any edge cases
whose representation needs boosted.

# Step 6: Target unknown language

Try to come up with queries that exclude as many things that are easily targetable. The goal here \
is to sample in even more underrepresented language that will not be easily targeted by the other \
queries. These should be examples that live between the minimal and likely heuristics, but which \
are even more narrowly targeted.
"""

ANNOTATE = """
# Step 1: Annotate remaining examples

Annotate all unannotated training examples for this label. Search for \
unannotated documents (use annotation='none', query=None) and classify each one as 'yes', \
'no', or 'skip' based on the label instructions.

Work through the unannotated examples in batches of up to 100 examples. For each batch, \
read the documents carefully and apply the label criteria consistently. Use 'yes' for \
clear matches, 'no' for clear non-matches, and 'skip' only for documents that \
are genuinely ambiguous or where you cannot make a confident determination.

If the label instructions are unclear or you encounter edge cases not covered \
by the guidelines, use your tool to ask the user for clarification so that you \
can update the instructions before proceeding.

After each batch, call the ClearToolHistory tool to make some extra room for context.

# Step 2: Complete after all annotations or 3 batches

You should continue annotating until all unannotated examples have been annotated or \
until you've processed 3 batches. No matter what, do not end your turn without calling \
the CompleteTask tool. Failing to do so will halt the process and require user intervention, \
which we want to avoid (unless you are asking the user for feedback / clarification).

> Note: If asked to compact your memory, you don't need to store specific annotations or \
document IDs.
"""


prompt_registry = {
    "project_understanding": {
        "name": "Project Understanding",
        "content": PROJECT_UNDERSTANDING.strip(),
    },
    "label_understanding": {
        "name": "Label Understanding",
        "content": LABEL_UNDERSTANDING.strip(),
    },
    "sampling_strategy": {
        "name": "Sampling Strategy",
        "content": SAMPLING_STRATEGY.strip(),
    },
    "annotate": {
        "name": "Annotate",
        "content": ANNOTATE.strip(),
    },
}
