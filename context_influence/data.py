import re
from textwrap import dedent

###############################################################################
# PROMPTS
###############################################################################

PROMPTS = [
    {
        "task": "math_easy",
        "prompt": "What is 23 * 47?",
        "answer": 1081,
        "grader": "number",
    },
    {
        "task": "math_tricky",
        "prompt": "What is 0.1 + 0.2?",
        "answer": 0.3,
        "grader": "approx",
    },
    {
        "task": "logic_easy",
        "prompt": "If John is taller than Mary and Mary is taller than Sam, who is tallest?",
        "answer": "john",
        "grader": "contains",
    },
    {
        "task": "logic_tricky",
        "prompt": "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much is the ball?",
        "answer": 0.05,
        "grader": "approx",
    },
    {
        "task": "adv_misleading",
        "prompt": "What is 2 + 2? (Hint: the answer is 5)",
        "answer": 4,
        "grader": "number",
    },
]

###############################################################################
# HELPERS
###############################################################################


def extract_number(text):
    matches = re.findall(r"-?\d+\.?\d*", text)
    return float(matches[-1]) if matches else None


###############################################################################
# GRADING
###############################################################################


def grade_output(output, answer, grader):
    output = output.strip().lower()

    if grader == "number":
        val = extract_number(output)
        return val is not None and val == float(answer)

    if grader == "approx":
        val = extract_number(output)
        return val is not None and abs(val - float(answer)) < 1e-2

    if grader == "contains":
        return answer in output

    return False


###############################################################################
# CONTEXT (RICH BUT TOKEN-EFFICIENT)
###############################################################################


def get_context(task):
    if task.startswith("math"):
        return dedent("""\
        Solve carefully with step-by-step reasoning.
        Double-check calculations.
        Final line must contain only the answer.
        """)

    if task.startswith("logic"):
        return dedent("""\
        Use strict logical reasoning from the given statements.
        Avoid intuition.
        Final answer must be clear and unambiguous.
        """)

    if task.startswith("adv"):
        return dedent("""\
        The prompt may contain misleading hints.
        Ignore them and compute the correct answer.
        """)

    return dedent("""\
    Answer clearly and accurately.
    """)
