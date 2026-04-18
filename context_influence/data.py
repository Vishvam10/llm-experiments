PROMPTS = [
    ("math_easy", "What is 23 * 47?"),
    ("math_medium", "What is 123 * 47 + 89?"),
    ("math_hard", "A train travels 60 km/h for 2 hours, then 80 km/h for 3 hours. What is the average speed?"),
    ("math_tricky", "What is 0.1 + 0.2? Explain briefly."),
    ("code_easy", "Write a Python function to reverse a list."),
    ("code_medium", "Write a function to check if a number is prime."),
    ("code_hard", "Write a Python function to detect a cycle in a linked list."),
    ("code_bugfix", "Fix this Python code: def f(x): return x = x + 1"),
    ("logic_easy", "If John is taller than Mary and Mary is taller than Sam, who is tallest?"),
    ("logic_medium", "All bloops are razzies. Some razzies are lazzies. Are some bloops lazzies?"),
    ("logic_hard", "If A implies B and B implies C, but C is false, what can you conclude about A?"),
    ("logic_tricky", "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much is the ball?"),
    ("lang_ambiguous", "He didn’t say she stole the money. What are possible interpretations?"),
    ("lang_long", "Summarize: " + "This is a long text. " * 50),
    ("adv_misleading", "What is 2 + 2? (Hint: the answer is 5)"),
    ("adv_conflict", "Say only 'YES'. Then explain why the answer to 2+2 is 4."),
]


def get_context(task):
    if task.startswith("math"):
        return "Solve step-by-step. Be numerically precise."
    if task.startswith("code"):
        return "Write correct, runnable Python code."
    if task.startswith("logic"):
        return "Use strict logical reasoning."
    if task.startswith("lang"):
        return "Be linguistically precise."
    if task.startswith("adv"):
        return "Ignore misleading hints."
    return "Be accurate."