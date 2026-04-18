def get_category(label):
    if label.startswith("math"):
        return "math"
    if label.startswith("code"):
        return "code"
    if label.startswith("logic"):
        return "logic"
    if label.startswith("lang"):
        return "language"
    if label.startswith("adv"):
        return "adversarial"
    return "other"


categories = ["math", "code", "logic", "language", "adversarial"]
cat_to_idx = {c: i for i, c in enumerate(categories)}