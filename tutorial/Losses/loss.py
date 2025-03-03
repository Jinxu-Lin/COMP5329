from Losses.cross_entropy import CrossEntropy

def cross_entropy(logits, targets):
    return CrossEntropy(logits, targets)