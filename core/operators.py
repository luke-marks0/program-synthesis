import torch

ALL_OPERATIONS = [
    "zero",           # 0: always 0
    "and",            # 1: a AND b
    "not_implies",    # 2: a AND NOT b
    "a",              # 3: a
    "not_implied_by", # 4: NOT a AND b
    "b",              # 5: b
    "xor",            # 6: a XOR b
    "or",             # 7: a OR b
    "not_or",         # 8: NOT (a OR b)
    "not_xor",        # 9: NOT (a XOR b)
    "not_b",          # 10: NOT b
    "implied_by",     # 11: a OR NOT b
    "not_a",          # 12: NOT a
    "implies",        # 13: NOT a OR b
    "not_and",        # 14: NOT (a AND b)
    "one",            # 15: always 1
]


def bin_op(a, b, op_idx):
    """Apply a specific binary logic operation.

    Args:
        a: First input tensor
        b: Second input tensor
        op_idx: Index of the operation to apply (0-15)

    Returns:
        Result of applying the operation
    """
    if op_idx == 0:    # zero
        return torch.zeros_like(a)
    elif op_idx == 1:  # and
        return a * b
    elif op_idx == 2:  # not_implies (a AND NOT b)
        return a * (1 - b)
    elif op_idx == 3:  # a
        return a
    elif op_idx == 4:  # not_implied_by (NOT a AND b)
        return (1 - a) * b
    elif op_idx == 5:  # b
        return b
    elif op_idx == 6:  # xor
        return a + b - 2 * a * b
    elif op_idx == 7:  # or
        return a + b - a * b
    elif op_idx == 8:  # not_or
        return 1 - (a + b - a * b)
    elif op_idx == 9:  # not_xor
        return 1 - (a + b - 2 * a * b)
    elif op_idx == 10: # not_b
        return 1 - b
    elif op_idx == 11: # implied_by (a OR NOT b)
        return 1 - b + a * b
    elif op_idx == 12: # not_a
        return 1 - a
    elif op_idx == 13: # implies (NOT a OR b)
        return 1 - a + a * b
    elif op_idx == 14: # not_and
        return 1 - a * b
    elif op_idx == 15: # one
        return torch.ones_like(a)
    else:
        raise ValueError(f"Invalid operation index: {op_idx}")


def bin_op_s(a, b, probs):
    """Apply all binary logic operations with their probabilities.

    Args:
        a: First input tensor
        b: Second input tensor
        probs: Probabilities for each operation

    Returns:
        Weighted combination of all operations
    """
    result = torch.zeros_like(a)

    # weighted sum of gates
    for i in range(16):
        result += probs[..., i] * bin_op(a, b, i)

    return result
