from typing import Any


def is_none(val: Any) -> bool:
    return val is None or (isinstance(val, str) and val.strip().lower() == "none")


def destruct(d, *keys):
    """
    Return a tuple of values from a dictionary based on the given keys.

    Examples:
    >>> d = {"a": 1, "b": 2, "c": 3}
    >>> a, b, c = destruct(d, "a", "b", "c")
    >>> a
    1
    >>> b
    2
    >>> c
    3
    """
    return (d[k] if k in d else None for k in keys)
