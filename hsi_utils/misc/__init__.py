from typing import Any

def is_none(val: Any) -> bool:
    return val is None or (isinstance(val, str) and val.strip().lower() == "none")