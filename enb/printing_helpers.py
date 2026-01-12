import math
from typing import Optional, Any


def _fmt_float(x: Optional[float], digits: int = 6) -> str:
    if x is None:
        return "—"
    try:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return str(x)
        return f"{x:.{digits}f}"
    except Exception:
        return str(x)

def _fmt_int(x: Optional[int]) -> str:
    if x is None:
        return "—"
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)

def _truncate(s: Optional[str], max_len: int = 140) -> str:
    if not s:
        return ""
    s = " ".join(s.split())  # collapse whitespace
    return s if len(s) <= max_len else s[: max_len - 1] + "…"

def _print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))

def _print_kv(rows: list[tuple[str, str]]) -> None:
    if not rows:
        return
    key_w = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"{k:<{key_w}} : {v}")

def _print_table(headers: list[str], rows: list[list[Any]]) -> None:
    if not headers:
        return
    # Convert all cells to strings
    srows = [[("" if c is None else str(c)) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in srows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def fmt_row(r: list[str]) -> str:
        return "  ".join(f"{c:<{widths[i]}}" for i, c in enumerate(r))

    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for r in srows:
        print(fmt_row(r))
