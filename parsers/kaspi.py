# kaspi_statement_parser.py
# Parser for Kaspi Bank PDF statements (Russian & English).
# Requires: pdfplumber OR PyPDF2, pandas.

import os
import re
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import pandas as pd

# ---------------- Constants / Regex ----------------

DATE_RE = re.compile(r"^(\d{2}\.\d{2}\.\d{2,4})\b")

# Canonical operation -> (default_sign, [aliases])
OP_MAP: Dict[str, Tuple[int, List[str]]] = {
    "Purchases":     (-1, ["Покупка", "Покупки"]),
    "Transfers":     (-1, ["Перевод", "Переводы"]),
    "Replenishment": (+1, ["Пополнение", "Пополнения"]),
    "Commission":    (-1, ["Комиссия"]),
    "Refund":        (+1, ["Возврат"]),
    "Cashback":      (+1, ["Кэшбэк"]),
    "Withdrawal":    (-1, ["Снятие", "Снятия", "Withdrawals"]),
    "Others":        (-1, ["Разное", "Другое"]),
}

# Reverse lookup: any name/alias -> (canonical, sign)
_OP_LOOKUP: Dict[str, Tuple[str, int]] = {}
for _canon, (_sign, _aliases) in OP_MAP.items():
    _OP_LOOKUP[_canon] = (_canon, _sign)
    for _alias in _aliases:
        _OP_LOOKUP[_alias] = (_canon, _sign)

# Regex from all names, longest first
_all_op_names = sorted(_OP_LOOKUP.keys(), key=len, reverse=True)
OP_RE = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(n) for n in _all_op_names) + r")(?!\w)"
)


# ---------------- Utilities ----------------

def normalize_ws(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def normalize_operation(raw_op: Optional[str]) -> Tuple[Optional[str], int]:
    """Return (canonical_name, default_sign) for a matched operation string."""
    if not raw_op:
        return None, -1
    cleaned = raw_op.replace("\n", " ").strip()
    hit = _OP_LOOKUP.get(cleaned)
    if hit:
        return hit
    # Case-insensitive fallback
    for key, val in _OP_LOOKUP.items():
        if key.lower() == cleaned.lower():
            return val
    return cleaned, -1


def parse_kaspi_date(s: str) -> Optional[str]:
    """Convert 05.11.25 or 05.11.2025 to 2025-11-05."""
    s = s.strip()
    try:
        if len(s.split(".")[-1]) == 4:
            dt = datetime.strptime(s, "%d.%m.%Y")
        else:
            dt = datetime.strptime(s, "%d.%m.%y")
        return dt.date().isoformat()
    except Exception:
        return None


def parse_amount_kzt(raw: str) -> Tuple[str, Optional[float]]:
    """Extract number with sign from various formats."""
    txt = normalize_ws(raw)

    sign = 1
    if txt.startswith("+"):
        sign = 1
        txt = txt[1:].strip()
    elif txt.startswith("-"):
        sign = -1
        txt = txt[1:].strip()

    if txt.endswith("-"):
        sign = -1
        txt = txt[:-1].strip()
    elif txt.endswith("+"):
        sign = 1
        txt = txt[:-1].strip()

    num = txt.replace(" ", "")

    if "," in num and "." in num:
        num = num.replace(",", "")
    elif "," in num:
        num = num.replace(",", ".")

    try:
        val = float(num) * sign
    except ValueError:
        return raw.strip(), None

    return raw.strip(), val


def extract_text_pages(pdf_path: str) -> List[str]:
    pages: List[str] = []
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(text)
        if pages:
            return pages
    except Exception:
        pass

    try:
        import PyPDF2
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                text = p.extract_text() or ""
                pages.append(text)
        return pages
    except Exception as e:
        raise RuntimeError(f"Cannot extract text from PDF '{pdf_path}': {e}")


# ---------------- Detection ----------------

def is_kaspi_statement(full_text: str) -> bool:
    t = full_text.lower()
    markers = [
        "kaspi bank", "www.kaspi.kz", "kaspi.kz",
        "kaspi gold", "касpi gold", "касpi.kz",
    ]
    return any(m in t for m in markers)


# ---------------- Header filtering ----------------

def is_header_like(line: str) -> bool:
    """Filter obvious headers/junk lines."""
    lw = line.lower()
    if not lw.strip():
        return True
    if "kaspi bank" in lw or "www.kaspi.kz" in lw or "kaspi.kz" in lw:
        return True
    if lw.startswith("выписка") or lw.startswith("kaspi gold") or "balance statement" in lw:
        return True
    if "краткое содержание" in lw or "transaction summary" in lw:
        return True
    # Table header row
    if ("дата" in lw and "сумма" in lw) or ("date" in lw and "amount" in lw):
        return True
    # Summary lines like "Доступно на" / "Card balance"
    if "доступно на" in lw or "card balance" in lw:
        return True
    # Summary items
    if any(k in lw for k in [
        "валюта счета", "currency:", "номер карты", "card number",
        "номер счета", "account number", "лимит на снятие",
        "cash withdrawal", "остаток зарплатных", "salary",
        "другие пополнения", "other deposits", "итого", "total",
    ]):
        return True
    # Blocked-amount disclaimers
    if "the amount is blocked" in lw or "сумма заблокирована" in lw:
        return True
    return False


# ---------------- Line parsing ----------------

def parse_operation_line(line: str) -> Optional[Dict]:
    """Parse a single Kaspi transaction line.

    Expected format:
      DD.MM.YY  [+/-] amount ₸  Operation  Details
    """
    m_date = DATE_RE.match(line)
    if not m_date:
        return None

    date_str = m_date.group(1)
    rest = line[m_date.end():].strip()
    date_iso = parse_kaspi_date(date_str)

    if not rest:
        return None

    # Amount: look for number block followed by ₸ or KZT
    m_amt = re.search(r"([+\-]?\s*[\d\s.,]+(?:[-+])?)\s*(₸|KZT)", rest)
    if not m_amt:
        m_amt = re.search(r"([+\-]?\s*[\d\s.,]+(?:[-+])?)\b", rest)
        if not m_amt:
            return None
    currency = "KZT"

    amount_raw_str = m_amt.group(1)
    amount_raw, amount_val = parse_amount_kzt(amount_raw_str)

    after_amt = rest[m_amt.end():].strip()

    # Operation: use bilingual OP_RE
    opm = OP_RE.search(after_amt)
    raw_op = opm.group(1) if opm else None
    operation, op_sign = normalize_operation(raw_op)

    # Details
    if raw_op and opm:
        details = after_amt[opm.end():].strip()
    else:
        details = after_amt.strip()

    # Clean newlines and strip quotes/brackets
    details = re.sub(r"\s+", " ", details).strip() if details else ""
    details = re.sub(r'["\'\[\]\(\)]', '', details).strip()

    # Sign: prefer explicit +/- from amount string
    sign = op_sign
    if amount_raw.strip().startswith("+"):
        sign = 1
    elif amount_raw.strip().startswith("-"):
        sign = -1

    amount = amount_val  # already has sign from parse_amount_kzt

    return {
        "date": date_iso,
        "amount_raw": f"{amount_raw} {currency}".strip(),
        "amount": amount,
        "currency": currency,
        "operation": operation,
        "details": details or None,
    }


# ---------------- Main parser ----------------

def parse_kaspi_pdf(pdf_path: str) -> List[Dict]:
    pages = extract_text_pages(pdf_path)
    if not pages:
        raise ValueError("PDF has no text")

    full_text = "\n".join(pages)
    if not is_kaspi_statement(full_text):
        raise ValueError("Not a Kaspi Bank statement")

    rows: List[Dict] = []
    current: Optional[Dict] = None

    for page in pages:
        for raw_line in page.splitlines():
            line = normalize_ws(raw_line)
            if not line:
                continue

            if is_header_like(line):
                continue

            parsed = parse_operation_line(line)
            if parsed:
                if current is not None:
                    rows.append(current)
                current = parsed
            else:
                # Continuation of previous transaction's details
                if current is not None:
                    extra = line.strip()
                    if not is_header_like(extra):
                        prev_details = str(current.get("details") or "")
                        current["details"] = (prev_details + " " + extra).strip()

    if current is not None:
        rows.append(current)

    # Add bank metadata
    for r in rows:
        r["bank"] = "Kaspi Bank"
        r["source_file"] = os.path.basename(pdf_path)

    return rows


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Kaspi Bank PDF statement parser -> CSV/XLSX"
    )
    ap.add_argument("--input", required=True, help="PDF file or folder path.")
    ap.add_argument("--out", required=True, help="Output CSV path.")
    ap.add_argument("--xlsx", help="Optional XLSX output path.")
    ap.add_argument("--dedupe", action="store_true", help="Remove duplicates.")
    ap.add_argument("--prune-empty", action="store_true", help="Remove empty columns.")

    args = ap.parse_args()

    pdf_files: List[str] = []
    if os.path.isdir(args.input):
        for root, _, files in os.walk(args.input):
            for name in files:
                if name.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, name))
        pdf_files.sort()
    elif os.path.isfile(args.input) and args.input.lower().endswith(".pdf"):
        pdf_files = [args.input]
    else:
        raise SystemExit("Provide an existing PDF file or folder.")

    if not pdf_files:
        raise SystemExit("No PDF files found.")

    all_rows: List[Dict] = []
    log: List[str] = []

    for path in pdf_files:
        try:
            rows = parse_kaspi_pdf(path)
            all_rows.extend(rows)
        except ValueError as e:
            log.append(f"[SKIP] {os.path.basename(path)}: {e}")
        except Exception as e:
            log.append(f"[ERROR] {os.path.basename(path)}: {e}")

    if not all_rows:
        raise SystemExit("Could not extract any transactions.")

    df = pd.DataFrame(all_rows)

    base_cols = [
        "date", "amount_raw", "amount", "currency",
        "operation", "details", "bank", "source_file",
    ]
    for c in base_cols:
        if c not in df.columns:
            df[c] = None
    df = df[base_cols + [c for c in df.columns if c not in base_cols]]

    if args.dedupe:
        df = df.drop_duplicates(subset=["date", "amount", "details"], keep="first")

    def _to_dt(x):
        try:
            return datetime.strptime(str(x), "%Y-%m-%d")
        except Exception:
            return pd.NaT

    if "date" in df.columns:
        df["_dt"] = df["date"].apply(_to_dt)
        df = df.sort_values(by=["_dt", "source_file"], kind="stable").drop(columns=["_dt"])

    if args.prune_empty:
        import numpy as np
        null_like = {"", " ", "null", "<null>", "None", "NaN", "nan"}
        for c in df.columns:
            df[c] = df[c].apply(
                lambda v: np.nan if isinstance(v, str) and v.strip() in null_like else v
            )
        df = df.dropna(axis=1, how="all")

    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    if args.xlsx:
        df.to_excel(args.xlsx, index=False)

    if log:
        log_path = os.path.splitext(args.out)[0] + "_kaspi_log.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            for line in log:
                f.write(line + "\n")
        print(f"[INFO] Log: {log_path}")

    print(f"Done: {args.out} ({len(df)} transactions)")
    if args.xlsx:
        print(f"XLSX: {args.xlsx}")


if __name__ == "__main__":
    main()
