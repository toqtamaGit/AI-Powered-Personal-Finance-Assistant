# bank_statement_parser_plus.py

import os
import re
import glob
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import pandas as pd

# --- Константы / регулярки ---------------------------------------------------

DATE_RE = re.compile(r"(^|\s)(\d{2}\.\d{2}\.\d{4})(?=\s|$)")
CURRENCY_MAP = {"₸": "KZT", "$": "USD", "€": "EUR", "₽": "RUB"}

AMOUNT_RE = re.compile(
    r"\b([+-]?\d[\d\s,\.]*)\s*([₸$€₽]?)\s*(KZT|USD|EUR|RUB)?\b"
)

OP_RE = re.compile(
    r"\b(Покупка|Платеж|Пополнение|Перевод|Снятие|Другое|Комиссия|Возврат|Кэшбэк)\b"
)

MCC_RE = re.compile(r"\bMCC\s*(\d{4})\b", re.IGNORECASE)

CARD_MASK_RE = re.compile(
    r"\b(\d{4}\*{4}\d{4}|\d{4}\s\*{4}\s\d{4})\b"
)

BALANCE_RE = re.compile(
    r"(?:Баланс|Остаток)\s*[:\-]?\s*([+-]?\d[\d\s,\.]*)\s*"
    r"(KZT|USD|EUR|RUB|₸|\$|€|₽)?",
    re.IGNORECASE,
)

# FX-блоки
FX_BLOCK_RES = [
    re.compile(
        r"Сумма\s*в\s*валюте\s*операции\s*[:\-]?\s*"
        r"([+-]?\d[\d\s,\.]*)\s*(USD|EUR|RUB|KZT|\$|€|₽|₸)",
        re.IGNORECASE,
    ),
    re.compile(r"Курс\s*[:\-]?\s*([\d,\.]+)", re.IGNORECASE),
    re.compile(
        r"(Списано|Итого\s*списано)\s*[:\-]?\s*"
        r"([+-]?\d[\d\s,\.]*)\s*(KZT|USD|EUR|RUB|₸|\$|€|₽)",
        re.IGNORECASE,
    ),
]

# Базовые категории (можно переопределить через YAML)
DEFAULT_CATEGORIES: Dict[str, List[str]] = {
    "еда": [
        "бургер", "кофе", "cafe", "restaurant", "stolovaya",
        "food", "еда", "пицца", "kfc", "burger", "doner",
    ],
    "транспорт": [
        "taxi", "yandex go", "индер", "bus", "metro", "aero", "rail",
    ],
    "продукты": [
        "magnum", "small", "grocery", "market", "supermarket",
    ],
    "онлайн-сервисы": [
        "spotify", "netflix", "aviata", "steam", "apple", "google",
    ],
    "мобильная связь": [
        "tele2", "kcell", "beeline", "altel",
    ],
    "аптека/здоровье": [
        "аптека", "pharma", "apteka", "doctor",
    ],
    "наличные": [
        "снятие", "atm", "банкомат",
    ],
    "переводы": [
        "перевод", "p2p", "transfer", "внутренний перевод",
    ],
}


# --- Утилиты ------------------------------------------------------------------


def normalize_ws(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def parse_amount(s: str) -> Optional[float]:
    if not s:
        return None
    s = s.strip()
    sign = -1 if s.startswith("-") else 1
    s_clean = re.sub(r"[^\d,.\-]", "", s)

    if "," in s_clean and "." in s_clean:
        # и запятая, и точка: запятая как разделитель тысяч
        s_clean = s_clean.replace(",", "")
    else:
        # только запятая: если ,\d{3} — считаем разделитель тысяч
        if re.search(r",\d{3}\b", s_clean):
            s_clean = s_clean.replace(",", "")
        else:
            s_clean = s_clean.replace(",", ".")

    try:
        return sign * float(s_clean)
    except ValueError:
        return None


def extract_text_pages(pdf_path: str) -> List[str]:
    pages: List[str] = []

    # 1) pdfplumber
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

    # 2) PyPDF2
    try:
        import PyPDF2

        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                text = p.extract_text() or ""
                pages.append(text)
        return pages
    except Exception as e:
        raise RuntimeError(f"Не удалось извлечь текст из PDF '{pdf_path}': {e}")


def extract_tables(pdf_path: str) -> List[pd.DataFrame]:
    """Пробуем вытащить таблицы, если банк рисует таблично."""
    dfs: List[pd.DataFrame] = []
    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for t in tables or []:
                    df = (
                        pd.DataFrame(t)
                        .dropna(how="all", axis=1)
                        .dropna(how="all", axis=0)
                    )
                    if not df.empty:
                        dfs.append(df)
    except Exception:
        pass
    return dfs


def map_table_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Пытаемся привести заголовки к: date, amount_raw, currency, operation, details."""
    df2 = df.copy()
    header_row = df2.iloc[0].astype(str).str.lower()
    if any("дата" in x for x in header_row):
        df2.columns = header_row
        df2 = df2.iloc[1:].reset_index(drop=True)

    colmap: Dict[str, str] = {}
    for c in df2.columns:
        lc = str(c).lower()
        if "дата" in lc:
            colmap[c] = "date"
        elif "сумм" in lc or "amount" in lc:
            colmap[c] = "amount_raw"
        elif "валют" in lc or "curr" in lc:
            colmap[c] = "currency"
        elif "операц" in lc or "тип" in lc:
            colmap[c] = "operation"
        elif any(
            key in lc
            for key in ("детал", "описан", "назнач", "merchant")
        ):
            colmap[c] = "details"

    df2 = df2.rename(columns=colmap)
    return df2


def chunk_transactions_from_lines(lines: List[str]) -> List[List[str]]:
    """Режем текст по строкам, которые начинаются с даты."""
    chunks: List[List[str]] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if DATE_RE.search(ln):
            current = [ln]
            j = i + 1
            while j < len(lines) and not DATE_RE.search(lines[j]):
                if lines[j].strip():
                    current.append(lines[j])
                j += 1
            chunks.append(current)
            i = j
        else:
            i += 1
    return chunks


def guess_merchant(details: str) -> Optional[str]:
    ds = details or ""

    # 1) ТОО/ИП "..."
    m = re.search(r'(?:ТОО|ИП)\s*[«"]\s*([^»"]+?)\s*[»"]', ds)
    if m:
        return m.group(1).strip()

    # 2) любые кавычки
    m = re.search(r'[«"]\s*([^»"]+?)\s*[»"]', ds)
    if m:
        return m.group(1).strip()

    # 3) Kaspi/Halyk/QR
    m = re.search(
        r"(KASPI\s*QR|QR\s*оплата|HALYK\s*POS|HALYK\s*QR|Kaspi Red|Kaspi Pay)",
        ds,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    # 4) CAPS + город/страна
    m = re.search(
        r"\b([A-Z0-9][A-Z0-9\s\.\-\/&']{3,}?)\s+"
        r"(?:ASTANA|NUR-SULTAN|ALMATY|KZ|KAZAKHSTAN)\b",
        ds,
    )
    if m:
        return m.group(1).strip()

    # 5) ТОО/ИП без кавычек
    m = re.search(
        r"\b(ТОО|ИП)\s+([A-Za-zА-ЯЁа-яё0-9\"«»\-\.\s]+)",
        ds,
    )
    if m:
        tail = m.group(2).split("За ")[0].split(".")[0]
        return tail.strip()

    # 6) домен/email
    m = re.search(
        r"\b([A-Za-z0-9\-]+\.(?:kz|ru|com|io|org))\b",
        ds,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    # 7) fallback — первые 5–6 слов
    if ds:
        return " ".join(ds.split()[:6])

    return None


def parse_fx_block(
    details: str,
) -> Tuple[Optional[float], Optional[str], Optional[float], Optional[float], Optional[str]]:
    """Парсим подсказки по валютным операциям."""
    if not details:
        return None, None, None, None, None

    orig_amount = orig_curr = None
    rate = charged_amt = charged_curr = None

    m1 = FX_BLOCK_RES[0].search(details)
    if m1:
        orig_amount = parse_amount(m1.group(1))
        cur = (
            m1.group(2)
            .upper()
            .replace("$", "USD")
            .replace("€", "EUR")
            .replace("₽", "RUB")
            .replace("₸", "KZT")
        )
        orig_curr = cur

    m2 = FX_BLOCK_RES[1].search(details)
    if m2:
        rate = parse_amount(m2.group(1))

    m3 = FX_BLOCK_RES[2].search(details)
    if m3:
        charged_amt = parse_amount(m3.group(2))
        cur = (
            m3.group(3)
            .upper()
            .replace("$", "USD")
            .replace("€", "EUR")
            .replace("₽", "RUB")
            .replace("₸", "KZT")
        )
        charged_curr = cur

    return orig_amount, orig_curr, rate, charged_amt, charged_curr


def detect_bank_from_text(text: str) -> Optional[str]:
    """Грубая детекция банка по шапке/тексту выписки."""
    t = text.lower()

    # Freedom Bank
    if (
        "фридом банк казахстан" in t
        or 'ao "фридом банк казахстан"' in t
        or "ao «фридом банк казахстан»" in t
        or "freedom bank kazakhstan" in t
        or "bankffin.kz" in t
        or "выписка по карте super card" in t
    ):
        return "Freedom Bank Kazakhstan"

    # здесь можно добавить Kaspi/Halyk/Jusan и т.д.
    return None


def extract_fields_from_chunk(chunk: List[str]) -> Dict:
    text = normalize_ws(" ".join(chunk))

    # дата
    mdate = DATE_RE.search(text)
    date_iso = None
    if mdate:
        date_str = mdate.group(2)
        try:
            date_iso = datetime.strptime(
                date_str, "%d.%m.%Y"
            ).date().isoformat()
        except Exception:
            date_iso = date_str

    post = text[mdate.end():] if mdate else text

    # сумма/валюта
    am = AMOUNT_RE.search(post)
    amount_raw = amount_val = currency = None
    if am:
        amount_raw = am.group(1)
        amount_val = parse_amount(amount_raw)
        symbol = (am.group(2) or "").strip()
        code = (am.group(3) or "").strip()
        currency = code or (
            CURRENCY_MAP.get(symbol) if symbol in CURRENCY_MAP else None
        )

    # тип операции
    opm = OP_RE.search(post)
    operation = opm.group(1) if opm else None

    # детали
    if operation:
        details = post.split(operation, 1)[1].strip()
    elif am:
        details = post.split(am.group(0), 1)[1].strip()
    else:
        details = post.strip()

    merchant = guess_merchant(details)

    # знак
    sign = None
    if isinstance(amount_raw, str):
        if amount_raw.strip().startswith("+"):
            sign = 1
        elif amount_raw.strip().startswith("-"):
            sign = -1
    if sign is None:
        sign = 1 if operation in ("Пополнение", "Возврат", "Кэшбэк") else -1
    amount = (
        amount_val * sign
        if isinstance(amount_val, (int, float))
        else None
    )

    # MCC, карта, баланс
    mcc_m = MCC_RE.search(details)
    mcc = mcc_m.group(1) if mcc_m else None

    card_m = CARD_MASK_RE.search(details)
    card_mask = card_m.group(1) if card_m else None

    bal_m = BALANCE_RE.search(details)
    balance_amt = (
        parse_amount(bal_m.group(1)) if bal_m else None
    )
    balance_curr = None
    if bal_m and bal_m.group(2):
        bc = (
            bal_m.group(2)
            .upper()
            .replace("$", "USD")
            .replace("€", "EUR")
            .replace("₽", "RUB")
            .replace("₸", "KZT")
        )
        balance_curr = bc

    # FX
    (
        fx_orig_amt,
        fx_orig_cur,
        fx_rate,
        fx_charged_amt,
        fx_charged_cur,
    ) = parse_fx_block(details)

    return {
        "date": date_iso,
        "amount_raw": amount_raw,
        "amount": amount,
        "amount_abs": (
            abs(amount)
            if isinstance(amount, (int, float))
            else None
        ),
        "currency": currency,
        "operation": operation,
        "merchant": merchant,
        "details": details,
        "mcc": mcc,
        "card_mask": card_mask,
        "balance_after": balance_amt,
        "balance_currency": balance_curr,
        "fx_orig_amount": fx_orig_amt,
        "fx_orig_currency": fx_orig_cur,
        "fx_rate": fx_rate,
        "fx_charged_amount": fx_charged_amt,
        "fx_charged_currency": fx_charged_cur,
    }


def try_parse_tables(pdf_path: str) -> List[Dict]:
    """Парсим, если PDF — нормальная таблица."""
    out: List[Dict] = []
    tdfs = extract_tables(pdf_path)
    for df in tdfs:
        dfm = map_table_headers(df)
        if "date" not in dfm.columns:
            continue
        for _, row in dfm.iterrows():
            date_iso = None
            if pd.notna(row.get("date")):
                s = str(row["date"]).strip()
                m = DATE_RE.search(" " + s)
                if m:
                    try:
                        date_iso = datetime.strptime(
                            m.group(2), "%d.%m.%Y"
                        ).date().isoformat()
                    except Exception:
                        date_iso = m.group(2)

            ar = str(
                row.get("amount_raw") or row.get("amount") or ""
            ).strip()
            am_val = parse_amount(ar) if ar else None

            curr = row.get("currency")
            curr = (
                str(curr).strip()
                if pd.notna(curr)
                else None
            )

            op = (
                str(row.get("operation")).strip()
                if pd.notna(row.get("operation"))
                else None
            )
            details = (
                str(row.get("details")).strip()
                if pd.notna(row.get("details"))
                else ""
            )

            merchant = guess_merchant(details)
            sign = 1 if op in ("Пополнение", "Возврат", "Кэшбэк") else -1
            amount = (
                am_val * sign
                if isinstance(am_val, (int, float))
                else None
            )

            out.append(
                {
                    "date": date_iso,
                    "amount_raw": ar or None,
                    "amount": amount,
                    "amount_abs": (
                        abs(amount)
                        if isinstance(amount, (int, float))
                        else None
                    ),
                    "currency": curr,
                    "operation": op,
                    "merchant": merchant,
                    "details": details,
                    "mcc": None,
                    "card_mask": None,
                    "balance_after": None,
                    "balance_currency": None,
                    "fx_orig_amount": None,
                    "fx_orig_currency": None,
                    "fx_rate": None,
                    "fx_charged_amount": None,
                    "fx_charged_currency": None,
                }
            )
    return out


def parse_pdf(pdf_path: str) -> Tuple[List[Dict], List[str]]:
    """
    Возвращает (records, rejects).
    Здесь же вставляем detect_bank_from_text, чтобы каждому record приписать bank.
    """
    pages = extract_text_pages(pdf_path)
    full_text = "\n".join(pages)
    bank_name = detect_bank_from_text(full_text)

    # 1) Табличный режим
    records = try_parse_tables(pdf_path)
    if records:
        if bank_name:
            for r in records:
                r["bank"] = bank_name
        return records, []

    # 2) Построчный режим
    rejects: List[str] = []
    for page_text in pages:
        lines = [
            normalize_ws(x)
            for x in page_text.splitlines()
            if normalize_ws(x)
        ]
        chunks = chunk_transactions_from_lines(lines)
        if not chunks:
            chunks = [lines]

        for ch in chunks:
            rec = extract_fields_from_chunk(ch)
            if not rec.get("date") and not rec.get("amount"):
                rejects.append(" ".join(ch))
            else:
                if bank_name:
                    rec["bank"] = bank_name
                records.append(rec)

    return records, rejects


def collect_pdfs(input_path: str) -> List[str]:
    """Собираем список PDF (рекурсивно по папке или один файл)."""
    if os.path.isdir(input_path):
        pats = [
            os.path.join(input_path, "**", "*.pdf"),
            os.path.join(input_path, "**", "*.PDF"),
        ]
        files: List[str] = []
        for p in pats:
            files.extend(glob.glob(p, recursive=True))
        files = sorted(set(files))
        if not files:
            raise FileNotFoundError(f"В папке нет PDF: {input_path}")
        return files

    if os.path.isfile(input_path) and input_path.lower().endswith(".pdf"):
        return [input_path]

    raise FileNotFoundError(f"Не найден PDF или папка: {input_path}")


def load_categories(path: Optional[str]) -> Dict[str, List[str]]:
    cats = DEFAULT_CATEGORIES.copy()
    if not path:
        return cats
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if isinstance(v, list):
                cats[k] = v
    except Exception as e:
        print(f"[WARN] не удалось загрузить категории из {path}: {e}")
    return cats


def classify_category(
    merchant: Optional[str],
    details: Optional[str],
    cats: Dict[str, List[str]],
) -> Optional[str]:
    text = " ".join([merchant or "", details or ""]).lower()
    best_cat = None
    best_hit = 0
    for cat, kws in cats.items():
        hits = sum(1 for kw in kws if kw.lower() in text)
        if hits > best_hit:
            best_cat, best_hit = cat, hits
    return best_cat


# --- main --------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Усиленный парсер банковских PDF-выписок -> CSV/XLSX."
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Путь к PDF или папке (рекурсивно).",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Путь к CSV.",
    )
    ap.add_argument(
        "--xlsx",
        help="Опционально — путь к XLSX.",
    )
    ap.add_argument(
        "--categories",
        help="YAML: {категория: [ключевые слова]} для автокатегоризации.",
    )
    ap.add_argument(
        "--dedupe",
        action="store_true",
        help="Удалить дубликаты (date, amount, currency, merchant).",
    )
    ap.add_argument(
        "--prune-empty",
        action="store_true",
        help="Удалить колонки, где все значения пустые/NaN.",
    )
    ap.add_argument(
        "--match",
        help="Глоб-шаблон по имени файла внутри папки (например, *2025-04*.pdf).",
    )
    ap.add_argument(
        "--regex",
        help="Regex по полному пути/имени файла.",
    )
    ap.add_argument(
        "--latest",
        action="store_true",
        help="Взять самый свежий файл (по mtime) после фильтров.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Ограничить число файлов после фильтров (0 = без ограничения).",
    )

    args = ap.parse_args()

    pdfs = collect_pdfs(args.input)

    # фильтры выбора файлов
    from pathlib import Path

    def _apply_filters(files: List[str]) -> List[str]:
        out = list(files)

        # --match по basename
        if args.match:
            pat = args.match
            out = [f for f in out if Path(f).match(pat)]

        # --regex по полному пути
        if args.regex:
            rx = re.compile(args.regex)
            out = [f for f in out if rx.search(f)]

        # --latest: берём один самый свежий
        if args.latest and out:
            out = [max(out, key=lambda p: os.path.getmtime(p))]

        # --limit
        if args.limit and args.limit > 0:
            out = out[: args.limit]

        return out

    pdfs = _apply_filters(pdfs)
    if not pdfs:
        raise SystemExit("После применения фильтров файлы не найдены.")

    cats = load_categories(args.categories)

    all_rows: List[Dict] = []
    all_rejects: List[str] = []

    for f in pdfs:
        try:
            rows, rejects = parse_pdf(f)
            for r in rows:
                r["source_file"] = os.path.basename(f)
            all_rows.extend(rows)
            all_rejects.extend(rejects)
        except Exception as e:
            all_rejects.append(
                f"[{os.path.basename(f)}] ERROR: {e}"
            )

    if not all_rows:
        raise SystemExit("Не удалось извлечь ни одной операции.")

    # Формируем DataFrame
    df = pd.DataFrame(all_rows)

    # Гарантируем наличие всех колонок и порядок
    desired_cols = [
        "date",
        "amount_raw",
        "amount",
        "amount_abs",
        "currency",
        "operation",
        "merchant",
        "details",
        "mcc",
        "card_mask",
        "balance_after",
        "balance_currency",
        "fx_orig_amount",
        "fx_orig_currency",
        "fx_rate",
        "fx_charged_amount",
        "fx_charged_currency",
        "bank",
        "source_file",
    ]
    for c in desired_cols:
        if c not in df.columns:
            df[c] = None
    df = df[desired_cols + [c for c in df.columns if c not in desired_cols]]

    # Категоризация
    df["category"] = df.apply(
        lambda r: classify_category(
            r.get("merchant"), r.get("details"), cats
        ),
        axis=1,
    )

    # Дедупликация
    if args.dedupe:
        df = df.drop_duplicates(
            subset=["date", "amount", "currency", "merchant"],
            keep="first",
        )

    # Сортировка по дате
    def _try_dt(x):
        try:
            return datetime.strptime(str(x), "%Y-%m-%d")
        except Exception:
            return pd.NaT

    df["_dt"] = df["date"].apply(_try_dt)
    df = df.sort_values(
        by=["_dt", "source_file"], kind="stable"
    ).drop(columns=["_dt"])

    # Удаление полностью пустых колонок
    if args.prune_empty:
        import numpy as np

        null_like = {
            "",
            " ",
            "null",
            "<null>",
            "None",
            "NaN",
            "nan",
        }
        for c in df.columns:
            df[c] = df[c].apply(
                lambda x: np.nan
                if isinstance(x, str)
                and x.strip() in null_like
                else x
            )
        df = df.dropna(axis=1, how="all")

    # Сохранение
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    if args.xlsx:
        df.to_excel(args.xlsx, index=False)

    # Лог проблемных кусков
    if all_rejects:
        rej_path = (
            os.path.splitext(args.out)[0]
            + "_rejects.txt"
        )
        with open(rej_path, "w", encoding="utf-8") as f:
            for line in all_rejects:
                f.write(line.strip() + "\n")
        print(
            f"[INFO] Пропущено/не разобрано: {len(all_rejects)} строк(и). "
            f"См. {rej_path}"
        )

    print(
        f"Готово: {args.out} (всего операций: {len(df)})"
    )
    if args.xlsx:
        print(f"Также создан XLSX: {args.xlsx}")


if __name__ == "__main__":
    main()
