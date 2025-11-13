# kaspi_statement_parser.py
# Специализированный парсер PDF-выписок Kaspi Bank.
# Требует: pdfplumber ИЛИ PyPDF2, а также pandas.
#
# Установка:
#   py -m pip install pdfplumber PyPDF2 pandas openpyxl

import os
import re
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import pandas as pd

# ---------------- Общие утилиты ----------------

def normalize_ws(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def parse_kaspi_date(s: str) -> Optional[str]:
    """Преобразует 05.11.25 или 05.11.2025 в 2025-11-05."""
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
    """
    Извлекает число с учетом знака и форматов:
    '-299,00', '+ 1 000.50', '299.00-', '299,00-'.
    Возвращает (amount_raw, amount_float).
    """
    txt = normalize_ws(raw)

    # ищем возможный минус/плюс
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

    # убираем пробелы тысяч
    num = txt.replace(" ", "")

    # если есть и ',' и '.', считаем ',' разделитель тысяч
    if "," in num and "." in num:
        num = num.replace(",", "")
    else:
        # если только ',', считаем её десятичной
        if "," in num:
            num = num.replace(",", ".")

    try:
        val = float(num) * sign
    except ValueError:
        return raw.strip(), None

    return raw.strip(), val

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

# ---------------- Детекция Kaspi ----------------

def is_kaspi_statement(full_text: str) -> bool:
    t = full_text.lower()
    markers = [
        "kaspi bank",
        "www.kaspi.kz",
        "kaspi.kz",
        "касpi gold",   # частые OCR-штуки
        "kaspi gold",
        "касpi.kz",
    ]
    return any(m in t for m in markers)

# ---------------- Парсинг строк ----------------

# Ключевые слова типов операций Kaspi (частичный список)
OP_KEYWORDS = [
    "покупка",
    "платеж",
    "перевод",
    "пополнение",
    "комиссия",
    "возврат",
    "кэшбэк",
    "списание",
    "зачисление",
]

def parse_operation_line(line: str) -> Optional[Dict]:
    """
    Пытаемся распарсить одну строку операции Kaspi.
    Ожидаем:
      Дата  Сумма(₸/KZT)  [Операция]  [Детали]
    Возвращает dict без bank/source_file или None, если не похоже.
    """
    # 1) дата в начале
    m_date = re.match(r"^(\d{2}\.\d{2}\.\d{2,4})\b", line)
    if not m_date:
        return None

    date_str = m_date.group(1)
    rest = line[m_date.end():].strip()
    date_iso = parse_kaspi_date(date_str)

    if not rest:
        return None

    # 2) сумма до символа валюты
    # Ищем первый блок, в котором есть цифры и потом ₸ или KZT
    m_amt = re.search(r"([+\-]?\s*[\d\s.,]+(?:[-+])?)\s*(₸|KZT)", rest)
    if not m_amt:
        # иногда валюта может потеряться OCR'ом, пробуем без неё
        m_amt = re.search(r"([+\-]?\s*[\d\s.,]+(?:[-+])?)\b", rest)
        if not m_amt:
            return None
        currency = "KZT"
    else:
        currency = "KZT"

    amount_raw_str = m_amt.group(1)
    amount_raw, amount_val = parse_amount_kzt(amount_raw_str)

    after_amt = rest[m_amt.end():].strip() if m_amt else rest[m_amt.end():].strip()

    # 3) тип операции — первое подходящее ключевое слово
    operation = None
    op_pos = -1
    low = after_amt.lower()
    for op in OP_KEYWORDS:
        pos = low.find(op)
        if pos != -1:
            if op_pos == -1 or pos < op_pos:
                op_pos = pos
                operation = op.capitalize()
    details = ""
    if operation is not None and op_pos >= 0:
        details = after_amt[op_pos + len(operation):].strip()
    else:
        details = after_amt.strip()

    merchant = details or None

    return {
        "date": date_iso,
        "amount_raw": f"{amount_raw} {currency}".strip(),
        "amount": amount_val,
        "currency": currency,
        "operation": operation,
        "merchant": merchant,
        "details": details or None,
    }

def is_header_like(line: str) -> bool:
    """Фильтруем очевидные заголовки/мусор Kaspi."""
    l = line.lower()
    if not l.strip():
        return True
    if "kaspi bank" in l or "www.kaspi.kz" in l:
        return True
    if l.startswith("выписка"):
        return True
    if "краткое содержание операций" in l:
        return True
    if "дата" in l and "сумма" in l and "операц" in l:
        return True
    return False

# ---------------- Основной парсер Kaspi ----------------

def parse_kaspi_pdf(pdf_path: str) -> List[Dict]:
    pages = extract_text_pages(pdf_path)
    if not pages:
        raise ValueError("PDF без текста")

    full_text = "\n".join(pages)
    if not is_kaspi_statement(full_text):
        raise ValueError("Не похоже на выписку Kaspi Bank")

    rows: List[Dict] = []
    current: Optional[Dict] = None

    for page in pages:
        for raw_line in page.splitlines():
            line = normalize_ws(raw_line)
            if not line:
                continue

            # пропускаем заголовки/служебные строки
            if is_header_like(line):
                continue

            # пробуем распарсить как новую операцию
            parsed = parse_operation_line(line)
            if parsed:
                # закрываем предыдущую операцию
                if current is not None:
                    rows.append(current)

                # добавляем bank/source_file позже, здесь только "ядро"
                current = parsed
            else:
                # если есть текущая операция — это продолжение деталей
                if current is not None:
                    extra = line.strip()
                    if not is_header_like(extra):
                        if current.get("details"):
                            current["details"] = (str(current["details"]) + " " + extra).strip()
                        else:
                            current["details"] = extra
                        if not current.get("merchant"):
                            # если merchant пустой — возьмём из первых слов описания
                            current["merchant"] = " ".join(extra.split()[:6])
                # иначе игнорируем мусор до первой операции

    # добавляем последнюю операцию
    if current is not None:
        rows.append(current)

    # дополняем общими полями
    for r in rows:
        r["bank"] = "Kaspi Bank"
        r["source_file"] = os.path.basename(pdf_path)

    return rows

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Парсер PDF-выписок Kaspi Bank -> CSV/XLSX"
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Путь к PDF-файлу или папке с PDF-выписками Kaspi."
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Путь к CSV файлу результата."
    )
    ap.add_argument(
        "--xlsx",
        help="(опционально) путь к XLSX файлу."
    )
    ap.add_argument(
        "--dedupe",
        action="store_true",
        help="Удалить дубликаты (date, amount, merchant)."
    )
    ap.add_argument(
        "--prune-empty",
        action="store_true",
        help="Удалить колонки, где все значения пустые."
    )

    args = ap.parse_args()

    # Собираем PDF
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
        raise SystemExit("Укажи существующий PDF или папку с PDF.")

    if not pdf_files:
        raise SystemExit("В указанном пути нет PDF-файлов.")

    all_rows: List[Dict] = []
    log: List[str] = []

    for path in pdf_files:
        try:
            rows = parse_kaspi_pdf(path)
            all_rows.extend(rows)
        except ValueError as e:
            # просто логируем, не падаем (например, это не Kaspi)
            log.append(f"[SKIP] {os.path.basename(path)}: {e}")
        except Exception as e:
            log.append(f"[ERROR] {os.path.basename(path)}: {e}")

    if not all_rows:
        raise SystemExit("Не удалось извлечь операции (формат выписки Kaspi не узнан).")

    df = pd.DataFrame(all_rows)

    # порядок колонок
    base_cols = [
        "date",
        "amount_raw",
        "amount",
        "currency",
        "operation",
        "merchant",
        "details",
        "bank",
        "source_file",
    ]
    for c in base_cols:
        if c not in df.columns:
            df[c] = None
    df = df[base_cols + [c for c in df.columns if c not in base_cols]]

    # дедупликация
    if args.dedupe:
        df = df.drop_duplicates(subset=["date", "amount", "merchant"], keep="first")

    # сортировка по дате
    def _to_dt(x):
        try:
            return datetime.strptime(str(x), "%Y-%m-%d")
        except Exception:
            return pd.NaT

    if "date" in df.columns:
        df["_dt"] = df["date"].apply(_to_dt)
        df = df.sort_values(by=["_dt", "source_file"], kind="stable").drop(columns=["_dt"])

    # удаляем полностью пустые колонки
    if args.prune_empty:
        import numpy as np
        null_like = {"", " ", "null", "<null>", "None", "NaN", "nan"}
        for c in df.columns:
            df[c] = df[c].apply(
                lambda v: np.nan
                if isinstance(v, str) and v.strip() in null_like
                else v
            )
        df = df.dropna(axis=1, how="all")

    # сохраняем
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    if args.xlsx:
        df.to_excel(args.xlsx, index=False)

    # лог
    if log:
        log_path = os.path.splitext(args.out)[0] + "_kaspi_log.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            for line in log:
                f.write(line + "\n")
        print(f"[INFO] Лог: {log_path}")

    print(f"Готово: {args.out} (операций: {len(df)})")
    if args.xlsx:
        print(f"Также создан XLSX: {args.xlsx}")

if __name__ == "__main__":
    main()
