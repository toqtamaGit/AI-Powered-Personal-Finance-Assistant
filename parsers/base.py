"""Abstract base classes for bank statement parsers and normalizers."""

from abc import ABC, abstractmethod
import os
from typing import Dict, List, Set, Tuple

import pandas as pd


class BaseParser(ABC):
    """Base class for bank PDF statement parsers.

    Subclasses implement bank-specific PDF parsing logic.
    Shared utilities (PDF text extraction) live here.
    """

    @property
    @abstractmethod
    def bank_name(self) -> str:
        """Canonical bank name (e.g. 'Kaspi Bank')."""
        ...

    @abstractmethod
    def detect(self, text: str) -> bool:
        """Return True if *text* (full PDF text) belongs to this bank."""
        ...

    @abstractmethod
    def parse(self, pdf_path: str) -> Tuple[List[Dict], List[str]]:
        """Parse a PDF and return ``(transactions, rejected_lines)``.

        Each transaction is a dict with at least:
        ``date, amount_raw, amount, currency, operation, details``.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_text_pages(pdf_path: str) -> List[str]:
        """Extract per-page text from a PDF (pdfplumber → PyPDF2 fallback)."""
        pages: List[str] = []
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    pages.append(page.extract_text() or "")
            if pages:
                return pages
        except Exception:
            pass

        try:
            import PyPDF2
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for p in reader.pages:
                    pages.append(p.extract_text() or "")
            return pages
        except Exception as e:
            raise RuntimeError(f"Cannot extract text from PDF '{pdf_path}': {e}")


class BaseNormalizer(ABC):
    """Base class for bank statement normalizers.

    Subclasses implement ``parse_details`` with bank-specific logic.
    The shared ``normalize_dataframe`` / ``normalize_csv`` workflow
    handles deduplication, column creation, and output.
    """

    @property
    @abstractmethod
    def bank_name(self) -> str:
        """Canonical bank name."""
        ...

    @property
    @abstractmethod
    def business_ops(self) -> Set[str]:
        """Set of operation names that carry business info."""
        ...

    @property
    def detail_fields(self) -> List[str]:
        """Column names produced by ``parse_details``.

        Override to add bank-specific fields (e.g. ``country`` for Freedom).
        Must match the length and order of the tuple returned by ``parse_details``.
        """
        return ['business_type', 'business_name', 'address', 'city']

    @abstractmethod
    def parse_details(self, details: str, operation: str) -> tuple:
        """Extract structured business fields from a transaction detail string.

        Returns a tuple whose length matches ``detail_fields``.
        """
        ...

    # ------------------------------------------------------------------
    # Hooks for bank-specific pre/post-processing
    # ------------------------------------------------------------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hook called after deduplication, before ``parse_details``.

        Override to add bank-specific preprocessing (e.g. hyphen normalization).
        """
        return df

    # ------------------------------------------------------------------
    # Shared workflow
    # ------------------------------------------------------------------

    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate, parse details, and add business columns in-place."""
        # Deduplicate by details
        before = len(df)
        df = df.drop_duplicates(subset=['details'], keep='first').reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            print(f"  Deduplicated: dropped {dropped} rows with duplicate details")

        # Bank-specific preprocessing
        df = self.preprocess(df)

        # Apply parse_details to each row
        parsed = df.apply(
            lambda row: self.parse_details(
                row.get('details', ''),
                row.get('operation', ''),
            ),
            axis=1,
        )

        for i, field in enumerate(self.detail_fields):
            df[field] = [x[i] for x in parsed]

        return df

    def normalize_csv(self, input_file: str, output_file: str) -> pd.DataFrame:
        """Read CSV → normalize → save.  Calls ``normalize_dataframe``."""
        df = pd.read_csv(input_file)
        df = self.normalize_dataframe(df)

        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')

        self.print_summary(df, output_file)
        return df

    def print_summary(self, df: pd.DataFrame, output_file: str) -> None:
        """Print a human-readable summary after normalization."""
        biz = df[df['operation'].isin(self.business_ops)]
        print(f"Done: {output_file}")
        print(f"  Total rows: {len(df)}")
        print(f"  Business ops: {len(biz)}")
        for col in self.detail_fields:
            if col in df.columns:
                print(f"  {col} filled: {biz[col].notna().sum()}")
