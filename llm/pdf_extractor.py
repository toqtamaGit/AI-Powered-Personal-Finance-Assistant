"""PDF text extraction utilities."""

from typing import List


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using pdfplumber or PyPDF2."""
    pages_text: List[str] = []

    # Try pdfplumber first
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages_text.append(text)
        if pages_text:
            return "\n".join(pages_text)
    except Exception:
        pass

    # Fallback to PyPDF2
    try:
        import PyPDF2
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text() or ""
                pages_text.append(text)
        return "\n".join(pages_text)
    except Exception:
        pass

    return ""
