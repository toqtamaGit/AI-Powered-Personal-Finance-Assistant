"""Bank detection from PDF text content."""

from typing import Optional


def detect_bank_from_text(text: str) -> Optional[str]:
    """Detect bank name from PDF text content."""
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

    # Kaspi Bank
    if (
        "kaspi bank" in t
        or "каспи банк" in t
        or "kaspi.kz" in t
        or "ao kaspi bank" in t
        or 'ао "kaspi bank"' in t
        or "kaspi gold" in t
        or "kaspi red" in t
    ):
        return "Kaspi Bank"

    # Halyk Bank
    if (
        "halyk bank" in t
        or "халык банк" in t
        or "народный банк" in t
        or "halykbank.kz" in t
    ):
        return "Halyk Bank"

    # Jusan Bank
    if (
        "jusan bank" in t
        or "жусан банк" in t
        or "jusan.kz" in t
    ):
        return "Jusan Bank"

    return None
