"""Dashboard analytics endpoints.

Reads data/pipeline_output/all_combined.csv and returns filtered,
aggregated JSON for the iOS dashboard views.
"""

import os
from datetime import datetime, timedelta

import pandas as pd
from flask import Blueprint, jsonify, request

dashboard_bp = Blueprint("dashboard", __name__, url_prefix="/api/dashboard")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "pipeline_output",
    "all_combined.csv",
)

# Category label mapping (Russian -> English)
CATEGORY_MAP = {
    "переводы": "Transfers",
    "покупки": "Shopping",
    "рестораны и кафе": "Food",
    "маркетплейсы": "Shopping",
    "транспорт": "Transport",
    "супермаркеты": "Food",
    "подписки": "Subscriptions",
}

# Bank name normalisation
BANK_MAP = {
    "Kaspi Bank": "kaspi",
    "Freedom Bank Kazakhstan": "freedom",
}


def _load_df() -> pd.DataFrame:
    """Load and lightly clean the CSV."""
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["category_en"] = df["category"].map(CATEGORY_MAP).fillna("Other")
    df["bank_key"] = df["bank"].map(BANK_MAP).fillna("other")
    return df


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply time / bank / type query-param filters."""
    # --- time filter ---
    period = request.args.get("period")
    start = request.args.get("start")
    end = request.args.get("end")

    if start and end:
        try:
            df = df[
                (df["date"] >= pd.to_datetime(start))
                & (df["date"] <= pd.to_datetime(end))
            ]
        except Exception:
            pass
    elif period:
        now = datetime.now()
        deltas = {"1m": 30, "3m": 90, "6m": 180, "1y": 365}
        days = deltas.get(period)
        if days:
            cutoff = now - timedelta(days=days)
            df = df[df["date"] >= cutoff]

    # --- bank filter ---
    bank = request.args.get("bank")
    if bank and bank in ("kaspi", "freedom"):
        df = df[df["bank_key"] == bank]

    # --- type filter ---
    txn_type = request.args.get("type")
    if txn_type == "transfer":
        df = df[df["operation"].isin(["Transfers", "Transfer"])]
    elif txn_type == "expenses":
        df = df[df["amount"] < 0]

    return df


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@dashboard_bp.route("/overview")
def overview():
    """High-level KPIs for the overview card."""
    df = _apply_filters(_load_df())

    total_spent = float(df[df["amount"] < 0]["amount"].sum())
    total_income = float(df[df["amount"] > 0]["amount"].sum())
    transaction_count = len(df)

    # Top category by absolute spend
    expenses = df[df["amount"] < 0].copy()
    expenses["abs_amount"] = expenses["amount"].abs()
    top_category = None
    if not expenses.empty:
        top_category = (
            expenses.groupby("category_en")["abs_amount"]
            .sum()
            .idxmax()
        )

    # Monthly trend (current vs previous month)
    now = datetime.now()
    this_month = expenses[
        (expenses["date"].dt.year == now.year)
        & (expenses["date"].dt.month == now.month)
    ]["abs_amount"].sum()

    prev = now.replace(day=1) - timedelta(days=1)
    last_month = expenses[
        (expenses["date"].dt.year == prev.year)
        & (expenses["date"].dt.month == prev.month)
    ]["abs_amount"].sum()

    return jsonify(
        {
            "total_spent": round(total_spent, 2),
            "total_income": round(total_income, 2),
            "balance": round(total_income + total_spent, 2),
            "transaction_count": transaction_count,
            "top_category": top_category,
            "this_month_spent": round(float(this_month), 2),
            "last_month_spent": round(float(last_month), 2),
            "trend": "up" if this_month > last_month else "down",
        }
    )


@dashboard_bp.route("/categories")
def categories():
    """Spending breakdown by category."""
    df = _apply_filters(_load_df())
    expenses = df[df["amount"] < 0].copy()
    expenses["abs_amount"] = expenses["amount"].abs()

    if expenses.empty:
        return jsonify({"categories": []})

    grouped = (
        expenses.groupby("category_en")
        .agg(total=("abs_amount", "sum"), count=("abs_amount", "size"))
        .reset_index()
    )
    grand_total = grouped["total"].sum()
    grouped["percentage"] = (grouped["total"] / grand_total * 100).round(1)
    grouped = grouped.sort_values("total", ascending=False)

    result = []
    for _, row in grouped.iterrows():
        result.append(
            {
                "category": row["category_en"],
                "amount": round(float(row["total"]), 2),
                "count": int(row["count"]),
                "percentage": float(row["percentage"]),
            }
        )

    return jsonify({"categories": result, "total": round(float(grand_total), 2)})


@dashboard_bp.route("/banks")
def banks():
    """Spending breakdown by bank."""
    df = _apply_filters(_load_df())
    expenses = df[df["amount"] < 0].copy()
    expenses["abs_amount"] = expenses["amount"].abs()

    if expenses.empty:
        return jsonify({"banks": []})

    grouped = (
        expenses.groupby("bank")
        .agg(
            total=("abs_amount", "sum"),
            count=("abs_amount", "size"),
            avg=("abs_amount", "mean"),
        )
        .reset_index()
    )
    grand_total = grouped["total"].sum()
    grouped["percentage"] = (grouped["total"] / grand_total * 100).round(1)
    grouped = grouped.sort_values("total", ascending=False)

    # Monthly breakdown per bank for side-by-side chart
    expenses["month"] = expenses["date"].dt.strftime("%Y-%m")
    monthly = (
        expenses.groupby(["month", "bank"])["abs_amount"]
        .sum()
        .reset_index()
    )
    monthly_data = {}
    for _, row in monthly.iterrows():
        m = row["month"]
        if m not in monthly_data:
            monthly_data[m] = {}
        monthly_data[m][row["bank"]] = round(float(row["abs_amount"]), 2)

    result = []
    for _, row in grouped.iterrows():
        result.append(
            {
                "bank": row["bank"],
                "amount": round(float(row["total"]), 2),
                "count": int(row["count"]),
                "average": round(float(row["avg"]), 2),
                "percentage": float(row["percentage"]),
            }
        )

    return jsonify(
        {
            "banks": result,
            "total": round(float(grand_total), 2),
            "monthly": [
                {"month": m, **vals}
                for m, vals in sorted(monthly_data.items())
            ],
        }
    )


@dashboard_bp.route("/operations")
def operations():
    """Spending breakdown by operation type."""
    df = _apply_filters(_load_df())

    grouped = (
        df.groupby("operation")
        .agg(
            total=("amount", lambda x: float(x.abs().sum())),
            count=("amount", "size"),
        )
        .reset_index()
    )
    grand_total = grouped["total"].sum()
    grouped["percentage"] = (grouped["total"] / grand_total * 100).round(1)
    grouped = grouped.sort_values("total", ascending=False)

    # Monthly stacked data
    df["month"] = df["date"].dt.strftime("%Y-%m")
    monthly = (
        df.groupby(["month", "operation"])["amount"]
        .apply(lambda x: float(x.abs().sum()))
        .reset_index(name="total")
    )
    monthly_data: dict[str, dict[str, float]] = {}
    for _, row in monthly.iterrows():
        m = row["month"]
        if m not in monthly_data:
            monthly_data[m] = {}
        monthly_data[m][row["operation"]] = round(row["total"], 2)

    result = []
    for _, row in grouped.iterrows():
        result.append(
            {
                "operation": row["operation"],
                "amount": round(float(row["total"]), 2),
                "count": int(row["count"]),
                "percentage": float(row["percentage"]),
            }
        )

    return jsonify(
        {
            "operations": result,
            "total": round(float(grand_total), 2),
            "monthly": [
                {"month": m, **vals}
                for m, vals in sorted(monthly_data.items())
            ],
        }
    )


@dashboard_bp.route("/trend")
def trend():
    """Monthly spending over time for line chart."""
    df = _apply_filters(_load_df())

    expenses = df[df["amount"] < 0].copy()
    expenses["abs_amount"] = expenses["amount"].abs()
    income = df[df["amount"] > 0].copy()

    expenses["month"] = expenses["date"].dt.strftime("%Y-%m")
    income["month"] = income["date"].dt.strftime("%Y-%m")

    exp_monthly = expenses.groupby("month")["abs_amount"].sum()
    inc_monthly = income.groupby("month")["amount"].sum()

    # Merge into a single series
    all_months = sorted(set(exp_monthly.index) | set(inc_monthly.index))
    result = []
    for m in all_months:
        result.append(
            {
                "month": m,
                "expenses": round(float(exp_monthly.get(m, 0)), 2),
                "income": round(float(inc_monthly.get(m, 0)), 2),
            }
        )

    return jsonify({"trend": result})
