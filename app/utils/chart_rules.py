"""
Rule-based chart recommendation engine.
No LLM calls — pure if/else logic on column semantics and task keywords.
Supports advanced chart types: treemap, waterfall, box, area, sunburst.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("viz-agent")


# ── Column Profile Helpers ────────────────────────────────────────────────────

def _count_by_semantic(columns: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {"numeric": 0, "categorical": 0, "datetime": 0, "other": 0}
    for c in columns:
        sem = c.get("semantic", "other")
        counts[sem] = counts.get(sem, 0) + 1
    return counts


def _get_names_by_type(columns: list[dict], sem_type: str) -> list[str]:
    return [c["name"] for c in columns if c.get("semantic") == sem_type]


def _max_cardinality(columns: list[dict], sem_type: str = "categorical") -> int:
    return max(
        ((c.get("unique") or 0) for c in columns if c.get("semantic") == sem_type),
        default=0,
    )


# ── Task Keyword Groups ──────────────────────────────────────────────────────

_DISTRIBUTION_KW = {"distribution", "spread", "histogram", "frequency", "density", "outlier", "variability"}
_PROPORTION_KW   = {"proportion", "share", "percent", "percentage", "composition", "breakdown", "ratio", "mix"}
_CORRELATION_KW  = {"correlat", "relationship", "versus", "vs", "association", "impact", "dependency"}
_TREND_KW        = {"trend", "over time", "growth", "decline", "forecast", "time series", "temporal", "timeline", "progress", "evolution"}
_COMPARISON_KW   = {"compare", "comparison", "rank", "ranking", "top", "bottom", "best", "worst", "benchmark", "performance"}
_HIERARCHY_KW    = {"hierarchy", "tree", "nested", "drill", "breakdown by", "segment by"}
_FLOW_KW         = {"flow", "waterfall", "build-up", "bridge", "contribution", "change from"}
_BOXPLOT_KW      = {"box", "quartile", "median", "interquartile", "range of", "variability of"}
_CUMULATIVE_KW   = {"cumulative", "running total", "accumulated", "stacked area", "fill"}


def _task_matches(task: str, keywords: set[str]) -> bool:
    task_l = task.lower()
    return any(kw in task_l for kw in keywords)


# ── Main Recommender ──────────────────────────────────────────────────────────

def recommend_chart(columns: list[dict], task: str) -> Optional[str]:
    """
    Pure rule-based chart type selection.
    Returns a chart type string, or None if the rules are ambiguous.
    Supports advanced types: treemap, waterfall, box, area.
    """
    counts = _count_by_semantic(columns)
    has_datetime = counts["datetime"] > 0
    has_numeric  = counts["numeric"] > 0
    numeric_count = counts["numeric"]
    cat_names = _get_names_by_type(columns, "categorical")
    cat_card  = _max_cardinality(columns, "categorical")

    # ── Distribution / spread / outliers ──
    if _task_matches(task, _BOXPLOT_KW) and has_numeric:
        return "box"
    if _task_matches(task, _DISTRIBUTION_KW):
        return "histogram"

    # ── Hierarchy and treemap ──
    if _task_matches(task, _HIERARCHY_KW) and len(cat_names) >= 2 and has_numeric:
        return "treemap"

    # ── Waterfall / flow ──
    if _task_matches(task, _FLOW_KW) and has_numeric:
        return "waterfall"

    # ── Proportion / share (donut pie for <=8 slices) ──
    if _task_matches(task, _PROPORTION_KW) and cat_card <= 8:
        return "pie"

    # ── Correlation / relationship ──
    if _task_matches(task, _CORRELATION_KW) and numeric_count >= 2:
        return "heatmap" if numeric_count > 5 else "scatter"

    # ── Cumulative / stacked area ──
    if _task_matches(task, _CUMULATIVE_KW) and has_datetime and has_numeric:
        return "area"

    # ── Time-series trend ──
    if _task_matches(task, _TREND_KW) and has_datetime and has_numeric:
        return "line"

    # ── Datetime + numeric (default to line) ──
    if has_datetime and has_numeric:
        return "line"

    # ── Comparison / ranking ──
    if _task_matches(task, _COMPARISON_KW) and cat_names and has_numeric:
        return "bar"

    # ── Categorical + numeric ──
    if cat_names and has_numeric:
        if cat_card <= 15:
            return "bar"
        return None  # ambiguous → let LLM decide

    # ── Two+ numerics, no categorical ──
    if numeric_count >= 2:
        return "scatter"

    return "bar"


# ── Auto-Insight Suggestion (Advanced) ────────────────────────────────────────

def suggest_best_insights(columns: list[dict], max_insights: int = 5) -> list[dict]:
    """
    Auto-suggest the best visualization tasks from column profiles.
    Now includes advanced chart types for richer insights.
    """
    insights: list[dict] = []
    numerics = _get_names_by_type(columns, "numeric")
    categoricals = _get_names_by_type(columns, "categorical")
    datetimes = _get_names_by_type(columns, "datetime")
    cat_card = _max_cardinality(columns, "categorical")

    # 1. Time-series trend with area fill
    if datetimes and numerics:
        dt_col = datetimes[0]
        insights.append({
            "chart_type": "line",
            "task": f"Reveal the trend and trajectory of {numerics[0]} over {dt_col}",
        })

    # 2. Comparison bar chart
    if categoricals and numerics:
        insights.append({
            "chart_type": "bar",
            "task": f"Compare {numerics[0]} performance across {categoricals[0]} segments",
        })

    # 3. Donut chart for proportions
    if categoricals and numerics and cat_card <= 8:
        insights.append({
            "chart_type": "pie",
            "task": f"Show market share breakdown of {numerics[0]} by {categoricals[0]}",
        })

    # 4. Box plot for distribution analysis
    if numerics and categoricals:
        insights.append({
            "chart_type": "box",
            "task": f"Analyze the distribution and outliers in {numerics[0]} across {categoricals[0]}",
        })
    elif numerics:
        insights.append({
            "chart_type": "histogram",
            "task": f"Examine the distribution pattern of {numerics[0]}",
        })

    # 5. Scatter with trend for correlation
    if len(numerics) >= 2:
        insights.append({
            "chart_type": "scatter",
            "task": f"Identify the relationship strength between {numerics[0]} and {numerics[1]}",
        })

    # 6. Treemap for hierarchical breakdown
    if len(categoricals) >= 2 and numerics:
        insights.append({
            "chart_type": "treemap",
            "task": f"Visualize {numerics[0]} hierarchy across {categoricals[0]} and {categoricals[1]}",
        })

    # 7. Heatmap correlation matrix
    if len(numerics) > 3:
        insights.append({
            "chart_type": "heatmap",
            "task": f"Reveal correlations between {', '.join(numerics[:5])}",
        })

    # 8. Area chart for cumulative trends
    if datetimes and len(numerics) >= 2:
        insights.append({
            "chart_type": "area",
            "task": f"Show cumulative contribution of {numerics[0]} and {numerics[1]} over {datetimes[0]}",
        })

    # Deduplicate by chart_type
    seen_types: set[str] = set()
    unique: list[dict] = []
    for item in insights:
        if item["chart_type"] not in seen_types:
            seen_types.add(item["chart_type"])
            unique.append(item)
        if len(unique) >= max_insights:
            break

    logger.info("suggest_best_insights -> %d insights from %d columns", len(unique), len(columns))
    return unique