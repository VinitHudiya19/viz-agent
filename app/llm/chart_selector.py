"""
LLM-powered chart selection and Plotly spec generation via Groq API.
Advanced prompts for enterprise-grade, visually stunning charts.

Functions:
    - generate_spec()        -> generates a single premium Plotly JSON spec
    - auto_select_insights() -> asks LLM to pick the best visualizations
    - compute_data_stats()   -> computes statistics to enrich LLM context
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from fastapi import HTTPException
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_community.chat_models import ChatOllama

from app.config import settings
from app.utils.color_palettes import get_palette, get_background

logger = logging.getLogger("viz-agent")

def _get_llm(temperature=0.15):
    """Return a sync Langchain LLM client."""
    provider = getattr(settings, "LLM_PROVIDER", "ollama").lower()
    
    if getattr(settings, "XAI_API_KEY", "") and provider != "ollama":
        provider = "grok"
    elif getattr(settings, "GROQ_API_KEY", "") and provider != "ollama":
        provider = "groq"
        
    if provider == "groq":
        logger.info("Viz Agent LLM: Groq (LangChain)")
        return ChatGroq(api_key=settings.GROQ_API_KEY, model_name=settings.GROQ_MODEL, temperature=temperature)
    elif provider == "grok":
        logger.info("Viz Agent LLM: Grok (LangChain)")
        return ChatOpenAI(api_key=settings.XAI_API_KEY, base_url="https://api.x.ai/v1", model="grok-2-latest", temperature=temperature)
    elif provider == "azure_openai":
        logger.info("Viz Agent LLM: Azure OpenAI (LangChain)")
        return AzureChatOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            temperature=temperature
        )
    elif provider == "openai":
        logger.info("Viz Agent LLM: OpenAI (LangChain)")
        return ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model="gpt-4o",
            temperature=temperature
        )

    else:
        logger.info("Viz Agent LLM: Ollama (LangChain)")
        return ChatOllama(base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_MODEL, temperature=temperature)


# ── Data Statistics Helper ────────────────────────────────────────────────────

def compute_data_stats(columns: list[dict], rows: list[dict]) -> dict:
    """
    Compute summary statistics from data rows to enrich LLM context.
    Returns stats like min, max, mean, top categories, etc.
    """
    stats: dict = {}
    if not rows:
        return stats

    for col in columns:
        name = col["name"]
        sem = col.get("semantic", "other")
        values = [r.get(name) for r in rows if r.get(name) is not None]

        if not values:
            continue

        if sem == "numeric":
            try:
                nums = [float(v) for v in values]
                stats[name] = {
                    "type": "numeric",
                    "min": round(min(nums), 2),
                    "max": round(max(nums), 2),
                    "mean": round(sum(nums) / len(nums), 2),
                    "count": len(nums),
                    "range": round(max(nums) - min(nums), 2),
                }
            except (ValueError, TypeError):
                pass
        elif sem == "categorical":
            from collections import Counter
            counts = Counter(values)
            top_items = counts.most_common(8)
            stats[name] = {
                "type": "categorical",
                "unique_count": len(counts),
                "top_values": [{"value": str(v), "count": c} for v, c in top_items],
            }
        elif sem == "datetime":
            str_vals = sorted(str(v) for v in values)
            stats[name] = {
                "type": "datetime",
                "min": str_vals[0] if str_vals else None,
                "max": str_vals[-1] if str_vals else None,
                "count": len(str_vals),
            }

    return stats


# ── Prompt Templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a world-class data visualization expert who creates award-winning, Bloomberg/McKinsey-quality Plotly charts.
You output ONLY valid JSON — no markdown, no explanation, no extra text.
Every JSON you produce must have exactly two top-level keys: "data" (an array of Plotly traces) and "layout" (a Plotly layout object).
Do NOT wrap the JSON in ```json code fences.

YOUR DESIGN PHILOSOPHY:
- Charts must look like they belong in a Fortune 500 boardroom presentation
- Clean, spacious layouts with clear visual hierarchy
- Subtle gradient fills and professional color usage  
- Rich hover templates with formatted numbers
- Every chart tells a clear data story"""

SPEC_USER_PROMPT = """Generate an ADVANCED, premium-quality Plotly figure specification.

Chart type: {chart_type}
Task: {task}

Column definitions: {columns}
Data statistics: {stats}
Color palette (use these exact hex colors): {palette}
Background config: {bg_config}
Data sample ({n_rows} rows): {data_sample}

ADVANCED STYLING REQUIREMENTS:
1. Use ALL data values from the sample to build complete traces
2. COLORS: Use the provided palette colors exactly. Apply them to traces via marker.color or line.color
3. BACKGROUNDS: Use plot_bgcolor="{plot_bg}", paper_bgcolor="{paper_bg}"
4. GRID: gridcolor="{grid_color}", gridwidth=1, zeroline=false, showgrid=true
5. FONT: family "Inter, -apple-system, sans-serif", title size 18 bold, axis labels size 13, tick size 11
6. TITLE: Clear, insight-driven title (not just "Chart of X"). Add subtitle using <br><sub> tag in smaller font
7. MARGINS: l=70, r=40, t=90, b=70 — spacious and breathing room
8. TEXT COLOR: Use "{text_color}" for all text elements

CHART-SPECIFIC EXCELLENCE:
- BAR: Use rounded corners (marker.line.width=0), add value labels on top using textposition="outside", texttemplate="%{{value:,.0f}}"
- LINE: Use line width=3, mode="lines+markers", marker size=8, fill="tozeroy" with opacity 0.1 for area effect
- PIE: Use hole=0.45 for donut, textinfo="label+percent", pull=[0.02]*n for slight separation, add center annotation
- SCATTER: Use marker size=10-12, add opacity=0.8, include trendline annotation if relevant
- HISTOGRAM: Use bargap=0.05, add mean line as shape, nbinsx=15-20 for smooth distribution
- HEATMAP: Use colorscale based on palette, add text annotations on cells, zmin/zmax from data
- BOX: Use boxmean="sd" to show mean+std, jitter=0.3, pointpos=-1.5
- TREEMAP: Use textinfo="label+value+percent entry"
- WATERFALL: Use increasing/decreasing marker colors from palette
- AREA: Use stackgroup="one", fill with opacity

HOVER TEMPLATE (MANDATORY on every trace):
- Format numbers with thousands separator: %{{y:,.0f}}
- Include axis labels: "<b>%{{x}}</b><br>Value: %{{y:,.0f}}<extra></extra>"
- Use bold for primary values

ANNOTATIONS & SHAPES:
- Add a subtle horizontal reference line at the mean/average if relevant
- For time series: highlight max/min points with annotations
- For bar charts: add a subtle gradient effect using marker.color as an array of slightly varied shades

LEGEND: orientation="h", y=-0.15, x=0.5, xanchor="center" (horizontal below chart)

Output ONLY the raw JSON object."""

CORRECTION_PROMPT = """Your previous response was not valid JSON or was missing required keys.
Return ONLY a JSON object with exactly two keys: "data" (array) and "layout" (object).
Apply professional styling: proper colors, hovertemplates, clean backgrounds.
No markdown, no code fences, no explanation. Pure JSON only."""

AUTO_INSIGHT_SYSTEM = """You are a senior data scientist at McKinsey. Given dataset column profiles, statistics, and a data sample, identify the most IMPACTFUL visualizations that reveal business-critical insights.

Think about:
- What would a CEO/stakeholder want to see?
- What patterns, anomalies, or trends are hidden in this data?
- Which comparisons reveal the most interesting story?

Return ONLY a JSON array of objects. Each object must have:
  - "chart_type": one of "bar", "line", "scatter", "pie", "histogram", "heatmap", "box", "treemap", "waterfall", "area"
  - "task": a specific, insight-driven description (NOT generic like "show X", but "Reveal the revenue concentration across top regions" or "Identify the profit-revenue correlation strength")
  - "x_col": the column name for the x-axis
  - "y_col": the column name for the y-axis (or null for histogram/pie/treemap)
  - "insight_hint": a brief note on what this chart might reveal

Do NOT wrap the output in code fences. Output ONLY the raw JSON array."""

AUTO_INSIGHT_USER = """Dataset columns:
{columns}

Data statistics:
{stats}

Data sample ({n_rows} rows):
{data_sample}

Identify the {n_insights} most IMPACTFUL visualizations for a business stakeholder. 
Requirements:
- Mix chart types for variety (no more than 2 of the same type)
- Prioritize: trends > comparisons > distributions > correlations
- Each chart should tell a different data story
- Use advanced chart types when appropriate (treemap for hierarchies, waterfall for breakdowns, area for cumulative trends)"""


# ── JSON Parsing Helpers ──────────────────────────────────────────────────────

def _strip_markdown_fences(raw: str) -> str:
    """Remove any markdown code-fence wrapping."""
    raw = raw.strip()
    match = re.match(r"^```(?:json)?\s*\n?(.*?)```\s*$", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        inner = parts[1] if len(parts) >= 3 else parts[-1]
        if inner.startswith("json"):
            inner = inner[4:]
        return inner.strip()
    return raw


def _parse_spec(raw: str) -> dict:
    """Parse Plotly spec JSON, validating required keys."""
    cleaned = _strip_markdown_fences(raw)
    spec = json.loads(cleaned)
    if "data" not in spec or "layout" not in spec:
        raise ValueError("Parsed JSON missing 'data' or 'layout' keys")
    return spec


def _parse_insight_array(raw: str) -> list[dict]:
    """Parse the LLM insight array response."""
    cleaned = _strip_markdown_fences(raw)
    arr = json.loads(cleaned)
    if not isinstance(arr, list):
        raise ValueError("Expected a JSON array")
    return arr


# ── Groq API Calls ────────────────────────────────────────────────────────────

def generate_spec(
    chart_type: str,
    task: str,
    columns: list[dict],
    data_sample: list[dict],
    color_scheme: str = "corporate",
) -> dict:
    """
    Generate an advanced, premium-quality Plotly JSON spec via Groq LLM.
    Includes data statistics and background configs for enterprise styling.
    """
    palette = get_palette(color_scheme)
    bg = get_background(color_scheme)
    sample = data_sample[:10]
    stats = compute_data_stats(columns, data_sample)

    user_prompt = SPEC_USER_PROMPT.format(
        chart_type=chart_type,
        task=task,
        columns=json.dumps(columns, indent=1),
        stats=json.dumps(stats, indent=1),
        palette=json.dumps(palette),
        bg_config=json.dumps(bg),
        n_rows=len(sample),
        data_sample=json.dumps(sample, indent=1, default=str),
        plot_bg=bg["plot_bg"],
        paper_bg=bg["paper_bg"],
        grid_color=bg["grid"],
        text_color=bg["text"],
    )

    llm = _get_llm(temperature=0.15)
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]

    # ── First attempt ──
    raw = ""
    try:
        resp = llm.invoke(messages)
        raw = resp.content.strip()
        spec = _parse_spec(raw)
        logger.info("generate_spec OK (chart_type=%s, scheme=%s)", chart_type, color_scheme)
        return spec
    except Exception as first_err:
        logger.warning("generate_spec attempt 1 failed: %s", first_err)

    # ── Retry with correction prompt ──
    try:
        from langchain_core.messages import AIMessage
        messages.append(AIMessage(content=raw))
        messages.append(HumanMessage(content=CORRECTION_PROMPT))

        llm_retry = _get_llm(temperature=0.0)
        resp_retry = llm_retry.invoke(messages)
        raw_retry = resp_retry.content.strip()
        spec = _parse_spec(raw_retry)
        logger.info("generate_spec OK on retry (chart_type=%s)", chart_type)
        return spec
    except Exception as retry_err:
        logger.error("generate_spec FAILED after retry: %s", retry_err)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate valid Plotly spec after 2 attempts. "
                   f"Last error: {retry_err}",
        )


def auto_select_insights(
    columns: list[dict],
    data_sample: list[dict],
    n_insights: int = 5,
) -> list[dict]:
    """
    Ask Groq LLM to pick the most impactful visualizations for the dataset.
    Enriched with data statistics for smarter insight selection.
    """
    sample = data_sample[:10]
    stats = compute_data_stats(columns, data_sample)

    user_prompt = AUTO_INSIGHT_USER.format(
        columns=json.dumps(columns, indent=1),
        stats=json.dumps(stats, indent=1),
        n_rows=len(sample),
        data_sample=json.dumps(sample, indent=1, default=str),
        n_insights=n_insights,
    )

    llm = _get_llm(temperature=0.25)
    messages = [
        SystemMessage(content=AUTO_INSIGHT_SYSTEM),
        HumanMessage(content=user_prompt)
    ]

    try:
        resp = llm.invoke(messages)
        raw = resp.content.strip()
        insights = _parse_insight_array(raw)
        logger.info("auto_select_insights -> %d insights from LLM", len(insights))
        return insights[:n_insights]
    except Exception as exc:
        logger.warning("auto_select_insights LLM call failed: %s", exc)
        return []