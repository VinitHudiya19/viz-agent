"""
Visualization Agent — FastAPI Microservice (Advanced Edition)
All 6 endpoints: /health, /recommend, /chart, /chart/render, /dashboard, /auto-insights
Premium chart generation with 10 color schemes and enterprise-grade styling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings
from app.llm.chart_selector import auto_select_insights, generate_spec
from app.utils.chart_rules import recommend_chart, suggest_best_insights
from app.utils.color_palettes import SCHEME_NAMES
from app.utils.renderer import render_png

logger = logging.getLogger("viz-agent")

# ── Create output dirs ────────────────────────────────────────────────────────
os.makedirs(settings.CHART_OUTPUT_PATH, exist_ok=True)

# ── Color scheme type ─────────────────────────────────────────────────────────
ColorScheme = Literal[
    "corporate", "executive", "vibrant", "neon", "pastel",
    "ocean", "dark", "midnight", "monochrome", "slate"
]

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Visualization Agent",
    description=(
        "Microservice 04 — GEMRSLIZE Platform\n\n"
        "**Advanced visualization engine** that receives aggregated datasets, auto-selects "
        "optimal chart types (rule-based + Groq LLM), and generates premium Plotly specs "
        "with rendered PNGs.\n\n"
        "**10 color schemes**: corporate, executive, vibrant, neon, pastel, ocean, dark, midnight, monochrome, slate\n\n"
        "**Advanced chart types**: bar, line, scatter, pie, histogram, heatmap, box, treemap, waterfall, area"
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Models ───────────────────────────────────────────────────────────


class ColumnProfile(BaseModel):
    name: str = Field(..., description="Column name")
    semantic: Literal["numeric", "categorical", "datetime"] = Field(
        ..., description="Column semantic type"
    )
    unique: Optional[int] = Field(None, description="Number of unique values (for categorical)")


class DataPayload(BaseModel):
    columns: list[ColumnProfile] = Field(..., description="Column definitions")
    rows: list[dict] = Field(..., description="Data rows (max 1000)")


class ChartRequest(BaseModel):
    task: str = Field(..., description="What insight the chart should reveal", examples=["Reveal revenue concentration across top regions"])
    data: DataPayload
    chart_type: Optional[str] = Field(None, description="Override chart type (auto-selected if empty)")
    color_scheme: ColorScheme = Field("corporate", description="Color palette to use")
    render_png: bool = Field(True, description="Whether to render a PNG image")
    width: int = Field(1000, ge=200, le=3000)
    height: int = Field(600, ge=200, le=2000)


class RecommendRequest(BaseModel):
    columns: list[ColumnProfile]
    task: str


class RenderRequest(BaseModel):
    """Existing Plotly spec to render as PNG."""
    data: list = Field(..., description="Plotly traces array")
    layout: dict = Field(..., description="Plotly layout object")
    width: int = Field(1000, ge=200, le=3000)
    height: int = Field(600, ge=200, le=2000)


class DashboardRequest(BaseModel):
    charts: list[ChartRequest] = Field(..., max_length=4, description="Up to 4 chart requests")
    title: str = Field("Data Insights Dashboard", description="Dashboard title")


class AutoInsightRequest(BaseModel):
    data: DataPayload
    color_scheme: ColorScheme = "corporate"
    render_png: bool = Field(True, description="Render PNGs for each chart")
    max_insights: int = Field(5, ge=1, le=8, description="Max number of insights to generate")
    width: int = Field(1000, ge=200, le=3000)
    height: int = Field(600, ge=200, le=2000)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _validate_rows(rows: list[dict]) -> None:
    if len(rows) > settings.MAX_DATA_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many rows. Max {settings.MAX_DATA_ROWS} allowed. Pre-aggregate your data first.",
        )


def _resolve_chart_type(columns: list[dict], task: str, explicit: Optional[str] = None) -> tuple[str, bool]:
    if explicit:
        return explicit, False
    rule_result = recommend_chart(columns, task)
    if rule_result is not None:
        return rule_result, False
    return "bar", False


def _columns_to_dicts(columns: list[ColumnProfile]) -> list[dict]:
    return [c.model_dump() for c in columns]


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", tags=["Health"])
def health():
    """Service health check."""
    provider = str(getattr(settings, "LLM_PROVIDER", "unknown"))
    model = ""
    if provider == "azure_openai":
        model = getattr(settings, "AZURE_OPENAI_DEPLOYMENT_NAME", "")
    elif provider == "groq":
        model = getattr(settings, "GROQ_MODEL", "")
    elif provider == "openai":
        model = "gpt-4o"
    elif provider == "ollama":
        model = getattr(settings, "OLLAMA_MODEL", "")
    
    return {
        "status": "ok",
        "agent": "viz-agent",
        "version": "2.0.0",
        "provider": provider,
        "model": model,
        "port": getattr(settings, "PORT", 8003),
        "storage": getattr(settings, "STORAGE_TYPE", "local"),
        "color_schemes": SCHEME_NAMES,
        "chart_types": ["bar", "line", "scatter", "pie", "histogram", "heatmap", "box", "treemap", "waterfall", "area"],
    }


@app.post("/recommend", tags=["Recommendation"])
def recommend(req: RecommendRequest):
    """Rule-based chart recommendation — no LLM call, instant response."""
    cols = _columns_to_dicts(req.columns)
    chart_type = recommend_chart(cols, req.task)
    return {
        "recommended_chart": chart_type or "bar",
        "used_llm": False,
        "reason": "Rule-based selection from column types and task keywords",
    }


@app.post("/chart", tags=["Chart Generation"])
async def generate_chart(req: ChartRequest):
    """
    Main chart pipeline: data + task -> premium Plotly spec + optional PNG.
    """
    _validate_rows(req.data.rows)

    cols = _columns_to_dicts(req.data.columns)
    chart_type, used_llm = _resolve_chart_type(cols, req.task, req.chart_type)

    logger.info("POST /chart -> chart_type=%s, scheme=%s, task=%s",
                chart_type, req.color_scheme, req.task[:60])

    spec = generate_spec(
        chart_type=chart_type,
        task=req.task,
        columns=cols,
        data_sample=req.data.rows,
        color_scheme=req.color_scheme,
    )

    result: dict = {
        "chart_type": chart_type,
        "used_llm_for_type": used_llm,
        "color_scheme": req.color_scheme,
        "spec": spec,
    }

    if req.render_png:
        png_b64, file_path = await render_png(spec, req.width, req.height)
        result["png_base64"] = png_b64
        result["file_path"] = file_path

    result["html"] = _build_html_snippet(spec)
    return result


@app.post("/chart/render", tags=["Chart Generation"])
async def render_chart(req: RenderRequest):
    """Render an existing Plotly spec to PNG."""
    spec = {"data": req.data, "layout": req.layout}
    png_b64, file_path = await render_png(spec, req.width, req.height)
    if not png_b64:
        raise HTTPException(500, "PNG render failed. Is kaleido installed?")
    return {"png_base64": png_b64, "file_path": file_path}


@app.post("/dashboard", tags=["Dashboard"])
async def generate_dashboard(req: DashboardRequest):
    """Generate a premium multi-chart HTML dashboard with up to 4 charts in parallel."""
    if len(req.charts) > 4:
        raise HTTPException(400, "Maximum 4 charts per dashboard.")

    async def _gen_one(chart_req: ChartRequest) -> dict:
        _validate_rows(chart_req.data.rows)
        cols = _columns_to_dicts(chart_req.data.columns)
        chart_type, _ = _resolve_chart_type(cols, chart_req.task, chart_req.chart_type)
        spec = generate_spec(
            chart_type=chart_type,
            task=chart_req.task,
            columns=cols,
            data_sample=chart_req.data.rows,
            color_scheme=chart_req.color_scheme,
        )
        return {"chart_type": chart_type, "spec": spec, "task": chart_req.task}

    results = await asyncio.gather(*[_gen_one(c) for c in req.charts])
    html = _build_dashboard_html(list(results), req.title)

    return {
        "html": html,
        "chart_count": len(results),
        "charts": [{"chart_type": r["chart_type"], "task": r["task"]} for r in results],
    }


@app.post("/run", tags=["Orchestrator"])
async def run_task(payload: dict):
    """Orchestrator pipeline integration endpoint. Extracts tabular data from upstream context and generates a chart."""
    task_description = payload.get("task_description") or payload.get("query") or "Auto chart"
    context = payload.get("_context", {})
    
    data_rows = None
    column_names = None
    
    # Locate dataset inside context dependencies
    for dep_id, dep_data in context.items():
        if isinstance(dep_data, dict) and "data_preview" in dep_data:
            data_rows = dep_data["data_preview"]
            column_names = dep_data.get("columns", [])
            break
            
    if data_rows is None:
        raise HTTPException(status_code=400, detail="No tabular data found in _context (data_preview missing). Cannot visualize empty data.")
        
    col_profiles = []
    if data_rows:
        if column_names:
            for col in column_names:
                name = col if isinstance(col, str) else col.get("name", "unknown")
                sample_val = data_rows[0].get(name) if data_rows else None
                semantic = "categorical"
                unique_cnt = None
                
                if isinstance(sample_val, (int, float)):
                    semantic = "numeric"
                elif isinstance(sample_val, str) and ("date" in name.lower() or "time" in name.lower()):
                    semantic = "datetime"
                else:
                    unique_cnt = len(set(str(r.get(name)) for r in data_rows if r.get(name) is not None))
                    
                col_profiles.append(ColumnProfile(name=name, semantic=semantic, unique=unique_cnt))
        else:
            for key, val in data_rows[0].items():
                semantic = "numeric" if isinstance(val, (int, float)) else "categorical"
                unique_cnt = len(set(str(r.get(key)) for r in data_rows if r.get(key) is not None)) if semantic == "categorical" else None
                col_profiles.append(ColumnProfile(name=key, semantic=semantic, unique=unique_cnt))
    else:
        # Empty dataframe workaround
        if column_names:
            for col in column_names:
                name = col if isinstance(col, str) else col.get("name", "unknown")
                col_profiles.append(ColumnProfile(name=name, semantic="categorical"))
                
    chart_req = ChartRequest(
        task=task_description,
        data=DataPayload(columns=col_profiles, rows=data_rows),
        color_scheme="vibrant",
        render_png=True,
        width=1000,
        height=600
    )
    
    result = await generate_chart(chart_req)
    return result


@app.post("/auto-insights", tags=["Auto Insights"])
async def auto_insights(req: AutoInsightRequest):
    """
    Smart endpoint: auto-picks the best 4-5 premium visualizations
    that reveal the most impactful insights about the dataset.
    """
    _validate_rows(req.data.rows)

    cols = _columns_to_dicts(req.data.columns)
    logger.info("POST /auto-insights -> %d columns, %d rows, scheme=%s",
                len(cols), len(req.data.rows), req.color_scheme)

    # Step 1: Rule-based suggestions
    rule_insights = suggest_best_insights(cols, max_insights=req.max_insights)

    # Step 2: LLM suggestions if rules didn't produce enough
    if len(rule_insights) < req.max_insights:
        llm_insights = auto_select_insights(cols, req.data.rows, n_insights=req.max_insights)
        existing_types = {i["chart_type"] for i in rule_insights}
        for li in llm_insights:
            if li.get("chart_type") not in existing_types:
                rule_insights.append({
                    "chart_type": li["chart_type"],
                    "task": li["task"],
                })
                existing_types.add(li["chart_type"])
            if len(rule_insights) >= req.max_insights:
                break

    insights_to_gen = rule_insights[:req.max_insights]
    logger.info("Generating %d insight charts", len(insights_to_gen))

    # Step 3: Generate all specs + PNGs in parallel
    async def _gen_insight(insight: dict, index: int) -> dict:
        chart_type = insight["chart_type"]
        task = insight["task"]
        try:
            spec = generate_spec(
                chart_type=chart_type,
                task=task,
                columns=cols,
                data_sample=req.data.rows,
                color_scheme=req.color_scheme,
            )
            result: dict = {
                "index": index,
                "chart_type": chart_type,
                "task": task,
                "spec": spec,
                "html": _build_html_snippet(spec),
                "status": "success",
            }
            if req.render_png:
                png_b64, file_path = await render_png(spec, req.width, req.height)
                result["png_base64"] = png_b64
                result["file_path"] = file_path
            return result
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Insight generation failed for %s: %s", chart_type, exc)
            return {
                "index": index,
                "chart_type": chart_type,
                "task": task,
                "spec": None,
                "status": "failed",
                "error": str(exc),
            }

    charts = await asyncio.gather(*[_gen_insight(i, idx) for idx, i in enumerate(insights_to_gen)])

    successful = [c for c in charts if c["status"] == "success"]

    # Auto-generate dashboard HTML from all successful charts
    dashboard_html = None
    if len(successful) >= 2:
        dashboard_data = [{"chart_type": c["chart_type"], "spec": c["spec"], "task": c["task"]} for c in successful]
        dashboard_html = _build_dashboard_html(dashboard_data, "Auto-Generated Insights Dashboard")

    return {
        "total_requested": len(insights_to_gen),
        "total_generated": len(successful),
        "charts": list(charts),
        "dashboard_html": dashboard_html,
    }


# ── HTML Helpers ──────────────────────────────────────────────────────────────


def _build_html_snippet(spec: dict) -> str:
    """Build a self-contained premium HTML snippet for a single chart."""
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{ margin: 0; font-family: Inter, -apple-system, sans-serif; background: #f8fafc; }}
  #chart {{ width: 100%; height: 100vh; }}
</style>
</head>
<body>
<div id="chart"></div>
<script>
  var spec = {json.dumps(spec)};
  Plotly.newPlot('chart', spec.data, spec.layout, {{
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    toImageButtonOptions: {{ format: 'png', filename: 'chart', height: 600, width: 1000, scale: 2 }}
  }});
</script>
</body>
</html>"""


def _build_dashboard_html(charts: list[dict], title: str = "Dashboard") -> str:
    """Build a premium multi-chart HTML dashboard with modern styling."""
    n = len(charts)
    cols = 2 if n > 1 else 1

    chart_cards = ""
    chart_scripts = ""
    for i, c in enumerate(charts):
        cid = f"chart-{i}"
        task = c.get("task", c.get("chart_type", f"Chart {i+1}"))
        chart_type = c.get("chart_type", "chart")

        chart_cards += f"""
        <div class="chart-card">
          <div class="card-header">
            <span class="chart-badge">{chart_type.upper()}</span>
            <h3>{task}</h3>
          </div>
          <div id="{cid}" class="chart-container"></div>
        </div>"""

        chart_scripts += f"""
  Plotly.newPlot('{cid}',
    {json.dumps(c['spec'].get('data', []))},
    Object.assign({{}}, {json.dumps(c['spec'].get('layout', {}))}, {{autosize: true, height: 380}}),
    {{responsive: true, displayModeBar: false}}
  );"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', -apple-system, sans-serif;
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
    min-height: 100vh;
    color: #E2E8F0;
    padding: 32px;
  }}
  .dashboard-header {{
    text-align: center;
    margin-bottom: 32px;
    padding: 24px;
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
  }}
  .dashboard-header h1 {{
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(135deg, #60A5FA, #A78BFA, #F472B6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
  }}
  .dashboard-header p {{
    color: #94A3B8;
    font-size: 14px;
  }}
  .chart-grid {{
    display: grid;
    grid-template-columns: repeat({cols}, 1fr);
    gap: 24px;
    max-width: 1400px;
    margin: 0 auto;
  }}
  .chart-card {{
    background: rgba(30, 41, 59, 0.8);
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
    backdrop-filter: blur(10px);
  }}
  .chart-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    border-color: rgba(96, 165, 250, 0.3);
  }}
  .card-header {{
    padding: 16px 20px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
  }}
  .card-header h3 {{
    font-size: 13px;
    font-weight: 500;
    color: #CBD5E1;
    margin-top: 6px;
    line-height: 1.4;
  }}
  .chart-badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
    background: linear-gradient(135deg, #3B82F6, #8B5CF6);
    color: white;
  }}
  .chart-container {{
    padding: 12px;
    min-height: 380px;
  }}
  @media (max-width: 768px) {{
    .chart-grid {{ grid-template-columns: 1fr; }}
    body {{ padding: 16px; }}
  }}
</style>
</head>
<body>
<div class="dashboard-header">
  <h1>{title}</h1>
  <p>{n} visualizations &bull; Auto-generated by Visualization Agent v2.0</p>
</div>
<div class="chart-grid">{chart_cards}
</div>
<script>{chart_scripts}
</script>
</body>
</html>"""