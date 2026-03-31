"""
Async PNG renderer using Plotly + Kaleido.
Runs the blocking render in asyncio executor.
Saves PNGs to local disk with UUID filenames.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import uuid
from pathlib import Path

from app.config import settings
from app.utils.storage import storage

logger = logging.getLogger("viz-agent")


async def render_png(
    spec: dict,
    width: int = 900,
    height: int = 500,
) -> tuple[str | None, str | None]:
    """
    Render a Plotly spec to PNG.

    Returns:
        (base64_string, saved_file_path) on success.
        (None, None) if kaleido fails — never crashes the caller.
    """
    loop = asyncio.get_event_loop()

    def _sync_render() -> tuple[str, str]:
        import plotly.graph_objects as go
        import plotly.io as pio

        fig = go.Figure(spec)
        img_bytes: bytes = pio.to_image(fig, format="png", width=width, height=height)

        # Save to Storage
        filename = f"{uuid.uuid4().hex}.png"
        filepath = storage.save_chart(img_bytes, filename)

        b64 = base64.b64encode(img_bytes).decode("ascii")
        logger.info("PNG saved -> %s (%d bytes)", filepath, len(img_bytes))
        return b64, filepath

    try:
        return await loop.run_in_executor(None, _sync_render)
    except Exception as exc:
        logger.warning("PNG render failed: %s", exc, exc_info=True)
        return None, None