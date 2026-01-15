from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Optional

_EXPORTED: set[str] = set()
logger = logging.getLogger(__name__)


def export_graph(
    graph: Any,
    *,
    name: str,
    output_dir: Optional[str] = None,
    print_mermaid: bool = True,
) -> None:
    """
    Export graph visualization to Mermaid and PNG formats.

    Args:
        graph: The graph object to export
        name: Name for the output files
        output_dir: Directory to save files (defaults to PROJECT_ROOT)
        print_mermaid: Whether to print mermaid syntax to console

    Environment Variables:
        TRADEREACT_EXPORT_GRAPHS: Set to "1", "true", or "True" to enable export
    """
    # Check if export is enabled
    if os.getenv("TRADEREACT_EXPORT_GRAPHS", "").strip() not in {"1", "true", "True"}:
        return

    # Avoid duplicate exports
    if name in _EXPORTED:
        return
    _EXPORTED.add(name)

    # Determine output directory
    if output_dir:
        out_dir = Path(output_dir)
    else:
        # Use project root instead of current working directory
        try:
            from tradereact.default_config import DEFAULT_CONFIG
            project_dir = Path(DEFAULT_CONFIG.get("project_dir", os.getcwd()))
            out_dir = project_dir / "graphs"
        except Exception:
            out_dir = Path(os.getcwd()) / "graphs"

    # Create output directory
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory {out_dir}: {e}")
        return

    # Export Mermaid diagram
    try:
        mermaid = graph.get_graph().draw_mermaid()
        if print_mermaid:
            print(f"\n[Graph Export] Mermaid diagram for '{name}':")
            print(mermaid)

        mermaid_file = out_dir / f"{name}.mmd"
        mermaid_file.write_text(mermaid, encoding="utf-8")
        logger.info(f"Exported Mermaid diagram to: {mermaid_file}")
    except Exception as e:
        logger.error(f"Failed to export Mermaid diagram for '{name}': {e}")

    # Export PNG image
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        png_file = out_dir / f"{name}.png"
        png_file.write_bytes(png_bytes)
        logger.info(f"Exported PNG image to: {png_file}")
    except Exception as e:
        logger.warning(f"Failed to export PNG image for '{name}': {e}")
        # PNG export often fails due to missing dependencies, so this is just a warning

