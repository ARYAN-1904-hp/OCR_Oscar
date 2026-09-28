"""Convert PPStructure output into normalized layout objects."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class LayoutBlock:
    block_type: str
    bbox: List[int]
    text: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LayoutResult:
    blocks: List[LayoutBlock] = field(default_factory=list)
    tables: List[LayoutBlock] = field(default_factory=list)
    headers: List[LayoutBlock] = field(default_factory=list)
    figures: List[LayoutBlock] = field(default_factory=list)


class LayoutAdapter:
    def from_ppstructure(self, pp_result) -> LayoutResult:
        layout = LayoutResult()
        if pp_result is None:
            return layout

        for region in pp_result.get("raw", []):
            region_type = region.get("type", "text")
            bbox = region.get("bbox", [])
            score = float(region.get("score", 0.0))
            metadata: Dict[str, Any] = {}
            text = ""

            if region_type == "text":
                lines = region.get("res", []) or []
                if isinstance(lines, list):
                    text = "\n".join(
                        (line.get("text", "") if isinstance(line, dict) else str(line))
                        for line in lines
                        if (line.get("text") if isinstance(line, dict) else line)
                    )
                else:
                    text = str(lines or "")
            elif region_type == "table":
                table_res = region.get("res", {}) or {}
                if isinstance(table_res, dict):
                    text = table_res.get("html", "") or ""
                    metadata["cell_bbox"] = table_res.get("cell_bbox", [])
                    metadata["html"] = table_res.get("html", "")
                else:
                    text = str(table_res)
                    metadata["html"] = text
            elif region_type in ("figure", "image", "chart"):
                metadata["image"] = True
                res = region.get("res")
                if isinstance(res, list):
                    text = "\n".join(
                        (x.get("text", "") if isinstance(x, dict) else str(x))
                        for x in res
                        if x
                    )
            elif region_type in ("title", "header"):
                text = region.get("text", "") or ""
                if not text:
                    res = region.get("res") or []
                    if isinstance(res, list):
                        text = "\n".join(
                            (x.get("text", "") if isinstance(x, dict) else str(x))
                            for x in res
                            if x
                        )
                    elif isinstance(res, dict):
                        text = res.get("text", "") or ""

            block = LayoutBlock(
                block_type=region_type,
                bbox=bbox,
                text=text,
                confidence=score,
                metadata=metadata,
            )
            layout.blocks.append(block)
            if region_type == "table":
                layout.tables.append(block)
            elif region_type in ("title", "header"):
                layout.headers.append(block)
            elif region_type in ("figure", "image", "chart"):
                layout.figures.append(block)

        return layout
