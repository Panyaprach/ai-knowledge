"""PDF -> (title, per-page lines, figures). Figures are cropped at native resolution, not whole pages."""
import re
from dataclasses import dataclass
from .config import DIAGRAM_MIN_DRAWINGS

MAX_SIDE_PX = 2000


@dataclass
class Figure:
    page: int
    png: bytes
    caption: str
    source: str  # 'embedded' | 'vector'


def _render(page, rect, dpi: float) -> bytes:
    dpi = int(max(72, min(dpi, MAX_SIDE_PX / max(rect.width, rect.height) * 72)))
    return page.get_pixmap(clip=rect, dpi=dpi).tobytes("png")


def _caption(text: str) -> str:
    m = re.search(r"^Figure\s+\d+[^\n]*", text, re.M)
    return m.group(0).strip() if m else ""


def find_figures(page, page_no: int, text: str) -> list[Figure]:
    import pymupdf
    page_area = page.rect.width * page.rect.height
    found = []

    # 1) embedded raster images (PNG/JPEG screenshots of diagrams): crop at native resolution
    for info in page.get_images(full=True):
        w, h = info[2], info[3]
        if w < 300 or h < 200:
            continue
        try:
            rect = page.get_image_bbox(info)
        except Exception:
            continue
        if rect.is_empty or rect.width * rect.height < 0.04 * page_area:
            continue
        native_ppi = w / (rect.width / 72)
        dpi = min(300, max(150, native_ppi * 1.5))  # a little upscaling helps small diagram text
        found.append((rect.width * rect.height, rect, dpi, "embedded"))

    # 2) vector-drawn diagrams: largest dense cluster of drawing paths
    try:
        drawings = page.get_drawings()
        if len(drawings) >= DIAGRAM_MIN_DRAWINGS and hasattr(page, "cluster_drawings"):
            for r in page.cluster_drawings(drawings=drawings):
                if r.width * r.height >= 0.10 * page_area:
                    r = pymupdf.Rect(r) + (-8, -8, 8, 8)
                    found.append((r.width * r.height, r & page.rect, 200, "vector"))
    except Exception:
        pass

    found.sort(key=lambda x: -x[0])
    cap = _caption(text)
    return [Figure(page_no, _render(page, rect, dpi), cap, src) for _, rect, dpi, src in found[:3]]


def parse_pdf(path: str, vision: bool = True):
    import pymupdf
    pages, figures = [], []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            pages.append((i, text.splitlines()))
            if vision:
                figures += find_figures(page, i, text)
    first = next((l.strip() for l in pages[0][1] if l.strip()), "") if pages else ""
    return first or path, pages, figures
