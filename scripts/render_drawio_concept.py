"""Render the SciRelay concept diagram as a compact vector PDF.

The figure uses two visibly separate channels: a dashed control channel for
fixed-width, score-free bids and a solid payload channel for selected full
abstracts. The small line icons are drawn as vector paths so the PDF stays
sharp at any paper scale.
"""

from __future__ import annotations

import math
from pathlib import Path

import reportlab
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


W, H = 1500, 440
REGULAR_FONT = "SciRelayVeraSans"
BOLD_FONT = "SciRelayVeraSans-Bold"


def register_fonts() -> None:
    if REGULAR_FONT not in pdfmetrics.getRegisteredFontNames():
        font_dir = Path(reportlab.__file__).resolve().parent / "fonts"
        pdfmetrics.registerFont(TTFont(REGULAR_FONT, font_dir / "Vera.ttf"))
        pdfmetrics.registerFont(TTFont(BOLD_FONT, font_dir / "VeraBd.ttf"))


def rgb(value: str):
    return colors.HexColor(value)


def cy(y: float) -> float:
    return H - y


def rect(c, x, y, w, h, fill, stroke, radius=10, width=1.6):
    c.setLineWidth(width)
    c.setStrokeColor(rgb(stroke))
    c.setFillColor(rgb(fill))
    c.roundRect(x, cy(y + h), w, h, radius, stroke=1, fill=1)


def flat_rect(c, x, y, w, h, fill):
    c.setFillColor(rgb(fill))
    c.rect(x, cy(y + h), w, h, stroke=0, fill=1)


def line(c, x1, y1, x2, y2, color="#486278", width=1.8, dashed=False):
    c.saveState()
    c.setStrokeColor(rgb(color))
    c.setLineWidth(width)
    c.setLineCap(1)
    c.setLineJoin(1)
    c.setDash(6, 5) if dashed else c.setDash()
    c.line(x1, cy(y1), x2, cy(y2))
    c.restoreState()


def arrowhead(c, x, y, dx, dy, size=11, color="#486278", filled=True):
    vx, vy = dx, -dy
    norm = math.hypot(vx, vy) or 1.0
    ux, uy = vx / norm, vy / norm
    px, py = -uy, ux
    tip = (x, cy(y))
    left = (x - ux * size + px * size * 0.42, cy(y) - uy * size + py * size * 0.42)
    right = (x - ux * size - px * size * 0.42, cy(y) - uy * size - py * size * 0.42)
    c.saveState()
    c.setStrokeColor(rgb(color))
    c.setFillColor(rgb(color))
    c.setLineWidth(1.5)
    p = c.beginPath()
    p.moveTo(*tip)
    p.lineTo(*left)
    p.lineTo(*right)
    p.close()
    c.drawPath(p, stroke=0 if filled else 1, fill=1 if filled else 0)
    c.restoreState()


def arrow(c, x1, y1, x2, y2, color="#486278", width=1.8, dashed=False, filled=True):
    line(c, x1, y1, x2, y2, color=color, width=width, dashed=dashed)
    arrowhead(c, x2, y2, x2 - x1, y2 - y1, size=11 if dashed else 13, color=color, filled=filled)


def poly_arrow(c, points, color="#486278", width=1.8, dashed=False, filled=True):
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        line(c, x1, y1, x2, y2, color=color, width=width, dashed=dashed)
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    arrowhead(c, x2, y2, x2 - x1, y2 - y1, size=11 if dashed else 13, color=color, filled=filled)


def text(c, x, y, w, h, value, size=15, color="#1F2937", bold=False, align="center", leading=None):
    c.saveState()
    c.setFillColor(rgb(color))
    c.setFont(BOLD_FONT if bold else REGULAR_FONT, size)
    lines = value.split("\n")
    leading = leading or size * 1.18
    base = cy(y + h / 2) + (len(lines) - 1) * leading / 2 - size * 0.35
    for i, item in enumerate(lines):
        yy = base - i * leading
        if align == "left":
            c.drawString(x, yy, item)
        elif align == "right":
            c.drawRightString(x + w, yy, item)
        else:
            c.drawCentredString(x + w / 2, yy, item)
    c.restoreState()


def icon(c, kind, x, y, color="#315A78", scale=1.0):
    """Draw a compact, consistent line icon centered at (x, y)."""
    c.saveState()
    c.setStrokeColor(rgb(color))
    c.setLineWidth(1.8 * scale)
    c.setLineCap(1)
    c.setLineJoin(1)
    if kind == "search":
        c.circle(x - 2 * scale, cy(y - 2 * scale), 7 * scale, stroke=1, fill=0)
        c.line(x + 3 * scale, cy(y + 3 * scale), x + 10 * scale, cy(y + 10 * scale))
    elif kind == "doc":
        c.roundRect(x - 9 * scale, cy(y + 11 * scale), 18 * scale, 22 * scale, 2 * scale, stroke=1, fill=0)
        c.line(x - 5 * scale, cy(y - 1 * scale), x + 5 * scale, cy(y - 1 * scale))
        c.line(x - 5 * scale, cy(y + 4 * scale), x + 5 * scale, cy(y + 4 * scale))
    elif kind == "vector":
        c.circle(x - 7 * scale, cy(y - 5 * scale), 2 * scale, stroke=1, fill=0)
        c.circle(x + 7 * scale, cy(y - 5 * scale), 2 * scale, stroke=1, fill=0)
        c.circle(x, cy(y + 8 * scale), 2 * scale, stroke=1, fill=0)
        c.line(x - 5 * scale, cy(y - 4 * scale), x - 1 * scale, cy(y + 7 * scale))
        c.line(x + 5 * scale, cy(y - 4 * scale), x + 1 * scale, cy(y + 7 * scale))
    elif kind == "alert":
        p = c.beginPath()
        p.moveTo(x, cy(y - 10 * scale))
        p.lineTo(x + 10 * scale, cy(y + 9 * scale))
        p.lineTo(x - 10 * scale, cy(y + 9 * scale))
        p.close()
        c.drawPath(p, stroke=1, fill=0)
        c.line(x, cy(y - 4 * scale), x, cy(y + 3 * scale))
        c.circle(x, cy(y + 7 * scale), 0.8 * scale, stroke=1, fill=1)
    elif kind == "bid":
        c.roundRect(x - 10 * scale, cy(y + 8 * scale), 20 * scale, 16 * scale, 2 * scale, stroke=1, fill=0)
        c.line(x - 6 * scale, cy(y - 1 * scale), x + 6 * scale, cy(y - 1 * scale))
        c.line(x - 6 * scale, cy(y + 4 * scale), x + 3 * scale, cy(y + 4 * scale))
    elif kind == "rank":
        for i, height in enumerate((7, 12, 17)):
            xx = x - 10 * scale + i * 9 * scale
            c.rect(xx, cy(y + height * scale), 5 * scale, height * scale, stroke=1, fill=0)
    elif kind == "filter":
        p = c.beginPath()
        p.moveTo(x - 10 * scale, cy(y - 8 * scale))
        p.lineTo(x + 10 * scale, cy(y - 8 * scale))
        p.lineTo(x + 3 * scale, cy(y))
        p.lineTo(x + 3 * scale, cy(y + 9 * scale))
        p.lineTo(x - 3 * scale, cy(y + 9 * scale))
        p.lineTo(x - 3 * scale, cy(y))
        p.close()
        c.drawPath(p, stroke=1, fill=0)
    elif kind == "gate":
        c.line(x - 10 * scale, cy(y - 9 * scale), x - 10 * scale, cy(y + 9 * scale))
        c.line(x + 10 * scale, cy(y - 9 * scale), x + 10 * scale, cy(y + 9 * scale))
        c.line(x - 10 * scale, cy(y - 8 * scale), x + 10 * scale, cy(y - 8 * scale))
        c.line(x - 10 * scale, cy(y + 8 * scale), x + 10 * scale, cy(y + 8 * scale))
        c.line(x - 5 * scale, cy(y - 6 * scale), x + 5 * scale, cy(y + 6 * scale))
        c.line(x + 5 * scale, cy(y - 6 * scale), x - 5 * scale, cy(y + 6 * scale))
    elif kind == "brain":
        c.circle(x - 5 * scale, cy(y), 6 * scale, stroke=1, fill=0)
        c.circle(x + 5 * scale, cy(y), 6 * scale, stroke=1, fill=0)
        c.line(x, cy(y - 7 * scale), x, cy(y + 7 * scale))
        c.line(x - 8 * scale, cy(y - 1 * scale), x - 2 * scale, cy(y - 1 * scale))
        c.line(x + 2 * scale, cy(y + 3 * scale), x + 8 * scale, cy(y + 3 * scale))
    c.restoreState()


def node(c, x, y, w, h, fill, stroke, kind, title, subtitle, title_size=14, subtitle_size=10):
    rect(c, x, y, w, h, fill, stroke, radius=9, width=1.7)
    icon(c, kind, x + 22, y + h / 2, stroke, scale=0.9)
    text(c, x + 40, y + 7, w - 46, h * 0.48, title, size=title_size, bold=True, align="left")
    text(c, x + 40, y + h * 0.49 - 1, w - 46, h * 0.35, subtitle, size=subtitle_size, color="#64748B", align="left")


def render(output: Path) -> None:
    register_fonts()
    output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output), pagesize=(W, H), pageCompression=1, initialFontName=REGULAR_FONT, initialFontSize=12)

    # Three stage panels, with a shared grid and generous whitespace.
    rect(c, 112, 38, 506, 292, "#F7FAFF", "#B7C9EE", radius=16, width=1.5)
    rect(c, 648, 38, 520, 292, "#FFFBF5", "#EBCB97", radius=16, width=1.5)
    rect(c, 1198, 38, 278, 292, "#F7FAFC", "#C9D3DE", radius=16, width=1.5)
    flat_rect(c, 136, 76, 458, 3, "#5F83D8")
    flat_rect(c, 672, 76, 472, 3, "#D48A24")
    flat_rect(c, 1222, 76, 230, 3, "#7A8A9A")

    text(c, 136, 48, 205, 22, "LOCAL RETRIEVAL", size=17, color="#334155", bold=True, align="left")
    text(c, 412, 48, 182, 22, "CONTROL CHANNEL", size=12, color="#6880A3", bold=True, align="right")
    text(c, 672, 48, 210, 22, "CONTROL DECISION", size=17, color="#334155", bold=True, align="left")
    text(c, 940, 48, 204, 22, "fixed-width · score-free", size=12, color="#A36A22", bold=True, align="right")
    text(c, 1222, 48, 230, 22, "PAYLOAD / REASONING", size=16, color="#334155", bold=True, align="left")

    # Query fan-out.
    c.setLineWidth(2.0)
    c.setStrokeColor(rgb("#334155"))
    c.setFillColor(rgb("#FFFFFF"))
    c.circle(53, cy(188), 31, stroke=1, fill=1)
    text(c, 22, 174, 62, 30, "q", size=26, bold=True)
    text(c, 0, 126, 106, 30, "scientific\nquery", size=12, color="#64748B", leading=14)
    arrow(c, 84, 188, 112, 188, color="#8998A8", width=1.7, filled=False)
    c.setFillColor(rgb("#E7EEF9")); c.setStrokeColor(rgb("#6688D6")); c.setLineWidth(1.5)
    c.circle(126, cy(188), 9, stroke=1, fill=1)
    for endpoint in (178, 222, 266):
        arrow(c, 135, 188, 170, endpoint, color="#8998A8", width=1.5, filled=False)

    # Local agents and the compact bid sketch channel.
    node(c, 170, 156, 248, 44, "#EAF1FF", "#6688D6", "search", "Agent 1 · lexical", "local shard D1", title_size=13, subtitle_size=10)
    node(c, 170, 200, 248, 44, "#EAF1FF", "#6688D6", "vector", "Agent 2 · dense", "local shard D2", title_size=13, subtitle_size=10)
    node(c, 170, 244, 248, 44, "#F0F3F8", "#7B8B9D", "alert", "Agent 3 · score fault", "local shard D3", title_size=13, subtitle_size=10)
    rect(c, 468, 145, 106, 120, "#E8F8F4", "#159A86", radius=10, width=1.8)
    icon(c, "bid", 492, 174, "#159A86", scale=0.85)
    text(c, 510, 157, 54, 38, "BID\nSKETCH", size=13, color="#0D7365", bold=True, align="left", leading=15)
    text(c, 478, 214, 87, 20, "20-byte", size=11, color="#52706D")
    text(c, 478, 234, 87, 18, "fixed width", size=10, color="#64748B")
    for ys, yt in ((178, 164), (222, 205), (266, 246)):
        arrow(c, 418, ys, 468, yt, color="#486278", width=1.7, dashed=True)
    rect(c, 420, 294, 154, 28, "#F1F4F7", "#7B8B9D", radius=8, width=1.4)
    icon(c, "doc", 438, 308, "#7B8B9D", scale=0.55)
    text(c, 451, 296, 116, 24, "selected local docs", size=10, color="#475569", bold=True, align="left")

    # Control decision path.
    node(c, 672, 142, 185, 64, "#FFF1DB", "#D48A24", "rank", "Rank calibration", "score-free signal", title_size=13, subtitle_size=10)
    node(c, 886, 142, 218, 64, "#FFF1DB", "#D48A24", "filter", "Greedy selection", "rank + coverage", title_size=13, subtitle_size=10)
    arrow(c, 574, 184, 672, 174, color="#486278", width=2.0, dashed=True)
    arrow(c, 857, 174, 886, 174, color="#486278", width=2.0, dashed=True)
    arrow(c, 995, 206, 995, 238, color="#486278", width=2.0, dashed=True)
    text(c, 712, 108, 240, 18, "no raw-score comparison", size=10, color="#64748B", align="left")

    # Selection requests are control messages; only the selected-doc node emits payload.
    text(c, 670, 260, 215, 18, "PAYLOAD CHANNEL", size=11, color="#5C6B7B", bold=True, align="left")
    text(c, 670, 278, 230, 18, "selected full abstracts only", size=10, color="#64748B", align="left")
    rect(c, 896, 236, 208, 70, "#EDF8EE", "#4F9C57", radius=10, width=1.8)
    icon(c, "gate", 923, 271, "#4F9C57", scale=0.9)
    text(c, 942, 245, 150, 26, "Payload gate", size=14, bold=True, align="left")
    text(c, 942, 272, 150, 20, "at most B abstracts", size=11, color="#64748B", align="left")
    rect(c, 1218, 236, 106, 70, "#EDF8EE", "#4F9C57", radius=10, width=1.8)
    icon(c, "doc", 1242, 271, "#4F9C57", scale=0.85)
    text(c, 1257, 245, 59, 26, "Selected", size=13, bold=True, align="left")
    text(c, 1257, 272, 59, 20, "abstracts", size=10, color="#64748B", align="left")
    rect(c, 1350, 236, 106, 70, "#F1F4F7", "#7A8A9A", radius=10, width=1.8)
    icon(c, "brain", 1374, 271, "#7A8A9A", scale=0.8)
    text(c, 1389, 245, 59, 26, "Downstream", size=11, bold=True, align="left")
    text(c, 1389, 272, 59, 20, "reasoning", size=10, color="#64748B", align="left")
    poly_arrow(c, [(995, 206), (995, 220), (600, 220), (600, 308), (574, 308)], color="#486278", width=1.8, dashed=True)
    arrow(c, 574, 308, 896, 308, color="#26384D", width=2.6)
    arrow(c, 896, 308, 896, 271, color="#26384D", width=2.6)
    arrow(c, 1104, 271, 1218, 271, color="#26384D", width=2.6)
    arrow(c, 1324, 271, 1350, 271, color="#26384D", width=2.6)
    text(c, 470, 326, 150, 18, "full abstracts only", size=10, color="#475569", align="left")

    # Legend.
    arrow(c, 560, 382, 612, 382, color="#486278", width=2.0, dashed=True)
    text(c, 622, 371, 190, 24, "control: bid sketch", size=11, color="#64748B", align="left")
    arrow(c, 875, 382, 927, 382, color="#26384D", width=2.6)
    text(c, 937, 371, 230, 24, "payload: selected abstract", size=11, color="#64748B", align="left")
    text(c, 1218, 371, 238, 24, "B = document-count budget", size=11, color="#64748B", align="right")

    c.showPage()
    c.save()


if __name__ == "__main__":
    render(Path(__file__).resolve().parents[1] / "paper" / "figures" / "scirelay_concept.pdf")
