"""Render the editable bid-field Draw.io layout as a vector PDF fallback."""

from __future__ import annotations

from pathlib import Path

import reportlab
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


W, H = 900, 230
REG = "SciRelayVeraSans"
BOLD = "SciRelayVeraSans-Bold"


def register_fonts() -> None:
    if REG not in pdfmetrics.getRegisteredFontNames():
        font_dir = Path(reportlab.__file__).resolve().parent / "fonts"
        pdfmetrics.registerFont(TTFont(REG, font_dir / "Vera.ttf"))
        pdfmetrics.registerFont(TTFont(BOLD, font_dir / "VeraBd.ttf"))


def rgb(value: str):
    return colors.HexColor(value)


def cy(y: float) -> float:
    return H - y


def text(c, x, y, w, h, value, size=12, color="#26384D", bold=False, align="center", leading=None):
    c.saveState()
    c.setFont(BOLD if bold else REG, size)
    c.setFillColor(rgb(color))
    lines = value.split("\n")
    leading = leading or size * 1.2
    base = cy(y + h / 2) + (len(lines) - 1) * leading / 2 - size * 0.35
    for i, line in enumerate(lines):
        yy = base - i * leading
        if align == "left":
            c.drawString(x, yy, line)
        else:
            c.drawCentredString(x + w / 2, yy, line)
    c.restoreState()


def segment(c, x, w, fill, stroke):
    c.setLineWidth(1.2)
    c.setFillColor(rgb(fill))
    c.setStrokeColor(rgb(stroke))
    c.rect(x, cy(140), w, 70, stroke=1, fill=1)


def render(output: Path) -> None:
    register_fonts()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Use the embedded body font for ReportLab's initial page preamble too;
    # otherwise it adds an unused, non-embedded Helvetica resource.
    c = canvas.Canvas(str(output), pagesize=(W, H), pageCompression=1, initialFontName=REG)
    text(c, 50, 25, 420, 24, "20-BYTE BID SKETCH", size=17, bold=True, align="left")
    text(c, 290, 25, 200, 24, "(per candidate)", size=12, color="#64748B", align="left")

    segment(c, 50, 320, "#EDF3FF", "#6C86C7")
    segment(c, 370, 80, "#FFF4E5", "#C9822B")
    segment(c, 450, 320, "#ECF9F6", "#218C7C")
    segment(c, 770, 80, "#F3F6F9", "#6B7C8F")

    text(c, 50, 79, 320, 21, "Document fingerprint", size=13, bold=True)
    text(c, 50, 101, 320, 18, "deduplication and payload binding", size=10, color="#64748B")
    text(c, 50, 119, 320, 18, "8 B", size=12, bold=True)

    text(c, 370, 76, 80, 28, "local rank\n+ coverage", size=9.5, leading=11)
    text(c, 370, 113, 80, 18, "2 B", size=11, bold=True)

    text(c, 450, 79, 320, 21, "Query-term bitmap", size=13, bold=True)
    text(c, 450, 101, 320, 18, "coverage of normalized query terms", size=10, color="#64748B")
    text(c, 450, 119, 320, 18, "8 B", size=12, bold=True)

    text(c, 770, 76, 80, 28, "source\n+ check", size=9.5, leading=11)
    text(c, 770, 113, 80, 18, "2 B", size=11, bold=True)

    c.setLineWidth(0.8)
    c.setStrokeColor(rgb("#94A3B8"))
    c.line(50, cy(165), 850, cy(165))
    for x, label, width in [(50, "0 B", 30), (370, "8 B", 30), (450, "10 B", 36), (770, "18 B", 36), (850, "20 B", 40)]:
        text(c, x - width / 2, 170, width, 18, label, size=9, color="#64748B")
    text(c, 50, 195, 600, 20, "Fixed-width control signal; payload requests are counted separately.", size=10, color="#64748B", align="left")

    c.showPage()
    c.save()


if __name__ == "__main__":
    render(Path(__file__).resolve().parents[1] / "paper" / "figures" / "bid_field_layout.pdf")
