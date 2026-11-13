"""Style biểu đồ dùng chung cho các câu b-e, theo quy ước của repo figures4papers.

Quy ước lấy từ ChenLiu-1996/figures4papers (scientific-figure-making/references/api.md, design-theory.md):
  bỏ spine trên/phải, legend không khung, font Helvetica/Arial/DejaVu Sans, nét 2-3pt, lưới tối giản,
  tight_layout(pad=2), xuất 300 DPI, chữ dạng vector (svg.fonttype='none'), dùng hatch để an toàn khi in đen trắng.
Sai khác có chủ đích: màu được kiểm tra bằng validate_palette.js (skill dataviz); bảng màu gốc của repo trượt
kiểm tra (xanh #0F4D92 quá tối, #42949E quá xám) nên dùng #3775BA và #2E9C8A trong cùng họ màu, xem docs/figure_style.md.
Cỡ chữ 14, nét 1,8 và khổ hình thu 0,5 lần (hình rộng khoảng 8-11 inch) để chữ vẫn đọc được khi chèn vào trang báo cáo; chữ nhỏ hơn 11 pt được nâng lên 11 pt.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, NullFormatter

# Thứ tự cố định đã qua validate_palette.js (light, surface #ffffff, cặp liền kề): blue, red, teal, violet
PALETTE = {"blue": "#3775BA", "red": "#B64342", "teal": "#2E9C8A", "violet": "#9A4D8E"}
SERIES = [PALETTE["blue"], PALETTE["red"], PALETTE["teal"], PALETTE["violet"]]
STAGE_COLORS = dict(zip(["D0 clean", "D1 +noise", "D2 +outliers", "D3 +missing"], SERIES))
STAGE_MARKERS = dict(zip(STAGE_COLORS, ["o", "s", "^", "D"]))
STAGE_LINESTYLES = dict(zip(STAGE_COLORS, ["-", "--", "-.", ":"]))
HATCHES = ["", "//", "..", "xx"]
NEUTRAL = {"ink": "#272727", "mid": "#8C8C8C", "light": "#CFCECE", "faint": "#EDEDED"}
BLUE_FAMILY = {"dark": "#0F4D92", "main": "#3775BA", "light": "#BBD3EC", "faint": "#EAF1FA"}

CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", [BLUE_FAMILY["faint"], BLUE_FAMILY["main"], BLUE_FAMILY["dark"]])
CMAP_DIV = LinearSegmentedColormap.from_list("div_bluered", [PALETTE["blue"], "#F4F4F2", PALETTE["red"]])

FONT_SIZE = 14
AXES_LW = 1.8
FIG_SCALE = 0.5   # mọi figsize được nhân hệ số này: hình rộng ~11 inch khi chèn vào trang báo cáo 6,4 inch chỉ thu ~0,57 lần
MIN_TEXT = 11     # mọi chữ trong hình (nhãn, tick, chú thích) tối thiểu cỡ này trước khi thu vào báo cáo
_PATCHED = False


def _scaled(fn):
    def wrapper(*args, **kwargs):
        if "figsize" in kwargs and kwargs["figsize"] is not None:
            kwargs["figsize"] = tuple(v * FIG_SCALE for v in kwargs["figsize"])
        return fn(*args, **kwargs)
    return wrapper


def apply_publication_style(font_size: int = FONT_SIZE, axes_lw: float = AXES_LW) -> None:
    global _PATCHED
    if not _PATCHED:  # thu khổ hình một lần để chữ còn đọc được sau khi chèn vào trang A4
        plt.figure = _scaled(plt.figure)  # plt.subplots gọi plt.figure nên chỉ cần bọc một chỗ (bọc cả hai sẽ nhân hệ số hai lần)
        _PATCHED = True
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": font_size, "axes.titlesize": font_size, "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1, "ytick.labelsize": font_size - 1, "legend.fontsize": font_size - 1,
        "axes.linewidth": axes_lw, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": axes_lw, "ytick.major.width": axes_lw,
        "lines.linewidth": 2.0, "lines.markersize": 7,
        "legend.frameon": False, "axes.grid": False,
        "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def panel_label(ax, letter: str) -> None:
    ax.text(-0.02, 1.08, f"({letter})", transform=ax.transAxes, fontsize=FONT_SIZE + 2, fontweight="bold",
            va="bottom", ha="right")


def finalize_figure(fig, out_dir: Path, name: str, formats=("png", "pdf"), dpi: int = 300, pad: float = 2.0) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:  # nhãn thang log dạng số thường (10, 100) thay vì 2x10^1 chồng chéo
        for axis in (ax.xaxis, ax.yaxis):
            if axis.get_scale() == "log":
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                axis.set_minor_formatter(NullFormatter())
    for txt in fig.findobj(mpl.text.Text):  # nâng mọi chữ nhỏ (chú thích, số trên cột...) lên mức tối thiểu
        if txt.get_text() and txt.get_fontsize() < MIN_TEXT:
            txt.set_fontsize(MIN_TEXT)
    fig.tight_layout(pad=pad)
    paths = []
    for ext in formats:
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    plt.close(fig)
    return paths
