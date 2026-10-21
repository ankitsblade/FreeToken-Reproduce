"""Build the 15 October weekly deck from the 8 October deck's own slide templates.

Every slide is cloned from a template slide of the 8 October deck (title block, rule,
heading/body blocks, numbered items, table, figure and callout). Masters, layouts,
theme, embedded fonts and the background image are copied unchanged. Numbers come
from the m-reduction result files named in SOURCES; charts are drawn with the
previous deck's palette and saved as PNG/SVG/PDF beside this script.
"""
from copy import deepcopy
import glob
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ASSETS = Path(__file__).resolve().parent
SOURCE = ROOT / "docs/presentations/FreeToken_Weekly_Progress_08-10-2026_with_static_pruning.pptx"
OUTPUT = ROOT / "docs/presentations/FreeToken_Weekly_Progress_15-10-2026.pptx"
OFFLINE = ROOT / "results/m_reduction/offline_stage1_20261014T200452Z"
STAGE2 = ROOT / "results/m_reduction/stage2_live_20261015T042628Z"
NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "a16": "http://schemas.microsoft.com/office/drawing/2014/main",
}
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)
EMU = 914400
MODELS = ["qwen_nvfp4", "gpt_oss_mxfp4", "gemma_nvfp4"]
LABELS = ["Qwen3.6", "GPT-OSS", "Gemma 4"]
COLORS = ["#287DE0", "#FF6633", "#18B57F", "#4A5B70", "#7A2331"]

with ZipFile(SOURCE) as package:
    original = {name: package.read(name) for name in package.namelist()}


def load(model, part):
    return json.loads((OFFLINE / model / part / "summary.json").read_text())[model]


p3 = {m: load(m, "p3") for m in MODELS}
p4 = {m: load(m, "p4") for m in MODELS}
p5 = {m: load(m, "p5") for m in MODELS}
p6 = {m: load(m, "p6") for m in MODELS}
p7 = {m: load(m, "p7") for m in MODELS}
p9 = {m: load(m, "p9") for m in MODELS}
seed = {m: json.loads((STAGE2 / "p1_seed" / m / "summary.json").read_text())[m] for m in MODELS}
SOURCES = sorted({str(path.relative_to(ROOT)) for path in [
    *[OFFLINE / m / p / "summary.json" for m in MODELS for p in ("p3", "p4", "p5", "p6", "p7", "p9")],
    *[STAGE2 / "p1_seed" / m / "summary.json" for m in MODELS]]})
chart_sources = {}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.titlesize": 14,
                     "axes.titleweight": "bold", "axes.labelsize": 13,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#999999", "xtick.color": "#555555", "ytick.color": "#555555",
                     "savefig.facecolor": "#FCFCFC", "figure.facecolor": "#FCFCFC", "axes.facecolor": "#FCFCFC"})


def finish_chart(fig, name):
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(ASSETS / f"{name}.{suffix}", dpi=180 if suffix == "png" else None)
    plt.close(fig)


def pct(value):
    return 100 * value


# P3: actual miss classes and the LRU/MIN diagnostic at the primary capacity.
CLASSES = [("compulsory", "first touch", COLORS[3]), ("admission_state_sensitive", "admission / state", COLORS[2]),
           ("capacity_like", "capacity-like", COLORS[0]), ("replacement_sensitive", "replacement-sensitive", COLORS[1])]
fig, (left, right) = plt.subplots(1, 2, figsize=(10.7, 5.65), gridspec_kw={"width_ratios": [1.25, 1]})
rows = {}
for i, model in enumerate(MODELS):
    cell = p3[model]["cells"]["primary/native"]
    total = sum(cell["miss_classes"].values())
    shares = [cell["miss_classes"].get(key, 0) / total * 100 for key, _, _ in CLASSES]
    rows[model] = {"mean_m": cell["mean_m"], "full_lru_mean_m": cell["full_lru_mean_m"],
                   "min_mean_m": cell["min_mean_m"], "class_share_percent": dict(zip([c[0] for c in CLASSES], shares))}
    start = 0
    for share, (_, label, color) in zip(shares, CLASSES):
        left.barh(i, share, left=start, color=color, label=label if i == 0 else None, height=.6)
        if share >= 6:
            left.text(start + share / 2, i, f"{share:.0f}%", ha="center", va="center", color="white", fontsize=11)
        start += share
left.set(yticks=range(3), yticklabels=LABELS, xlim=(0, 100), xlabel="Share of actual decode misses (%)",
         title="Why experts miss")
left.invert_yaxis()
left.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2, frameon=False, fontsize=10.5)
x = np.arange(3)
for j, (key, label) in enumerate([("mean_m", "actual (native q)"), ("full_lru_mean_m", "fetch all, LRU"),
                                  ("min_mean_m", "fetch all, Belady MIN")]):
    values = [rows[m][key] for m in MODELS]
    bars = right.bar(x + (j - 1) * .26, values, .25, color=[COLORS[0], COLORS[3], COLORS[2]][j], label=label)
    for bar, value in zip(bars, values):
        right.text(bar.get_x() + bar.get_width() / 2, value + .04, f"{value:.2f}", ha="center", fontsize=8, rotation=90, va="bottom")
right.set(xticks=x, xticklabels=LABELS, ylim=(0, 3.0), ylabel="Mean missing experts m per layer event",
          title="m at the primary cache size")
right.grid(axis="y", color="#E5E2DC"); right.set_axisbelow(True)
right.legend(loc="upper center", bbox_to_anchor=(.5, -.1), ncol=1, frameon=False, fontsize=10.5)
fig.subplots_adjust(left=.1, right=.985, bottom=.3, top=.92, wspace=.42)
finish_chart(fig, "p3_miss_classes")
chart_sources["p3_miss_classes"] = rows

# P4 q rules and P5 quotas: relative mean-m reduction against the native baseline.
P4_ARMS = [("paper_floor1", "paper fraction"), ("fixed25_floor1", "fixed 25%"),
           ("reuse_uniform_floor1", "Reuse-Aware, uniform"), ("reuse_freq_floor1", "Reuse-Aware, frequency")]
P5_ARMS = [("uniform", "uniform quotas"), ("utility", "utility quotas")]
fig, (a4, a5) = plt.subplots(1, 2, figsize=(10.7, 5.65), sharey=True, gridspec_kw={"width_ratios": [1.5, 1]})
rows = {}


def bars(ax, arms, getter, colors, width):
    for j, (key, label) in enumerate(arms):
        stats = [getter(m, key) for m in MODELS]
        rows.setdefault(label, {m: {"reduction": s["reduction"], "ci95": s["reduction_ci95"]} for m, s in zip(MODELS, stats)})
        means = np.array([pct(s["reduction"]) for s in stats])
        low = np.array([pct(s["reduction_ci95"][0]) for s in stats])
        high = np.array([pct(s["reduction_ci95"][1]) for s in stats])
        offset = (j - (len(arms) - 1) / 2) * width
        drawn = ax.bar(x + offset, means, width * .95, color=colors[j], label=label,
                       yerr=np.array([means - low, high - means]), capsize=2, error_kw={"elinewidth": 1})
        for bar, value in zip(drawn, means):
            ax.text(bar.get_x() + bar.get_width() / 2, value + (0.5 if value >= 0 else -1.6), f"{value:.1f}",
                    ha="center", fontsize=8.5)


bars(a4, P4_ARMS, lambda m, k: p4[m]["cells"]["primary/native"]["arms"][k]["vs_baseline_mean_m"], COLORS, .2)
bars(a5, P5_ARMS, lambda m, k: p5[m]["comparisons"][k]["vs_global_seeded_mean_m"], [COLORS[3], COLORS[4]], .34)
for ax, title in ((a4, "P4: which misses to fetch (q rule)"), (a5, "P5: slots per layer")):
    ax.axhline(5, color="#7A2331", linestyle="--", linewidth=1)
    ax.axhline(0, color="#777777", linewidth=.8)
    ax.set(xticks=x, xticklabels=LABELS, title=title, ylim=(-14, 13))
    ax.grid(axis="y", color="#E5E2DC"); ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.07), fontsize=10, frameon=False, ncol=1 if ax is a5 else 2)
a4.set_ylabel("Mean-m reduction vs native (%), 95% CI")
a5.text(2.45, 5.4, "5% G2 bar", color="#7A2331", fontsize=9.5, ha="right")
fig.suptitle("Offline replay, 90 D0 requests per model, primary cache; positive = fewer misses",
             fontsize=13, fontweight="bold", y=.97)
fig.subplots_adjust(left=.08, right=.985, bottom=.2, top=.86, wspace=.08)
finish_chart(fig, "p4_p5_exact_policies")
chart_sources["p4_p5_exact_policies"] = rows

# P7 routing screen.
fig, ax = plt.subplots(figsize=(10.7, 5.65))
rows = {}
P7_ARMS = [("r75_frequency", "R75 frequency mask", "#4A5B70"), ("f75_dropped", "F75 dropped demand", "#9AA5B1"),
           ("random", "random 75% masks (median, 5-95%)", "#C9B79C"), ("arm1_miss_targeted", "arm 1: miss-targeted exclusion", COLORS[0]),
           ("arm2_alias", "arm 2: cluster alias", COLORS[1])]
for j, (key, label, color) in enumerate(P7_ARMS):
    means, low, high = [], [], []
    for m in MODELS:
        if key == "random":
            base = p7[m]["comparisons"]["r75_frequency"]["baseline"]
            q05, q50, q95 = (1 - v / base for v in p7[m]["random_mask_mean_m"]["quantiles"])
            means.append(pct(q50)); low.append(pct(q95)); high.append(pct(q05))
            rows.setdefault(label, {})[m] = {"median": q50, "p5_reduction": q95, "p95_reduction": q05}
        else:
            s = p7[m]["comparisons"][key]
            means.append(pct(s["reduction"])); low.append(pct(s["reduction_ci95"][0])); high.append(pct(s["reduction_ci95"][1]))
            rows.setdefault(label, {})[m] = {"reduction": s["reduction"], "ci95": s["reduction_ci95"]}
    means, low, high = map(np.array, (means, low, high))
    drawn = ax.bar(x + (j - 2) * .16, means, .15, color=color, label=label,
                   yerr=np.array([means - low, high - means]), capsize=2, error_kw={"elinewidth": 1})
    for bar, value, top in zip(drawn, means, high):
        ax.text(bar.get_x() + bar.get_width() / 2, max(value, top) + .8, f"{value:.0f}", ha="center", fontsize=9.5)
ax.axhline(10, color="#7A2331", linestyle="--", linewidth=1, label="10% G2 bar (output-changing)")
ax.set(xticks=x, xticklabels=LABELS, ylim=(0, 58), ylabel="Mean-m reduction vs native (%)",
       title="75% retention routing screen: fixed logits, 90 D0 requests, out-of-fold masks")
ax.grid(axis="y", color="#E5E2DC"); ax.set_axisbelow(True)
ax.legend(loc="upper left", fontsize=10, frameon=False, ncol=2)
fig.subplots_adjust(left=.08, right=.985, bottom=.08, top=.9)
finish_chart(fig, "p7_routing_screen")
chart_sources["p7_routing_screen"] = rows
(ASSETS / "chart_source_rows.json").write_text(json.dumps(chart_sources, indent=2) + "\n")


# ---------------------------------------------------------------- slide templates
def parse(number):
    return ET.fromstring(original[f"ppt/slides/slide{number}.xml"])


def tree(slide):
    return slide.find("p:cSld/p:spTree", NS)


def text_of(shape):
    return "".join(node.text or "" for node in shape.findall(".//a:t", NS))


def text_shape(slide, starts):
    return next(shape for shape in tree(slide) if text_of(shape).startswith(starts))


def set_text(shape, value, size=None):
    body = shape.find("p:txBody", NS)
    first = body.find("a:p", NS)
    ppr = first.find("a:pPr", NS)
    rpr = first.find("a:r/a:rPr", NS)
    if rpr is None:
        rpr = first.find("a:pPr/a:defRPr", NS)
    for p in list(body.findall("a:p", NS)):
        body.remove(p)
    for line in value.split("\n"):
        p = ET.SubElement(body, f"{{{NS['a']}}}p")
        if ppr is not None:
            p.append(deepcopy(ppr))
        r = ET.SubElement(p, f"{{{NS['a']}}}r")
        if rpr is not None:
            style = deepcopy(rpr); style.tag = f"{{{NS['a']}}}rPr"
            if size:
                style.set("sz", str(size))
            r.append(style)
        ET.SubElement(r, f"{{{NS['a']}}}t").text = line
        if size and p.find("a:pPr/a:defRPr", NS) is not None:
            p.find("a:pPr/a:defRPr", NS).set("sz", str(size))


def geometry(shape, x, y, w, h):
    xf = shape.find(".//a:xfrm", NS)
    if xf is None:
        xf = shape.find("p:xfrm", NS)
    xf.find("a:off", NS).attrib.update(x=str(round(x * EMU)), y=str(round(y * EMU)))
    xf.find("a:ext", NS).attrib.update(cx=str(round(w * EMU)), cy=str(round(h * EMU)))


def base(label, title):
    slide = parse(1)
    tr = tree(slide)
    for shape in list(tr)[6:]:
        tr.remove(shape)
    set_text(list(tr)[3], label)
    set_text(list(tr)[4], title)
    return slide


def append_text(slide, template_text, content, x, y, w, h):
    shape = deepcopy(text_shape(parse(1), template_text))
    set_text(shape, content); geometry(shape, x, y, w, h); tree(slide).append(shape)
    return shape


def callout(slide, label, content):
    for shape in list(tree(parse(18)))[7:]:
        copy = deepcopy(shape)
        if text_of(copy) == "Reading":
            set_text(copy, label)
        elif text_of(copy):
            set_text(copy, content)
        tree(slide).append(copy)


def blocks(label, title, values, footer, top=1.98, step=1.27):
    slide = base(label, title)
    for i, (heading, body) in enumerate(values):
        y = top + i * step
        append_text(slide, "Reproduction complete", heading, .5, y, 12.333, .3333)
        append_text(slide, "Read the paper", body, .5, y + .397, 12.333, .70)
    callout(slide, *footer)
    return slide


def numbered(label, title, items, footer):
    slide = parse(2)
    tr = tree(slide)
    set_text(list(tr)[3], label)
    set_text(list(tr)[4], title)
    names = {"1": items[0][0], "Runtime-aware q selection": items[0][1], "Choose the CPU/GPU expert sp": items[0][2],
             "2": items[1][0], "Adaptive reduction of m": items[1][1], "Study selective expert pruni": items[1][2]}
    for shape in list(tr):
        node = shape.find("p:nvSpPr/p:cNvPr", NS)
        if node is None:
            continue
        if node.get("name") in names:
            set_text(shape, names[node.get("name")])
        elif node.get("name") == "Rectangle 27":
            tr.remove(shape)
    callout(slide, *footer)
    return slide


def table_slide(label, title, headers, rows, widths, caption, footer, heading="", body="",
                height=.5, size=1300, top=1.92):
    slide = base(label, title)
    frame = deepcopy(next(shape for shape in tree(parse(17)) if shape.tag.endswith("graphicFrame")))
    table = frame.find(".//a:tbl", NS)
    grid = table.find("a:tblGrid", NS)
    for node in list(grid):
        grid.remove(node)
    for width in widths:
        ET.SubElement(grid, f"{{{NS['a']}}}gridCol", w=str(round(width * EMU)))
    originals = table.findall("a:tr", NS)
    for row in originals:
        table.remove(row)
    heights = [height if i == 0 else (height * values[1] if isinstance(values, tuple) else height)
               for i, values in enumerate([headers, *rows])]
    for i, values in enumerate([headers, *rows]):
        cells = values[0] if isinstance(values, tuple) else values
        seed_row = originals[0] if i == 0 else originals[1 + (i - 1) % 2]
        row = ET.Element(f"{{{NS['a']}}}tr", h=str(round(heights[i] * EMU)))
        for j, value in enumerate(cells):
            cell = deepcopy(seed_row.findall("a:tc", NS)[min(j, len(seed_row.findall("a:tc", NS)) - 1)])
            proxy = ET.Element(f"{{{NS['p']}}}sp")
            cell_body = cell.find("a:txBody", NS)
            cell.remove(cell_body)
            cell_body.tag = f"{{{NS['p']}}}txBody"; proxy.append(cell_body)
            set_text(proxy, value, size)
            proxy.remove(cell_body); cell_body.tag = f"{{{NS['a']}}}txBody"; cell.insert(0, cell_body)
            row.append(cell)
        table.append(row)
    geometry(frame, .5, top, sum(widths), sum(heights))
    tree(slide).append(frame)
    end = top + sum(heights)
    caption_shape = deepcopy(text_shape(parse(17), "Latency:"))
    set_text(caption_shape, caption); geometry(caption_shape, .5, end + .1, 12.333, .36); tree(slide).append(caption_shape)
    if heading:
        append_text(slide, "Reproduction complete", heading, .5, end + .58, 12.333, .3333)
        append_text(slide, "Read the paper", body, .5, end + .97, 12.333, .70)
    callout(slide, *footer)
    return slide


def chart_slide(label, title, name, footer):
    slide = base(label, title)
    pic = deepcopy(list(tree(parse(18)))[6])
    pic.find(".//a:blip", NS).set(f"{{{NS['r']}}}embed", "rIdWeekFigure")
    pic.find("p:nvPicPr/p:cNvPr", NS).set("descr", title)
    tree(slide).append(pic)
    callout(slide, *footer)
    return slide


slides = []


def add(slide, notes, figure=None):
    slides.append((slide, notes, figure))


def notes(*points, sources=()):
    lines = ["• " + point for point in points]
    if sources:
        lines.append("Sources: " + "; ".join(sources))
    return "\n".join(lines)


def fmt(value, digits=1):
    return f"{pct(value):.{digits}f}%"


# ---------------------------------------------------------------- 1-2: context and objectives
add(blocks("LAST WEEK · CONTEXT 01", "Where the project stood on 8 October", [
    ("Runtime-aware q on one host", "Reuse-Aware gave a 1.52× geometric-mean decode speedup over q* for GPT-OSS-20B on the RTX 4070. The learned Ridge and Linear cost models did not generalise."),
    ("Static 75% expert retention", "Task-specific masks kept most decode accesses, but no checkpoint passed every activation check, and rerouting cut misses by at most 2.24%."),
    ("Two open questions", "Does the q rule hold on other GPUs and models? Can we lower m, the number of routed experts missing from the GPU cache, without losing quality?"),
], ("This week", "15 October 2026: plan the cross-architecture study and run the m-reduction study.")),
    notes("Recap of the 8 October deck in one slide; no new numbers here.",
          "Reuse-Aware: analytic rule that fetches a missing expert when its expected future reuse beats the marginal transfer cost; 1.52× was GPT-OSS only, one host, eager decode.",
          "Learned cost models (Ridge, Linear) degraded on unseen prompts and cache sizes, which is why we keep the analytic rule.",
          "Static pruning: coverage looked good per task, but no checkpoint passed all activation gates; fixed-cache rerouting gave Qwen 0.87-2.24%, GPT-OSS 0%, Gemma 0.46% fewer misses.",
          "The two open questions map directly onto this week's two directions.",
          sources=("8 October deck slides 3-10 and 27-30",)))

add(numbered("WEEKLY OBJECTIVES · 15 OCTOBER 2026", "Two directions for this week", [
    ("1", "Runtime-aware q across architectures", "Collect the same decision episodes on more GPUs and MoE models, so the q rule is fitted and tested across hardware rather than on one host."),
    ("2", "Family-adaptive reduction of m", "Test cache admission, replacement, per-layer allocation, routing, and 75% pruning or merging on GPT-OSS, Qwen and Gemma, then confirm survivors live."),
], ("Today", "Direction 1 is a plan. Direction 2 has qualified baselines, offline results and live runs in progress.")),
    notes("Direction 1 = how many misses to fetch (q) on different hardware; Direction 2 = how to have fewer misses (m) in the first place.",
          "q decides how existing misses are served (PCIe fetch vs CPU execution); m is the number of misses.",
          "Direction 1 slides explain the planned cross-architecture experiment (plan document pages 3-4); nothing is collected yet on other GPUs.",
          "Direction 2 slides first summarise every planned experiment (P1-P9), then show the results that are ready.",
          sources=("exp_plan.pdf pages 3-4", "docs/PLAN.md")))

# ---------------------------------------------------------------- Direction 1
add(table_slide("DIRECTION 1 · GPU ARCHITECTURES", "Target GPUs differ in PCIe speed and VRAM",
    ["GPU", "Architecture", "VRAM", "Host link (nominal)", "Status"],
    [["RTX 4070", "Ada Lovelace", "12 GB", "PCIe 4.0; measured x4 (B_P = 6.7 GB/s)", "Reference: data exists"],
     ["RTX 4090", "Ada Lovelace", "24 GB", "PCIe 4.0 x16", "Planned"],
     ["RTX 5090", "Blackwell", "32 GB", "PCIe 5.0 x16", "Planned"],
     ["RTX 3090, L40S, A100, H100", "As available", "-", "-", "Added when access exists"]],
    [3.0, 1.9, 1.1, 3.733, 2.6], "Nominal link speeds are vendor specifications; the actual link, B_P and B_H are measured on each host at Stage 0.",
    ("Key ratio", "rho = B_P / B_H: PCIe expert-fetch bandwidth relative to host CPU expert bandwidth."),
    heading="These GPUs differ in exactly what the rule depends on",
    body="A faster PCIe link raises B_P and so rho; more VRAM holds a larger expert cache. CPU model, cores and memory are recorded; the same GPU with a different CPU is a separate cell.",
    height=.5),
    notes("Only the RTX 4070 has data today; every other row is planned.",
          "Our host link is PCIe 4.0 but negotiated at x4, so measured B_P is 6.7 GB/s, far below a desktop x16 slot.",
          "The q rule depends on rho = B_P/B_H; a 4090 or 5090 at x16 changes rho a lot, which is the point of testing them.",
          "More VRAM means a larger expert cache, so fewer misses and a different reuse horizon.",
          "B_P and B_H are always measured on the machine, never taken from vendor numbers.",
          "Same GPU with a different CPU counts as a different cell because B_H changes.",
          sources=("exp_plan.pdf page 3, Table 4",)))

add(table_slide("DIRECTION 1 · MOE MODELS", "Three models span 4× in top_k/E and 7× in expert size",
    ["Profile", "Checkpoint (pinned revision)", "Layers", "Experts", "top_k", "top_k/E", "Expert format"],
    [["gpt_oss_mxfp4", "openai/gpt-oss-20b", "24", "32", "4", "0.125", "MXFP4, 13.2 MB/expert"],
     ["qwen_nvfp4", "nvidia/Qwen3.6-35B-A3B-NVFP4", "40", "256", "8", "0.031", "NVFP4, 1.8 MB/expert"],
     ["gemma_nvfp4", "nvidia/Gemma-4-26B-A4B-NVFP4", "30", "128", "8", "0.063", "NVFP4, 3.4 MB/expert"]],
    [1.9, 3.6, 1.0, 1.1, .95, 1.1, 2.683], "Pins in configs/pruning/models.json. Layers are MoE layers; experts are per layer.",
    ("Fitting", "Cross-cell models use dimensionless inputs: rho, top_k/E, residency in tokens and miss ratio."),
    heading="Why these three", body="top_k/E sets how often an expert is reused; expert size sets the price of one fetch. Both change which misses are worth fetching.",
    height=.5),
    notes("Same three checkpoints as the m-reduction study, so both directions share models.",
          "top_k/E is the fraction of experts used per token: GPT-OSS 4 of 32 (12.5%), Qwen 8 of 256 (3.1%), Gemma 8 of 128 (6.3%).",
          "Expert size: GPT-OSS 13.2 MB, Qwen 1.8 MB, a 7× spread; one GPT-OSS fetch costs much more PCIe time.",
          "Gemma's 3.4 MB comes from the m-reduction plan's recorded geometry; the experiment plan table leaves it blank.",
          "Prefer dimensionless features so a rule fitted on one cell can be tested on another.",
          sources=("exp_plan.pdf page 3, Table 5", "docs/PLAN.md section 2.2 (Gemma expert bytes)")))

add(table_slide("DIRECTION 1 · CELL MATRIX", "Each GPU and model pair is a separate cell",
    ["GPU", "gpt-oss-20b", "Qwen3.6-35B-A3B", "Gemma-4-26B-A4B"],
    [["RTX 4070", "Done (reference)", "Planned, if host RAM holds the expert bank", "Planned"],
     ["RTX 4090", "Planned", "Planned", "Planned"],
     ["RTX 5090", "Planned", "Planned", "Planned"],
     ["Others", "Planned", "Planned", "Planned"]],
    [2.3, 2.8, 4.533, 2.7], "A cell is attempted only after it passes Stage 0 qualification.",
    ("Recorded", "The chosen cache rates go into each cell's plan file, so every cell can be rebuilt."),
    heading="Cache sizes are set relative to each model",
    body="Three rates of the layer × expert pool give low, medium and high miss pressure. Each is at least the engine minimum (2 × experts per layer with prefill overlap) and fits the GPU's VRAM.",
    height=.5),
    notes("12 cells in total; only GPT-OSS on the RTX 4070 is done (last week's runtime-q study).",
          "Qwen on the 4070 depends on host RAM holding the pinned expert bank (about 21 GB server RSS on our 30 GB host).",
          "Cache size is not a fixed slot count across models: three fractions of each model's own pool, giving low/medium/high miss pressure.",
          "Floor: two full layers of experts for prefill overlap; ceiling: what fits in that GPU's VRAM.",
          "Stage 0 failure (kernels, VRAM, RAM, loader) is recorded as a failed cell, not silently skipped.",
          sources=("exp_plan.pdf page 3, Table 6 and cache-size paragraph",)))

add(blocks("DIRECTION 1 · PROTOCOL 1 OF 2", "Stage 0 qualifies every machine first", [
    ("Install and check the host", "Clone the repository, run setup and doctor before and after building the pinned FreeToken runtime, and download each pinned checkpoint."),
    ("Measure the machine", "Run the bandwidth calibration to measure B_P and B_H for each expert format, then the unpruned baseline for each model with three repeats."),
    ("Reproduce the reference", "Run the runtime-q pilot plan and confirm the hook reproduces the engine's q*: identical decisions and matching throughput."),
], ("Failed cell", "Missing kernels, too little VRAM or host RAM, or loader errors: kept with logs, not collected.")),
    notes("Commands: scripts/setup.sh --analysis-only, scripts/doctor.sh (before runtime), scripts/setup.sh (builds pinned runtime), doctor --require-runtime, then source scripts/env.sh under bash.",
          "doctor writes machine_before_runtime.json and machine_after_runtime.json, the hardware record for the cell.",
          "Calibration measures B_P (PCIe) and B_H (host CPU) per expert format on that machine.",
          "Baseline: scripts/baseline.sh --profile gpt_oss_mxfp4 --model $MODEL --gpu 0 --repeats 3.",
          "The pilot check proves the hook does not change the engine: same q* decisions and the same throughput.",
          "The repository (ankitsblade/FreeToken-Reproduce) packages all of this so collection is identical on every machine.",
          sources=("exp_plan.pdf pages 3-4, Stage 0; section 7 opening",)))

add(blocks("DIRECTION 1 · PROTOCOL 2 OF 2", "Collect, transfer, then analyse within and across cells", [
    ("Stage 1 · episode collection", "Four plans per model: collect, round 2, main and generalisation. Each run saves run.json, environment.json, logs, episodes and decisions; reruns skip finished episodes."),
    ("Stage 2 · transfer", "Results stay out of Git. Each machine archives and verifies its run directories; the analysis machine imports them into a fresh study root. Raw runs are never edited."),
    ("Stage 3 · analysis", "Per cell: policy comparison with paired 95% intervals, the wall(m, q) cost table, reuse analysis and ablations. Across cells: leave-one-architecture-out and leave-one-model-out."),
], ("Cross-cell fit", "Compare the rule's predicted fetch cost with each GPU's measured cost table.")),
    notes("Stage 1 command: bash scripts/run.sh runtime_q run --plan configs/runtime_q/<plan>.json --model $MODEL --gpu 0, for collect, round2, main and generalization plans.",
          "Each invocation writes results/runs/runtime_q/UTC_UUID/ with run.json (command, times, status) and environment.json (CPU, RAM, GPU, driver, packages, Git state, engine hashes).",
          "Interrupted runs resume because the driver skips episodes already present.",
          "Stage 2: archive, verify, transfer to the shared store, import into a fresh study root; nothing edited after collection.",
          "Stage 3 per cell repeats last week's single-machine tables; across cells, pool decisions with machine/model IDs and hold out one architecture or one model at a time.",
          "Key test: does the rule's predicted fetch cost match each GPU's measured wall(m, q) table?",
          sources=("exp_plan.pdf page 4, Stages 1-3",)))

add(table_slide("DIRECTION 1 · CONTROLS", "Controls that keep cells comparable",
    ["Control", "Rule"],
    [["Held constant", "Batch 1, greedy, 128 measured tokens, one warmup, eager decode, naive KV cache, same protocol and policies"],
     ["Prompts", "Frozen token-ID prompts: the GPT-OSS chat template embeds the local date, which changes prompt and routing"],
     ["One GPU job at a time", "Concurrent servers corrupt timings; no heavy CPU work, such as checkpoint-scale tensor jobs, beside a GPU run"],
     ["Burn-in and order", "Three discarded burn-in episodes per server; policy order rotated across repeats"],
     ["Memory", "Runs execute in a scope without swap; RAM or swap pressure rejects the whole paired unit"],
     ["Separate arms", "A change of quantization, cache size, driver, runtime source or execution mode is a new arm, not a repeat"],
     ["Correctness", "Output hashes per episode; divergent text is documented, since CPU and GPU kernels are not bit-identical"]],
    [2.6, 9.733], "Measurement hygiene carried over from the single-machine study.",
    ("Origin", "The date, memory and CPU/GPU rounding rules each come from a failure seen in our own runs."),
    height=.5, size=1250, top=1.85),
    notes("These rules make cells comparable; they do not make the machines identical.",
          "Date in template: GPT-OSS renders today's date into the prompt, so re-rendering on another day changes tokens and routing; we send frozen token IDs instead.",
          "One GPU job at a time: concurrent servers or heavy CPU work distort timing; we also had an out-of-memory incident from a CPU job beside a GPU run.",
          "Memory: runs execute inside a no-swap scope; swap or memory pressure rejects the whole paired unit, which is then repeated.",
          "Any change of quantization, cache size, driver or runtime is a new arm, never pooled with old repeats.",
          "CPU and GPU expert kernels are not bit-identical, so output divergence between policies is recorded rather than hidden.",
          sources=("exp_plan.pdf page 4, Table 7",)))

# ---------------------------------------------------------------- Direction 2: plan summary
add(blocks("DIRECTION 2 · MOTIVATION", "Lower the number of experts that miss the cache", [
    ("What m measures", "For each decode token and MoE layer, m counts routed experts not resident in the GPU expert cache. Each miss is fetched over PCIe (q of them) or run on the CPU (m − q)."),
    ("Why reduce it", "The q rule only decides how misses are served. Fewer misses means less PCIe and CPU work per token on the same hardware and cache bytes."),
    ("Five levers", "Admission (which misses to fetch), replacement (what to evict), allocation (slots per layer), routing (prefer resident experts), and removing or merging 25% of experts."),
], ("Boundary", "Lower m alone is not fewer transfers, lower latency or kept quality; each is measured.")),
    notes("m is counted before admission: unique routed experts for that token and layer minus those already in the GPU cache.",
          "q is the number of those misses fetched over PCIe; the other m − q run on the CPU.",
          "Direction 1 optimises q for a given m; Direction 2 tries to make m smaller.",
          "Levers 1-3 (admission, replacement, allocation) never change model outputs, so they only need m and latency evidence.",
          "Levers 4-5 (routing, pruning/merging) change which experts compute, so they also need quality evidence.",
          "m × expert bytes is missing demand, not measured PCIe traffic.",
          sources=("docs/PLAN.md sections 1 and 4.1",)))

geometry_rows = []
names = {"qwen_nvfp4": "Qwen3.6-35B-A3B NVFP4", "gpt_oss_mxfp4": "GPT-OSS-20B MXFP4", "gemma_nvfp4": "Gemma-4-26B-A4B NVFP4"}
caps = {"qwen_nvfp4": "2,048", "gpt_oss_mxfp4": "2,048", "gemma_nvfp4": "1,280"}
for model in MODELS:
    g = p3[model]["geometry"]
    geometry_rows.append([names[model], f"{g['layers']} × {g['experts']}", str(g["top_k"]), f"{g['cache_slots']:,}",
                          f"{g['cache_slots'] / (g['layers'] * g['experts']):.1%}", f"{g['expert_bytes'] / 1e6:.1f} MB", caps[model]])
add(table_slide("DIRECTION 2 · SETUP", "Three models at a fixed 20% expert cache",
    ["Model", "Layers × experts", "top-k", "Cache slots C", "C / pool", "Expert size", "Sequence cap"], geometry_rows,
    [3.3, 1.9, .9, 1.6, 1.35, 1.6, 1.683], "Capacity sensitivity uses 10% and 40% of the pool offline; equal-budget comparisons keep C and cache bytes fixed.",
    ("Rule", "Every experiment runs on all three models; a blocked cell is recorded and repaired."),
    heading="RTX 4070 (12 GB, PCIe Gen4 x4) + Ryzen 5 8500G, 30 GB RAM",
    body="FreeToken hybrid, batch 1, greedy, eager decode, five CPU expert workers. Every run is in a no-swap scope with host-memory, swap and pressure gates.",
    height=.5),
    notes("All values from the recorded model geometry; C is about 20% of each model's layer × expert pool.",
          "GPT-OSS has only 154 slots (about 6.4 per layer) for 32 experts per layer, so its cache is tight.",
          "Qwen holds 2,048 small experts, Gemma 768.",
          "Gemma's sequence cap is 1,280 tokens because of memory; prompts are capped at 1,024 and generation at 256 for all.",
          "Resource gates: at least 4 GiB free RAM, at most 128 MiB swap during inference, at most 1% memory pressure; a failed gate rejects and repeats the whole batch.",
          "All three models are mandatory for every experiment.",
          sources=("docs/PLAN.md sections 1.1, 2.2 and 2.3", "offline_stage1 */p3/summary.json geometry")))

add(table_slide("DIRECTION 2 · DATASETS", "Four datasets, each with one job",
    ["Dataset", "Content", "Role"],
    [(["D0 routing corpus", "90 requests per model: 30 math, 30 code, 30 incident; 5 grouped folds; 9 repeat IDs", "Fit and select every policy; replay; the 9-request live timing cohort"], 1.4),
     (["D1 synthetic quality", "300 tasks (10 families × 10 variants per domain) and 100 sequences", "Task-score and KL regression screen"], 1.4),
     (["D2 AIME 2025-I", "15 problems, one greedy sample each, 512-token input budget", "Independent math confirmation (W1-style)"], 1.4),
     (["D3 SWE-bench Verified", "5 psf/requests issues, 3 scripted OpenCode turns each", "Agentic confirmation (W2-style); harness not pinned yet"], 1.4)],
    [2.6, 5.4, 4.333], "D1-D3 never choose a candidate; they confirm or reject a frozen one.",
    ("Statistics", "10,000-resample paired bootstrap over requests; each workload weighs one third."),
    heading="Nested selection", body="Parameters are chosen on the inner folds of the 72 training requests and reported on the 18 untouched outer requests of each fold.",
    height=.46, size=1250),
    notes("D0 is the original routing corpus from the static pruning study, rendered to frozen token IDs per model.",
          "Five grouped outer folds: each holds out 18 requests (6 per domain); inside each, four inner folds pick parameters.",
          "Nine repeat IDs (3 per domain) form the live timing cohort, five paired repetitions each.",
          "D1 is a known regression screen (we have seen its earlier outcomes), so it is not fresh confirmation.",
          "D2: opencompass/AIME2025 at a pinned revision; all 15 problems fit the input budget for all three models.",
          "D3 needs a pinned OpenCode + SWE-bench harness, still open; it may not fit Gemma's 1,280-token cap.",
          "Small D2/D3 samples are directional only, never pooled into D0.",
          sources=("docs/PLAN.md sections 3.1-3.5 and 4.3",)))

add(table_slide("DIRECTION 2 · EXPERIMENTS 1 OF 2", "P1-P5: baseline, traces and exact cache policies",
    ["Exp.", "Question", "Arms", "Outputs"],
    [(["P1", "Does q* fill, and what baseline is valid?", "Native q with and without a one-expert floor; Reuse-Aware; prefill-informed cache seeding", "Unchanged"], 1.3),
     (["P2", "Does replay reproduce live serving?", "Full router logits, cache events, teacher forcing, zero-change controls", "Unchanged"], 1.3),
     (["P3", "Why do experts miss?", "Miss classes; fully admitting LRU vs Belady MIN; deletion estimates", "Diagnosis"], 1.3),
     (["P4", "Which misses should be fetched?", "Paper, overlap (native), fixed 25%, Reuse-Aware with uniform or frequency prior", "Unchanged"], 1.3),
     (["P5", "How many cache slots per layer?", "Global LRU vs uniform vs utility per-layer quotas at equal total bytes", "Unchanged"], 1.3)],
    [.8, 3.7, 6.233, 1.6], "Exact components: outputs must stay token-identical; m and latency decide adoption.",
    ("Baseline", "overlap_floor1, the native q with a one-expert fetch floor, is the reference arm throughout."),
    height=.46, size=1250, top=1.85),
    notes("P1: the paper says at least one fill per miss event; the shipped code can round q to zero. floor1 adds the one-expert floor; seeding warms the cache from prompt routes.",
          "P2: we need exact replay so offline results transfer to live serving; teacher forcing lets us measure quality on fixed token histories.",
          "P3 classifies each actual miss: first touch, admission-sensitive, capacity-like, replacement-sensitive; MIN is the optimal-eviction bound.",
          "P4 compares q rules: paper fraction, the measured-overlap fraction (native), fixed 25%, and Reuse-Aware with a uniform or recent-frequency reuse prior.",
          "P5 fixes slots per layer instead of one global LRU, at the same total bytes; utility quotas come from training-request miss curves.",
          sources=("docs/PLAN.md sections 5-8",)))

add(table_slide("DIRECTION 2 · EXPERIMENTS 2 OF 2", "P6-P9: routing, pruning, composition, replacement",
    ["Exp.", "Question", "Arms", "Outputs"],
    [(["P6", "Can routing prefer cached experts?", "A: cache-prior bonus, λ 0.1-0.4; B: low-score swap, alpha 0.1 or 0.3; C: A then B", "Changed"], 1.3),
     (["P7", "Does removing or merging 25% help?", "Arm 1: miss-targeted decode exclusion; arm 2: cluster merge + router alias; R75, F75, random", "Changed"], 1.3),
     (["P8", "Do routing and q gains compound?", "2 × 2: native or P6 routing × native or Reuse-Aware q; interaction term I_m", "Depends"], 1.3),
     (["P9", "Does protection against thrashing help?", "Segmented LRU with 25% or 50% protected; pinning 10% or 25% of slots", "Unchanged"], 1.3)],
    [.8, 3.7, 6.233, 1.6], "Output-changing components (P6, P7) need the quality gate before any timing result counts.",
    ("Controls", "Every changed-routing arm has a zero-change control that must reproduce native routes exactly."),
    height=.46, size=1250, top=1.85),
    notes("P6: bias routing toward experts already in cache; the top-J original experts are protected (J=1 GPT-OSS, J=2 Qwen/Gemma); gate weights stay from the original logits.",
          "P7 retains 75% of experts per layer (remove 64/8/32 for Qwen/GPT-OSS/Gemma). Arm 1 masks the experts with most misses and least saliency at decode only; arm 2 merges similar experts and aliases router IDs.",
          "R75 = keep the 75% most frequent; F75 = delete the dropped accesses (diagnostic only); 200 random masks as a sensitivity band.",
          "P8 asks whether routing and q improvements add up; it needs a qualified P6 finalist.",
          "P9 protects re-used experts from eviction (segmented LRU) or pins thrashing experts.",
          sources=("docs/PLAN.md sections 9-10",)))

add(blocks("DIRECTION 2 · STAGES AND GATES", "Each stage passes a gate before the next", [
    ("Stage 0 · qualify (P1, P2)", "Fresh baseline captures; observation must not change outputs; replay must match live event by event; teacher forcing must reproduce baseline logits."),
    ("Stage 1 · offline screens (P3-P9)", "Replay all 90 requests per model at three cache sizes and select parameters on inner folds. G1 sets priority but never skips a model."),
    ("Stage 2 · live and quality", "Fresh-server timing on 9 requests × 5 paired repeats; teacher-forced KL; D1 and D2 task quality; merged-weight execution for P7 arm 2."),
], ("G2 adoption", "Exact: ≥5% m cut, same outputs, time ≤1.02×. Output-changing: ≥10% m, KL ≤0.02, top-1 ≥98%.")),
    notes("G0 (Stage 0): kernels and resources qualified on all three models; observation off/on gives identical output; replay matches live exactly.",
          "G1 (after P3) only reorders work: e.g. many replacement misses would prioritise P9; all experiments still run on all models.",
          "G2 exact components: at least 5% mean-m reduction with a positive lower 95% bound (or 10% on P95), no output change, timing ratio upper bound ≤ 1.02.",
          "G2 output-changing: at least 10% mean-m reduction plus quality: KL upper bound ≤ 0.02 nats/token per workload, top-1 agreement lower bound ≥ 98%, D1 task loss ≤ 2 points, Qwen/Gemma perplexity ≤ +2%.",
          "Also: positive reduction in at least 4 of 5 folds and no confidently negative domain.",
          sources=("docs/PLAN.md sections 4, 7, 10.1 and 10.2",)))

# ---------------------------------------------------------------- Direction 2: results
add(blocks("RESULTS · STAGE 0 QUALIFICATION", "CPU and GPU expert paths had to agree first", [
    ("Outputs diverged when q changed", "Changing q moves experts between CPU and GPU. GPT-OSS pilot outputs differed in 12-13 of 15 requests; the first router-logit gap was only 0.0078."),
    ("Numerical alignment fixed it", "GPT-OSS: FP64 split-K accumulation with matched BF16 boundaries. Qwen and Gemma: an NVFP4 precision path. Every pilot arm then matched token for token."),
    ("Collection and P2 qualified", "90 primary and 18 repeat captures per model. Teacher forcing reproduced all 270 histories exactly; the zero-change control gave KL = 0."),
], ("Cost", "The aligned CPU path is slower (GPT-OSS concurrent CPU 34.9 → 4.3 GB/s); it is the new baseline.")),
    notes("This was the first blocker: the plan requires identical outputs across exact arms, and they were not.",
          "Cause: CPU and GPU expert kernels round differently; a 0.0078 router-logit difference at decode flipped a later token (first difference at token 47 on math_000).",
          "GPT-OSS fix (dot64): FP64 accumulation of split-K partial sums, same activation evaluation order, BF16 rounding at the same boundaries on both paths.",
          "Qwen/Gemma fix: NVFP4 precision path with FP64 accumulation and matched BF16 boundaries; disabling activation quantization alone was not enough.",
          "After the fix, all pilot arms matched; 270 primary captures (90 per model) plus 54 repeat captures were accepted.",
          "Slower CPU path is charged honestly: fresh calibration, not the faster arm's numbers.",
          "Host memory was the other recurring blocker (Qwen needs about 21 GB); retries and pausing background processes were logged.",
          sources=("docs/EXPERIMENT_LOG.md, 13-14 October entries", "docs/PLAN.md section 5.1")))

p3_lines = []
for model, label in zip(MODELS, LABELS):
    c = chart_sources["p3_miss_classes"][model]
    shares = c["class_share_percent"]
    p3_lines.append(f"{label}: m {c['mean_m']:.2f}; first touch {shares['compulsory']:.1f}%, admission {shares['admission_state_sensitive']:.1f}%, "
                    f"capacity-like {shares['capacity_like']:.1f}%, replacement-sensitive {shares['replacement_sensitive']:.1f}%; "
                    f"fetch-all LRU {c['full_lru_mean_m']:.2f}, MIN {c['min_mean_m']:.2f}.")
s = chart_slide("RESULTS · P3 MISS DIAGNOSIS", "Capacity and replacement cause most misses", "p3_miss_classes",
                ("Reading", "Decode starts with an empty cache, so first touches are 6-28% of misses."))
add(s, notes(*p3_lines,
             "Classes are assigned in order: first touch; admission-sensitive if fetch-all LRU would hit; capacity-like if both LRU and MIN miss; replacement-sensitive if only LRU misses.",
             "MIN (Belady) is an offline optimum with future knowledge: a bound, not an achievable policy at fixed q.",
             "All 270 captures enter decode with zero cached experts (prefill uses separate staging), so cold and native starts are identical.",
             "G1 outcome on all three: run P4/P5 first; P9 and P6 not prioritised (thrash concentration about 19%, below the 50% trigger); P7 not deprioritised.",
             sources=("offline_stage1 */p3/summary.json, cell primary/native",)), "p3_miss_classes")

p45 = chart_sources["p4_p5_exact_policies"]
s = chart_slide("RESULTS · P4/P5 EXACT CACHE POLICIES", "Exact cache policies give modest, model-specific gains", "p4_p5_exact_policies",
                ("Reading", "Only GPT-OSS quotas clear 5% offline; no q rule beats native by more than 3.4%."))
add(s, notes(
    "Bars are mean-m reduction against the native overlap_floor1 arm; error bars are paired request-bootstrap 95% CIs.",
    "P4 inner-fold selection: GPT-OSS fixed 25% (5/5 folds), Gemma Reuse-Aware frequency, Qwen Reuse-Aware uniform (5/5).",
    f"Qwen: fixed 25% is worse ({fmt(p45['fixed 25%']['qwen_nvfp4']['reduction'])}); Reuse-Aware uniform {fmt(p45['Reuse-Aware, uniform']['qwen_nvfp4']['reduction'])} fewer misses.",
    f"P5 GPT-OSS: uniform quotas {fmt(p45['uniform quotas']['gpt_oss_mxfp4']['reduction'])}, utility {fmt(p45['utility quotas']['gpt_oss_mxfp4']['reduction'])}; the only material offline gain. Qwen quotas do not help.",
    "None of these change outputs, so the live latency gate decides adoption; m alone is not latency.",
    "Live fidelity for all these arms on GPT-OSS is done (identical outputs); timing repetitions are running.",
    sources=("offline_stage1 */p4/summary.json (primary/native)", "offline_stage1 */p5/summary.json (vs_global_seeded_mean_m)")), "p4_p5_exact_policies")

p6_rows = []
for model, label in zip(MODELS, LABELS):
    for key, name in (("B_lamNone_alpha0.1", "B swap, alpha 0.1"), ("A_lam0.1_alphaNone", "A bonus, λ 0.1")):
        v = p6[model]["cells"]["native"]["settings"][key]
        eligible = v["mass_mean"] >= .98 and v["mass_p5_mean"] >= .95
        p6_rows.append([label, name, fmt(v["vs_native_mean_m"]["reduction"]), fmt(v["mass_mean"]), fmt(v["mass_p5_mean"]),
                        "Yes" if eligible else "No"])
add(table_slide("RESULTS · P6 RESIDENT-BIASED ROUTING", "Routing toward cached experts drifts too far",
    ["Model", "Setting", "m reduction", "Mass kept, mean", "Mass kept, 5th pct", "Eligible"], p6_rows,
    [1.8, 2.4, 1.9, 2.2, 2.4, 1.633], "Fixed-logit screen, all 90 requests. Eligible needs ≥98% mean and ≥95% 5th-percentile original probability mass.",
    ("Verdict", "No setting is eligible on any model, so P6 has no finalist and P8 cannot be formed."),
    height=.42, size=1250),
    notes("Mass kept = share of the original router probability on the experts actually used; it measures how far routing moved.",
          "B with alpha 0.1 is the gentlest swap and still loses 2.5-4% of mass on average and 10-20% on the worst requests.",
          "A λ 0.1 (cache bonus) cuts m by 31-58% but keeps only 87-93% of mass: it effectively routes to whatever is cached.",
          "All 11 nonzero settings (A, B, C) fail on all three models; zero-change controls reproduced native routes exactly.",
          "Consequence: no P6 teacher forcing, and P8's 2 × 2 interaction is recorded as missing rather than replaced.",
          sources=("offline_stage1 */p6/summary.json, cells.native.settings",)))

p7r = chart_sources["p7_routing_screen"]
s = chart_slide("RESULTS · P7 ROUTING SCREEN", "Targeted 75% arms cut m far more than frequency", "p7_routing_screen",
                ("Caveat", "Fixed-logit routing screen; quality and merged weights are measured in Stage 2."))
add(s, notes(
    "Each request is evaluated with masks fitted without it (out-of-fold); the identity alias reproduces native exactly.",
    "R75 (keep the 75% most used) barely helps: about 2-5% fewer misses, consistent with last week's static result.",
    "Arm 1 removes experts with many misses and low saliency (gate weight × output norm); arm 2 clusters experts and aliases router IDs to one physical expert.",
    f"Arm 1: Qwen {fmt(p7r['arm 1: miss-targeted exclusion']['qwen_nvfp4']['reduction'])}, GPT-OSS {fmt(p7r['arm 1: miss-targeted exclusion']['gpt_oss_mxfp4']['reduction'])}, Gemma {fmt(p7r['arm 1: miss-targeted exclusion']['gemma_nvfp4']['reduction'])}.",
    f"Arm 2: Qwen {fmt(p7r['arm 2: cluster alias']['qwen_nvfp4']['reduction'])}, GPT-OSS {fmt(p7r['arm 2: cluster alias']['gpt_oss_mxfp4']['reduction'])}, Gemma {fmt(p7r['arm 2: cluster alias']['gemma_nvfp4']['reduction'])}.",
    "These change routing, so they must also pass the KL/top-1/task gates; the screen holds upstream hidden states fixed and is not a quality measure.",
    "F75 just deletes dropped accesses; it is a diagnostic, not a real model.",
    sources=("offline_stage1 */p7/summary.json",)), "p7_routing_screen")

add(table_slide("RESULTS · P7 FIRST QUALITY CHECK", "Arm 1 lowers m but changes outputs too much",
    ["Metric", "Gate", "GPT-OSS arm 1 (smoke)"],
    [["Teacher-forced m reduction", "≥10%", "18.5% (first history)"],
     ["Mean KL(baseline || arm 1), nats/token", "≤0.02 (upper 95% bound)", "0.37"],
     ["Top-1 agreement with baseline", "≥98% (lower 95% bound)", "74%"],
     ["D1 sequence KL, nats/token", "≤0.02 (upper 95% bound)", "0.57 and 0.97"]],
    [4.6, 3.8, 3.933], "Development smoke runs: two D0 histories and two D1 sequences; not accepted evidence.",
    ("Status", "Full 90-history teacher forcing, D1 and D2 are queued for all three models."),
    heading="Far outside the quality gate on GPT-OSS",
    body="Masking 8 of 32 experts per layer at decode changes about one next-token choice in four. Qwen and Gemma, and merged arm 2, are still to be measured.",
    height=.48),
    notes("Teacher forcing: feed the baseline's own token history, let the candidate compute its own hidden states, and compare full-vocabulary next-token distributions.",
          "Gate is 0.02 nats/token; the smoke shows about 20× that. Top-1 agreement 74% vs a 98% gate.",
          "The m reduction transfers from the offline screen (18.8% offline vs 18.5% teacher forced on the first history).",
          "These are smoke runs on two histories, not the accepted 90-history batch; treat as an early signal.",
          "GPT-OSS has only 32 experts per layer, so 25% removal is coarse; Qwen (256) and Gemma (128) may behave differently; no result yet.",
          "Arm 2 needs merged and requantized weights; the merge code reproduces the original weights exactly in identity mode (requantization check passed).",
          sources=("docs/EXPERIMENT_LOG.md, Stage 2 entry (development checks)",)))

p9_rows = []
for key, name in (("slru_0.25", "SLRU, 25% protected"), ("slru_0.5", "SLRU, 50% protected"),
                  ("pin_0.1", "Pin 10% of slots"), ("pin_0.25", "Pin 25% of slots")):
    p9_rows.append([name, *[fmt(p9[m]["comparisons"][key]["vs_lru_mean_m"]["reduction"]) for m in MODELS]])
p9_rows.append(["Prefill seeding, all tokens", *[fmt(seed[m]["arms"]["seed_whole"]["vs_native_mean_m"]["reduction"]) for m in MODELS]])
p9_rows.append(["Prefill seeding, tokens 0-63", *[fmt(seed[m]["arms"]["seed_whole"]["vs_native_mean_m_early"]["reduction"]) for m in MODELS]])
p9_rows.append(["Seed copies per request", *[f"{seed[m]['arms']['seed_whole']['seed_bytes_mean'] / 1e9:.1f} GB" for m in MODELS]])
add(table_slide("RESULTS · P9 AND P1 SEEDING", "Replacement and seeding help only in a few cells",
    ["Arm (mean-m reduction)", *LABELS], p9_rows, [4.333, 2.667, 2.667, 2.666],
    "Offline replay. P9 against LRU; seeding against the empty native decode start. Positive = fewer misses.",
    ("Verdict", "GPT-OSS SLRU 50% awaits timing; Qwen seeding needs a timed live arm."),
    height=.44, size=1250),
    notes("P9 segmented LRU: new experts enter probation; a second hit promotes them to a protected segment that is evicted last.",
          f"GPT-OSS SLRU 50% gives {fmt(p9['gpt_oss_mxfp4']['comparisons']['slru_0.5']['vs_lru_mean_m']['reduction'])}: passes the m criteria, so its G2 decision is defer_timing (needs the live ≤1.02× timing bound).",
          "Pinning hurts or does nothing; Qwen and Gemma SLRU stay below 5%.",
          "P1 seeding: fill the empty decode cache with the most-routed experts from the prompt; whole-prompt and last-64-token seeds coincide because D0 prompts are short.",
          "Seeding mainly removes early misses (Qwen −22% in the first 64 tokens), but costs real copies (2-3.6 GB per request) and TTFT, which offline replay does not charge.",
          "Only Qwen reaches the 5% offline bar for seeding, so only Qwen gets a timed live seeding arm.",
          sources=("offline_stage1 */p9/summary.json", "stage2_live_20261015T042628Z/p1_seed/*/summary.json")))

DESCRIBE = {"overlap_floor1": "Native q fraction (baseline)", "paper_floor1": "Paper q fraction", "fixed25_floor1": "Fixed 25% fetch",
            "reuse_uniform_floor1": "Reuse-Aware, uniform prior", "reuse_freq_floor1": "Reuse-Aware, frequency prior",
            "quota_uniform": "Uniform per-layer quotas", "quota_utility": "Utility per-layer quotas", "slru_0.5": "Segmented LRU, 50% protected"}
log = (STAGE2 / "queue.log").read_text()
live_rows, attempts = [], {}
for completion in sorted(glob.glob(str(STAGE2 / "runs_live/gpt_oss_mxfp4/fidelity/fidelity/*/completion.json"))):
    arm = re.match(r"\d+T\d+Z_(.+)_[0-9a-f]{6}$", Path(completion).parent.name).group(1)
    record = json.loads(Path(completion).read_text())
    attempts.setdefault(arm, []).append(record["status"])
    if record["status"] != "complete":
        continue
    lines = re.findall(rf"^gpt_oss_mxfp4/fidelity/{re.escape(arm)} \S+: tokens=(\d+) identical=(\w+) replay_mismatches=(\d+)$", log, re.M)
    accepted_lines = lines[-record["requests"]:]
    assert len(accepted_lines) == record["requests"] == record["outputs_identical"] == 9
    live_rows.append([arm, DESCRIBE[arm], f"{record['outputs_identical']}/{record['requests']}",
                      str(sum(int(mismatch) for _, _, mismatch in accepted_lines))])
assert len(live_rows) == 8
rejected = {arm: statuses.count("rejected") for arm, statuses in attempts.items() if "rejected" in statuses}
add(table_slide("RESULTS · LIVE CONFIRMATION", "Live kernels reproduce the simulator exactly",
    ["Arm", "Mechanism", "Outputs identical", "Event mismatches"], live_rows, [3.2, 4.4, 2.4, 2.333],
    "GPT-OSS fidelity runs: 9 cohort requests × 256 tokens per arm, against native captures and the offline simulator.",
    ("Running", "Five paired fresh-server timing repetitions per arm; Gemma and Qwen follow."),
    height=.42, size=1200),
    notes("New live runtime: the GPU admission kernel gained per-layer quota and segmented-LRU modes, plus a decode-only P7 mask hook.",
          "Fidelity run = observed live serving; every decode event (m, q, admitted IDs, victims) is compared with the offline simulator: zero mismatches on all 8 arms.",
          "Outputs identical to the native capture for every request, so these arms really are exact (no output change).",
          "Rejected attempts (host swap gate), repeated in a fresh directory: " + ", ".join(f"{arm} x{count}" for arm, count in rejected.items()) + "; rejected batches are kept, never used.",
          "Timing: 5 repetitions, fresh server per arm, arm order shuffled per repetition, geometric latency ratios with request-cluster bootstrap.",
          "So the offline m results on the previous slides carry over to live serving for these arms; what remains is whether fewer misses turns into lower latency.",
          sources=("stage2_live_20261015T042628Z/runs_live/gpt_oss_mxfp4/fidelity/*/completion.json", "stage2_live_20261015T042628Z/queue.log")))

add(blocks("SUMMARY · STATUS AND NEXT STEPS", "What we know and what runs next", [
    ("Established", "Qualified baselines on three models, exact replay and teacher forcing, offline P3-P9 screens, and live kernels that match the simulator event for event."),
    ("Most promising", "GPT-OSS per-layer quotas (−9.9% m) and SLRU (−5.1%); Qwen prefill seeding (−6.8%). P7 arms cut m by 19-50% offline but must pass quality."),
    ("Next", "Live timing for every arm, P7 teacher forcing, D1 and D2 quality, merged-weight arm 2, then P8 closeout. Direction 1 starts as new GPUs become available."),
], ("Bottom line", "Exact cache policies give modest gains; the large m cuts so far change model outputs.")),
    notes("Direction 2 has passed all of Stage 0 and Stage 1 for all three models; Stage 2 is running unattended in a serial GPU queue.",
          "Exact wins are small and model-specific; no exact component is adopted until the live timing bound passes.",
          "Output-changing arms give big m cuts offline, but the first GPT-OSS quality check is far outside the gate.",
          "Arm 2 merging runs after the main queue on its own runtime copy; its requantization path reproduces the original weights exactly.",
          "Open items: D3 SWE-bench harness pinning; live Qwen seeding arm; P8 records a missing interaction because P6 has no finalist.",
          "Direction 1 needs access to a 4090/5090 or other GPUs; the repository already packages Stage 0-3.",
          sources=("docs/PLAN.md status", "docs/EXPERIMENT_LOG.md")))


# ---------------------------------------------------------------- package assembly
def serialize(element):
    return ET.tostring(element, encoding="utf-8", xml_declaration=True)


def strip_ids(slide):
    for node in slide.findall(".//p:cNvPr", NS):
        for ext in node.findall("a:extLst", NS):
            node.remove(ext)
    csld = slide.find("p:cSld", NS)
    for ext in csld.findall("p:extLst", NS):
        csld.remove(ext)
    for number, node in enumerate(slide.findall(".//p:cNvPr", NS), 1):
        node.set("id", str(number))


dropped = re.compile(r"^ppt/(slides|notesSlides)/|^ppt/media/(fig_|static_)")
replacement = {name: data for name, data in original.items() if not dropped.match(name)}
presentation = ET.fromstring(original["ppt/presentation.xml"])
slide_ids = presentation.find("p:sldIdLst", NS)
for node in list(slide_ids):
    slide_ids.remove(node)
rels = ET.fromstring(original["ppt/_rels/presentation.xml.rels"])
for relationship in list(rels):
    if relationship.get("Type").endswith("/slide"):
        rels.remove(relationship)
content_types = ET.fromstring(original["[Content_Types].xml"])
for override in list(content_types):
    if override.get("PartName", "").startswith(("/ppt/slides/", "/ppt/notesSlides/")):
        content_types.remove(override)
notes_template = ET.fromstring(original["ppt/notesSlides/notesSlide18.xml"])
for i, (slide, note_text, figure) in enumerate(slides, 1):
    strip_ids(slide)
    relationship_id = f"rIdWeekSlide{i}"
    ET.SubElement(slide_ids, f"{{{NS['p']}}}sldId", {"id": str(300 + i), f"{{{NS['r']}}}id": relationship_id})
    ET.SubElement(rels, f"{{{REL}}}Relationship", Id=relationship_id, Type=NS["r"] + "/slide", Target=f"slides/slide{i}.xml")
    slide_rels = ET.fromstring(original["ppt/slides/_rels/slide1.xml.rels"])
    for relationship in list(slide_rels):
        if relationship.get("Type").endswith("/notesSlide"):
            relationship.set("Target", f"../notesSlides/notesSlide{i}.xml")
    if figure:
        ET.SubElement(slide_rels, f"{{{REL}}}Relationship", Id="rIdWeekFigure", Type=NS["r"] + "/image", Target=f"../media/{figure}.png")
        replacement[f"ppt/media/{figure}.png"] = (ASSETS / f"{figure}.png").read_bytes()
    note = deepcopy(notes_template)
    body_shape = next(shape for shape in tree(note) if shape.find("p:nvSpPr/p:nvPr/p:ph", NS) is not None
                      and shape.find("p:nvSpPr/p:nvPr/p:ph", NS).get("type") == "body")
    set_text(body_shape, note_text)
    note_rels = ET.fromstring(original["ppt/notesSlides/_rels/notesSlide18.xml.rels"])
    for relationship in note_rels:
        if relationship.get("Type").endswith("/slide"):
            relationship.set("Target", f"../slides/slide{i}.xml")
    replacement[f"ppt/slides/slide{i}.xml"] = serialize(slide)
    replacement[f"ppt/slides/_rels/slide{i}.xml.rels"] = serialize(slide_rels)
    replacement[f"ppt/notesSlides/notesSlide{i}.xml"] = serialize(note)
    replacement[f"ppt/notesSlides/_rels/notesSlide{i}.xml.rels"] = serialize(note_rels)
    for part, kind in ((f"/ppt/slides/slide{i}.xml", "slide"), (f"/ppt/notesSlides/notesSlide{i}.xml", "notesSlide")):
        ET.SubElement(content_types, f"{{{CT}}}Override", PartName=part,
                      ContentType=f"application/vnd.openxmlformats-officedocument.presentationml.{kind}+xml")
replacement["ppt/presentation.xml"] = serialize(presentation)
replacement["ppt/_rels/presentation.xml.rels"] = serialize(rels)
replacement["[Content_Types].xml"] = serialize(content_types)

app = ET.fromstring(original["docProps/app.xml"])
app_ns = {"ep": "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties",
          "vt": "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"}
app.find("ep:Slides", app_ns).text = str(len(slides))
app.find("ep:Notes", app_ns).text = str(len(slides))
parts = app.find("ep:TitlesOfParts/vt:vector", app_ns)
headings = app.find("ep:HeadingPairs/vt:vector", app_ns)
prefix_count = 0
if headings is not None:
    variants = list(headings)
    for i, variant in enumerate(variants[:-1]):
        label = variant.find("vt:lpstr", app_ns)
        if label is not None and label.text == "Slide Titles":
            variants[i + 1].find("vt:i4", app_ns).text = str(len(slides))
        elif label is not None:
            prefix_count += int(variants[i + 1].find("vt:i4", app_ns).text)
if parts is not None:
    for node in list(parts)[prefix_count:]:
        parts.remove(node)
    for slide, _, _ in slides:
        ET.SubElement(parts, f"{{{app_ns['vt']}}}lpstr").text = text_of(list(tree(slide))[4])
    parts.set("size", str(len(parts)))
replacement["docProps/app.xml"] = serialize(app)

with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED) as package:
    package.writestr("[Content_Types].xml", replacement.pop("[Content_Types].xml"))
    for name, data in replacement.items():
        package.writestr(name, data)
with ZipFile(OUTPUT) as package:
    assert package.testzip() is None
    for name in original:
        if name.startswith(("ppt/fonts/", "ppt/slideMasters/", "ppt/slideLayouts/", "ppt/theme/")):
            assert package.read(name) == original[name]

record = {"source": str(SOURCE.relative_to(ROOT)), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          "output": str(OUTPUT.relative_to(ROOT)), "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
          "slides": len(slides), "data_sources": SOURCES,
          "templates": "slide parts cloned from the 8 October deck; masters, layouts, theme and fonts byte-identical",
          "figures": "PNG in slides; PNG/SVG/PDF and plotted values (chart_source_rows.json) beside this script"}
(ASSETS / "edit_record.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
