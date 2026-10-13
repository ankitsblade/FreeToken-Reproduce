"""Insert the completed static study using the supplied deck's native templates."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ASSETS = Path(__file__).resolve().parent
SOURCE = ROOT / "docs/presentations/FreeToken_Weekly_Progress_08-10-2026 (1).pptx"
OUTPUT = ROOT / "docs/presentations/FreeToken_Weekly_Progress_08-10-2026_with_static_pruning.pptx"
STUDY = ROOT / "results/moe_pruning/study_20261006_v1"
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
WORKLOADS = ["math", "code", "incident"]
COLORS = ["#287DE0", "#FF6633", "#18B57F"]


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


summaries = read_csv(STUDY / "retention75/summary.csv") + read_csv(STUDY / "additional/gemma_v1/retention75/summary.csv")
primary = read_csv(STUDY / "closeout75/primary_summary.csv")
stability = read_csv(STUDY / "retention75/stability.csv") + read_csv(STUDY / "additional/gemma_v1/retention75/stability.csv")
replay = read_csv(STUDY / "replay75/summary.csv")
gemma_replay = read_csv(STUDY / "additional/gemma_rerouting_v2/native_controls75/R75_summary.csv")
locks = read_csv(STUDY / "closeout75/checkpoint_lock.csv")
with ZipFile(SOURCE) as package:
    original = {name: package.read(name) for name in package.namelist()}


def find_row(rows, **values):
    selected = [row for row in rows if all(row.get(key) == value for key, value in values.items())]
    if len(selected) != 1:
        raise ValueError(f"ambiguous source cell: {values}")
    return selected[0]


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.titlesize": 14,
                     "axes.titleweight": "bold", "axes.labelsize": 14,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#999999", "xtick.color": "#555555", "ytick.color": "#555555",
                     "savefig.facecolor": "#FCFCFC", "figure.facecolor": "#FCFCFC", "axes.facecolor": "#FCFCFC"})
chart_sources = {}


def finish_chart(fig, name):
    fig.savefig(ASSETS / (name + ".png"), dpi=180)
    fig.savefig(ASSETS / (name + ".svg"))
    fig.savefig(ASSETS / (name + ".pdf"))
    plt.close(fig)


fig, ax = plt.subplots(figsize=(10.7, 5.65))
x = np.arange(3)
source_rows = []
for j, workload in enumerate(WORKLOADS):
    rows = [find_row(primary, model=model, workload=workload) for model in MODELS]
    source_rows += rows
    means = np.array([float(row["heldout_mean"]) * 100 for row in rows])
    low = np.array([float(row["ci_low"]) * 100 for row in rows])
    high = np.array([float(row["ci_high"]) * 100 for row in rows])
    bars = ax.bar(x + (j - 1) * .24, means, .23, color=COLORS[j], label=workload,
                  yerr=np.array([means-low, high-means]), capsize=3, error_kw={"elinewidth": 1})
    for bar, value in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, value+1.2, f"{value:.2f}", ha="center", fontsize=10)
ax.axhline(98, color="#7A2331", linestyle="--", linewidth=1, label="98% access screen")
ax.set(xticks=x, xticklabels=LABELS, ylim=(90, 104), ylabel="Held-out access coverage (%)",
       title="Same-workload decode masks: 30 held-out requests per workload, 95% CI")
ax.grid(axis="y", color="#E5E2DC"); ax.set_axisbelow(True)
ax.legend(loc="upper left", ncol=4, fontsize=11, frameon=False)
fig.subplots_adjust(left=.09, right=.985, bottom=.12, top=.88)
finish_chart(fig, "static_same_workload")
chart_sources["static_same_workload"] = source_rows

fig, axes = plt.subplots(1, 3, figsize=(10.7, 5.65))
source_rows = []
for ax, model, label in zip(axes, MODELS, LABELS):
    matrix = np.zeros((4, 3))
    for i, pool in enumerate([*WORKLOADS, "mixed"]):
        for j, workload in enumerate(WORKLOADS):
            row = find_row(summaries, model=model, ranking_phase="decode", evaluation_phase="decode",
                           calibration_pool=pool, evaluation_workload=workload)
            source_rows.append(row)
            matrix[i, j] = float(row["mean"]) * 100
    ax.imshow(matrix, cmap="Blues", vmin=60, vmax=100, aspect="auto")
    for i in range(4):
        for j in range(3):
            ax.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center", fontsize=11,
                    color="white" if matrix[i,j] > 84 else "#1B2638")
    ax.set(xticks=range(3), xticklabels=WORKLOADS, yticks=range(4), yticklabels=[*WORKLOADS, "mixed"], title=label)
    ax.set_xlabel("Held-out workload")
axes[0].set_ylabel("Calibration pool")
fig.suptitle("Decode access coverage (%): masks fitted without the held-out request", fontsize=13, fontweight="bold", y=.96)
fig.subplots_adjust(left=.10, right=.985, bottom=.14, top=.86, wspace=.48)
finish_chart(fig, "static_workload_transfer")
chart_sources["static_workload_transfer"] = source_rows

fig, axes = plt.subplots(1, 3, figsize=(10.7, 5.65), sharey=True)
source_rows = []
for ax, model, label in zip(axes, MODELS, LABELS):
    for j, phase in enumerate(("decode", "prefill")):
        rows = [find_row(summaries, model=model, ranking_phase="decode", evaluation_phase=phase,
                         calibration_pool="mixed", evaluation_workload=workload) for workload in WORKLOADS]
        source_rows += rows
        means = np.array([float(row["mean"]) * 100 for row in rows])
        low = np.array([float(row["ci_low"]) * 100 for row in rows])
        high = np.array([float(row["ci_high"]) * 100 for row in rows])
        ax.bar(np.arange(3)+(j-.5)*.34, means, .32, label=phase, color=COLORS[j],
               yerr=np.array([means-low, high-means]), capsize=2, error_kw={"elinewidth": 1})
    ax.axhline(98, color="#7A2331", linestyle="--", linewidth=1)
    ax.set(title=label, xticks=range(3), xticklabels=WORKLOADS, ylim=(70, 102))
    ax.grid(axis="y", color="#E5E2DC"); ax.set_axisbelow(True)
axes[0].set_ylabel("Held-out access coverage (%)")
fig.legend(*axes[1].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5,.91), ncol=2, frameon=False, fontsize=11)
fig.suptitle("Decode-derived mixed mask: decode versus prefill coverage (95% CI)", fontsize=13, fontweight="bold", y=.96)
fig.subplots_adjust(left=.08, right=.985, bottom=.12, top=.81, wspace=.15)
finish_chart(fig, "static_prefill_transfer")
chart_sources["static_prefill_transfer"] = source_rows
(ASSETS / "chart_source_rows.json").write_text(json.dumps(chart_sources, indent=2)+"\n")


def parse(number):
    return ET.fromstring(original[f"ppt/slides/slide{number}.xml"])


def tree(slide):
    return slide.find("p:cSld/p:spTree", NS)


def text_of(shape):
    return "".join(node.text or "" for node in shape.findall(".//a:t", NS))


def text_shape(slide, starts):
    return next(shape for shape in tree(slide) if text_of(shape).startswith(starts))


def set_text(shape, value):
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
            style = deepcopy(rpr); style.tag = f"{{{NS['a']}}}rPr"; r.append(style)
        ET.SubElement(r, f"{{{NS['a']}}}t").text = line


def geometry(shape, x, y, w, h):
    xf = shape.find(".//a:xfrm", NS)
    if xf is None:
        xf = shape.find("p:xfrm", NS)
    xf.find("a:off", NS).attrib.update(x=str(round(x*EMU)), y=str(round(y*EMU)))
    xf.find("a:ext", NS).attrib.update(cx=str(round(w*EMU)), cy=str(round(h*EMU)))


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
    shapes = list(tree(parse(18)))[7:]
    for shape in shapes:
        copy = deepcopy(shape)
        if text_of(copy) == "Reading":
            set_text(copy, label)
        elif text_of(copy):
            set_text(copy, content)
        tree(slide).append(copy)


def blocks(label, title, values, footer):
    slide = base(label, title)
    for i, (heading, body) in enumerate(values):
        y = 1.98 + i*1.27
        append_text(slide, "Reproduction complete", heading, .5, y, 12.333, .3333)
        append_text(slide, "Read the paper", body, .5, y+.397, 12.333, .70)
    callout(slide, *footer)
    return slide


def table_slide(label, title, headers, rows, widths, caption, heading, body, footer):
    slide = base(label, title)
    frame = deepcopy(next(shape for shape in tree(parse(17)) if shape.tag.endswith("graphicFrame")))
    table = frame.find(".//a:tbl", NS)
    grid = table.find("a:tblGrid", NS)
    for node in list(grid):
        grid.remove(node)
    for width in widths:
        ET.SubElement(grid, f"{{{NS['a']}}}gridCol", w=str(round(width*EMU)))
    originals = table.findall("a:tr", NS)
    for row in originals:
        table.remove(row)
    height = .64
    for i, values in enumerate([headers, *rows]):
        seed = originals[0] if i == 0 else originals[1 + (i-1)%2]
        row = ET.Element(f"{{{NS['a']}}}tr", h=str(round(height*EMU)))
        for j, value in enumerate(values):
            cell = deepcopy(seed.findall("a:tc", NS)[min(j, len(seed.findall('a:tc', NS))-1)])
            proxy = ET.Element(f"{{{NS['p']}}}sp")
            cell_body = cell.find("a:txBody", NS)
            cell.remove(cell_body)
            cell_body.tag = f"{{{NS['p']}}}txBody"; proxy.append(cell_body)
            set_text(proxy, value)
            proxy.remove(cell_body); cell_body.tag = f"{{{NS['a']}}}txBody"; cell.insert(0, cell_body)
            row.append(cell)
        table.append(row)
    geometry(frame, .5, 1.92, sum(widths), height*(len(rows)+1))
    tree(slide).append(frame)
    end = 1.92 + height*(len(rows)+1)
    caption_shape = deepcopy(text_shape(parse(17), "Latency:"))
    set_text(caption_shape, caption); geometry(caption_shape, .5, end+.12, 12.333, .40); tree(slide).append(caption_shape)
    if heading:
        append_text(slide, "Reproduction complete", heading, .5, end+.64, 12.333, .3333)
        append_text(slide, "Read the paper", body, .5, end+1.03, 12.333, .70)
    callout(slide, *footer)
    return slide


new_media = {}


def chart_slide(label, title, name, footer):
    slide = base(label, title)
    pic = deepcopy(list(tree(parse(18)))[6])
    pic.find(".//a:blip", NS).set(f"{{{NS['r']}}}embed", "rIdStaticFigure")
    tree(slide).append(pic)
    callout(slide, *footer)
    new_media[name] = (ASSETS / (name+".png")).read_bytes()
    return slide, name


slides = []


def add(slide, notes, figure=None):
    slides.append((slide, notes, figure))


add(blocks("OBJECTIVE 2 · STATIC PRUNING", "Can a fixed expert set reduce missing work?", [
    ("Retain the most frequently accessed 75% per layer", "Fit task-specific and mixed expert rankings on calibration requests. Shared branches remain active; native top-k routing is recorded before masking."),
    ("Collect and evaluate held-out routing", "90 distinct requests per checkpoint: 30 math, 30 code, 30 incident. Five grouped folds; 18 extra captures of nine requests assess repeat sensitivity."),
    ("Separate coverage from systems effects", "Measure access coverage, retained routing weight, workload transfer and ranking stability. Replay cache demand with fixed and reduced cache budgets."),
], ("Scope", "This stage measured routing and conditional cache replay. Task quality and serving speed were unmeasured.")),
    "Completed static study before actual M75 autoregressive inference. Source: closeout75/report.md, protocol/protocol.json, retention75/summary.csv. 90 independent requests/model and 18 additional repeat captures/model; repeats are not independent tasks. Native routes were captured with complete checkpoints; no weights were physically removed. U/F/R are separate cache replay interventions, not the running inference study.")

config_rows = []
for model in MODELS:
    lock = find_row(locks, model=model)
    config_rows.append([lock["checkpoint"], lock["layers"], lock["experts_per_layer"]+" / "+lock["retained75_per_layer"], lock["native_top_k"], lock["cache_slots"]])
add(table_slide("EXPERIMENT · MODEL CONFIGURATIONS", "Three supported MoE configurations on one device",
    ["Checkpoint", "Layers", "Experts / retained", "Top k", "Cache slots"], config_rows,
    [4.90, 1.05, 2.45, 1.10, 2.833], "Retained experts are per layer. Checkpoint precision: Qwen/Gemma NVFP4; GPT-OSS MXFP4.",
    "RTX 4070 (12 GB) + Ryzen 5 8500G", "FreeToken hybrid execution; one request at a time; CUDA graphs disabled. Gemma used text-only mode, memory ratio 0.8 and a 1,280-token sequence limit.",
    ("Controls", "Exact revisions and cache budgets were frozen. Kimi was excluded; shared experts were not ranked.")),
    "Source: closeout75/checkpoint_lock.csv and qualified_resources.csv; model-specific protocol/configuration artifacts. Full checkpoint revisions: "+json.dumps(locks)+". Qwen/GPT primary v5; Qwen repeat allocator v6 separately qualified. Gemma cache 768, prefill 1024, max sequence 1280, memory ratio .8. Same physical hardware does not make model hidden states or quantization layouts identical.")

s, f = chart_slide("STATIC PRUNING · HELD-OUT COVERAGE", "Task-specific masks preserve most decode accesses", "static_same_workload",
                   ("Reading", "Gemma reaches 99.32-99.57% coverage. Access coverage alone does not establish answer quality."))
add(s, "Source: closeout75/primary_summary.csv. Same-workload decode-derived top-75% masks, five grouped folds; held-out request macro coverage, 30 requests/workload/model. Error bars are 95% request bootstrap intervals conditional on fitted masks (10,000 resamples), not between-fold intervals. The 98% line is an access screening threshold; passing it does not satisfy all activation gates. Y-axis starts at 90% to show differences. All source cells saved in chart_source_rows.json.", f)

s, f = chart_slide("STATIC PRUNING · WORKLOAD TRANSFER", "A task-specific mask can lose coverage on another task", "static_workload_transfer",
                   ("Transfer", "GPT-OSS incident mask: 97.7% on incident, 63.2% on math. Mixed rankings trade specialisation for coverage."))
add(s, "Sources: retention75/summary.csv and additional/gemma_v1/retention75/summary.csv. Each heatmap cell is held-out macro decode access coverage (%), 30 requests. Rows identify the mask calibration pool; columns identify held-out evaluation workload. Masks fitted without the held-out grouped fold. No causal cross-architecture comparison: tokenizations and generated routes differ. Blue color scale is fixed at 60-100% across panels. Exact cells in chart_source_rows.json.", f)

s, f = chart_slide("STATIC PRUNING · PREFILL TRANSFER", "Decode rankings transfer poorly to prefill", "static_prefill_transfer",
                   ("Phase shift", "GPT-OSS mixed mask retains about 74-76% of prefill accesses versus 93-95% of decode accesses."))
add(s, "Sources: retention75/summary.csv and additional/gemma_v1/retention75/summary.csv. Every mask is decode-derived and calibrated on the mixed pool. Held-out prefill and decode evaluated separately on 30 requests/workload. Error bars are request bootstrap 95% intervals conditional on fitted masks. The 98% screen is dashed; y-axis starts at 70%. Phase values are paired observations, not independent task samples. Exact source cells in chart_source_rows.json.", f)

stability_rows = []
for model, label in zip(MODELS, LABELS):
    rows = [row for row in stability if row["model"]==model and row["kind"]=="disjoint_halves" and row["phase"]=="decode" and row["pool"]=="mixed"]
    values = [np.mean([float(row[name]) for row in rows]) for name in ("jaccard", "random_reference", "excess_jaccard")]
    stability_rows.append([label, f"{values[0]:.3f}", f"{values[1]:.3f}", f"{values[2]:.3f}"])
add(table_slide("STATIC PRUNING · RANKING STABILITY", "Stable rankings can still omit important routes",
    ["Model", "Retained-set Jaccard", "Random reference", "Excess overlap"], stability_rows,
    [3.10, 3.20, 3.15, 2.883], "Mean per-layer Jaccard across disjoint calibration halves; mixed decode mask, 75% retained.",
    "All three clear the 0.10 excess-overlap screen", "Retained sets overlap more than random 75% masks. The layer/subset comparisons are dependent; high overlap does not guarantee coverage across tasks or phases.",
    ("Sensitivity", "18 additional captures per checkpoint assess repeat sensitivity; they do not add independent tasks.")),
    "Sources: retention75/stability.csv; additional/gemma_v1/retention75/stability.csv; closeout75/masking_decisions.json; repeatability75 completion records. Rows restricted to kind=disjoint_halves, phase=decode, pool=mixed. Means over dependent comparisons: Qwen 2000, GPT-OSS 1200, Gemma 1500. Random reference is empirical random_reference, not assumed exactly 0.6. Excess equals the averaged observed-minus-reference Jaccard. Do not assign these comparisons independent confidence intervals.")

cache_rows = []
for workload in WORKLOADS:
    conditions = [find_row(replay, model="qwen_nvfp4", pool="mixed", mode="R", workload=workload,
                           state="cold", policy="calibrated_hybrid", capacity=capacity) for capacity in ("fixed_bytes", "relative_fraction")]
    cache_rows.append(["Qwen / "+workload, f"{100*float(conditions[0]['relative_m_reduction']):.2f}%", f"{100*float(conditions[1]['relative_m_reduction']):.2f}%", "3 requests"])
g_fixed = find_row(gemma_replay, state="cold", policy="calibrated_hybrid", slots="768")
g_reduced = find_row(gemma_replay, state="cold", policy="calibrated_hybrid", slots="576")
cache_rows += [["GPT-OSS / each task", "0.00%", "0.00%", "3 requests/task"],
               ["Gemma / all tasks", f"{100*float(g_fixed['relative_m_reduction']):.2f}%", f"{100*float(g_reduced['relative_m_reduction']):.2f}%", "9 requests"]]
add(table_slide("STATIC PRUNING · NATIVE CACHE REPLAY", "Rerouting gives little benefit at the fixed cache budget",
    ["Model / workload", "Fixed-cache reduction", "75%-cache reduction", "Scope"], cache_rows,
    [3.25, 3.20, 3.25, 2.633], "Relative reduction in missing experts/event under cold calibrated-hybrid R replay. Negative = more misses.",
    "", "", ("Replay scope", "R reroutes at recorded original hidden states. These are demand estimates; latency was not measured.")),
    "Sources: replay75/summary.csv (Qwen/GPT R, mixed, cold, calibrated_hybrid); additional/gemma_rerouting_v2/native_controls75/R75_summary.csv (Gemma nine-request aggregate). Baseline is original fixed cache in both budget columns. Reduced slots: Qwen 1536 vs 2048, GPT-OSS 116 vs 154, Gemma 576 vs 768. U replays original routes; F deletes excluded original accesses and lowers active k; R recomputes native masked top-k on unchanged recorded hidden states and preserves native k. This table uses R only. GPT hybrid remains at m=4 with q=0; unconditional-fill diagnostics differ and are not interchangeable. Gemma's repaired router includes native projection, learned expert scaling and original dtypes; 100% eligibility weights reproduce exactly, all 12 native cold cache controls matched. Gemma hybrid fractions reused Qwen NVFP4 calibration, so no Gemma-specific optimal split is established. Cold and post-prefill coincide for Gemma because observed initial cache is empty. Gemma fixed delta m=-0.0138562 with paired 95% CI [-0.0201311,-0.0072763]. Nine requests/model are descriptive conditional replay evidence, not a broad serving benchmark.")

add(blocks("STATIC PRUNING · DECISION AND NEXT STEP", "No checkpoint passed every static activation check", [
    ("Coverage and stability answer different questions", "Require access-CI lower bound >= 98%, retained-weight-CI lower bound >= 99%, and protection for weak requests, layers and 64-token windows. Ranking stability passed."),
    ("The systems screen also failed", "Native R replay needed at least 5% fixed-cache miss reduction in every workload. Qwen reached 0.87-2.24%; GPT-OSS 0%; repaired Gemma averaged 0.46%."),
    ("Next: actual inference with 25% of experts excluded", "A separate autoregressive 75%-retention study is evaluating task quality and serving performance despite the failed static screens. The 60% and 50% ablations follow later."),
], ("Conclusion", "Static routing evidence did not qualify deployment. The current inference study has its own evaluation.")),
    "Sources: closeout75/masking_decisions.json, closeout75/report.md, additional/gemma_rerouting_v2/report.md, docs/experiments/pruning/protocol.md. Original static gates also require request p5>=.95, every workload/layer mean>=.95, every complete rolling64 window>=.90, disjoint mean excess Jaccard>=.10, and paired delta CI upper<0 with >=5% reduction in every workload. All three were skipped_by_gate in the original closeout; Gemma native R was unavailable then and subsequently repaired without changing the frozen original report. Gemma aggregate .46% is not a per-workload estimate. Current M75 exploratory inference is separately authorised; no current results are included on these slides. Physical checkpoint compaction and measured memory savings are not claimed. 60/50 retention is deferred until current 75% work is finished.")

for _ in range(2):
    slide = base("", "")
    for shape in list(tree(slide))[3:]:
        tree(slide).remove(shape)
    add(slide, "Reserved for the current actual 75%-retention inference study once its report and serving measurements are final. Intentionally empty.")


def serialize(element):
    return ET.tostring(element, encoding="utf-8", xml_declaration=True)


replacement = {}
presentation = ET.fromstring(original["ppt/presentation.xml"])
slide_ids = presentation.find("p:sldIdLst", NS)
old_ids = list(slide_ids)
rels = ET.fromstring(original["ppt/_rels/presentation.xml.rels"])
content_types = ET.fromstring(original["[Content_Types].xml"])
next_id = max(int(node.get("id")) for node in old_ids)+1
notes_template = ET.fromstring(original["ppt/notesSlides/notesSlide18.xml"])
new_ids = []
for i, (slide, notes, figure) in enumerate(slides, 21):
    all_ids = slide.findall(".//p:cNvPr", NS)
    for number, node in enumerate(all_ids, 1):
        node.set("id", str(number))
    relationship_id = f"rIdStaticSlide{i}"
    new_ids.append(ET.Element(f"{{{NS['p']}}}sldId", {"id": str(next_id+i-21), f"{{{NS['r']}}}id": relationship_id}))
    ET.SubElement(rels, f"{{{REL}}}Relationship", Id=relationship_id, Type=NS["r"]+"/slide", Target=f"slides/slide{i}.xml")
    slide_rels = ET.fromstring(original["ppt/slides/_rels/slide1.xml.rels"])
    for relationship in list(slide_rels):
        if relationship.get("Type").endswith("/notesSlide"):
            slide_rels.remove(relationship)
    ET.SubElement(slide_rels, f"{{{REL}}}Relationship", Id="rIdStaticNotes", Type=NS["r"]+"/notesSlide", Target=f"../notesSlides/notesSlide{i}.xml")
    if figure:
        ET.SubElement(slide_rels, f"{{{REL}}}Relationship", Id="rIdStaticFigure", Type=NS["r"]+"/image", Target=f"../media/{figure}.png")
    note = deepcopy(notes_template)
    body_shape = next(shape for shape in tree(note) if shape.find("p:nvSpPr/p:nvPr/p:ph", NS) is not None
                      and shape.find("p:nvSpPr/p:nvPr/p:ph", NS).get("type")=="body")
    set_text(body_shape, notes)
    note_rels = ET.fromstring(original["ppt/notesSlides/_rels/notesSlide18.xml.rels"])
    for relationship in note_rels:
        if relationship.get("Type").endswith("/slide"):
            relationship.set("Target", f"../slides/slide{i}.xml")
    replacement[f"ppt/slides/slide{i}.xml"] = serialize(slide)
    replacement[f"ppt/slides/_rels/slide{i}.xml.rels"] = serialize(slide_rels)
    replacement[f"ppt/notesSlides/notesSlide{i}.xml"] = serialize(note)
    replacement[f"ppt/notesSlides/_rels/notesSlide{i}.xml.rels"] = serialize(note_rels)
    for part, kind in ((f"/ppt/slides/slide{i}.xml", "slide"), (f"/ppt/notesSlides/notesSlide{i}.xml", "notesSlide")):
        ET.SubElement(content_types, f"{{{CT}}}Override", PartName=part, ContentType=f"application/vnd.openxmlformats-officedocument.presentationml.{kind}+xml")
for node in old_ids:
    slide_ids.remove(node)
for node in [*old_ids[:2], *new_ids, *old_ids[2:]]:
    slide_ids.append(node)
replacement["ppt/presentation.xml"] = serialize(presentation)
replacement["ppt/_rels/presentation.xml.rels"] = serialize(rels)
replacement["[Content_Types].xml"] = serialize(content_types)
for name, data in new_media.items():
    replacement[f"ppt/media/{name}.png"] = data

app = ET.fromstring(original["docProps/app.xml"])
app_ns = {"ep":"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties", "vt":"http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"}
app.find("ep:Slides", app_ns).text = str(len(old_ids)+len(slides))
app.find("ep:Notes", app_ns).text = str(sum(1 for name in original if name.startswith('ppt/notesSlides/notesSlide') and name.endswith('.xml'))+len(slides))
parts = app.find("ep:TitlesOfParts/vt:vector", app_ns)
headings = app.find("ep:HeadingPairs/vt:vector", app_ns)
prefix_count = 0
if headings is not None:
    variants = list(headings)
    for i, variant in enumerate(variants[:-1]):
        label = variant.find("vt:lpstr", app_ns)
        if label is not None and label.text == "Slide Titles":
            count = variants[i+1].find("vt:i4", app_ns)
            count.text = str(len(old_ids)+len(slides))
        elif label is not None:
            prefix_count += int(variants[i+1].find("vt:i4", app_ns).text)
if parts is not None:
    # The original metadata lists seventeen titles although the deck has twenty slides.
    for node in list(parts)[prefix_count:]:
        parts.remove(node)
    actual_slides = [parse(1), parse(2), *[slide for slide, _, _ in slides], *[parse(i) for i in range(3,21)]]
    for slide in actual_slides:
        text = text_of(list(tree(slide))[4]) if len(list(tree(slide))) > 4 else ""
        ET.SubElement(parts, f"{{{app_ns['vt']}}}lpstr").text = text
    parts.set("size", str(len(parts)))
replacement["docProps/app.xml"] = serialize(app)

with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED) as package:
    for name, data in original.items():
        package.writestr(name, replacement.pop(name, data))
    for name, data in replacement.items():
        package.writestr(name, data)
with ZipFile(OUTPUT) as package:
    assert package.testzip() is None
    for i in range(1, 21):
        name = f"ppt/slides/slide{i}.xml"
        assert package.read(name) == original[name]
    for name in original:
        if name.startswith(("ppt/fonts/", "ppt/media/", "ppt/slideMasters/", "ppt/slideLayouts/", "ppt/theme/")):
            assert package.read(name) == original[name]
    for i in (29, 30):
        assert not any(node.text for node in ET.fromstring(package.read(f"ppt/slides/slide{i}.xml")).findall(".//a:t", NS))

record = {"source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          "output": str(OUTPUT), "slides":30, "added_static_slides":8, "inserted_at":3,
          "blank_slide_positions":[11,12], "original_slide3_new_position":13,
          "original_slides_preserved_byte_for_byte":True,
          "source_notes_included":True, "figures":"PNG in slides; SVG/PDF/PNG and exact source cells alongside builder"}
(ASSETS / "edit_record.json").write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps(record, indent=2))
