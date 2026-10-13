# Static pruning slides - 8 October 2026

Updated deck: [8 October presentation](../FreeToken_Weekly_Progress_08-10-2026_with_static_pruning.pptx).

- Slides 3-10: completed static routing/retention study and native conditional cache replay, including the repaired Gemma routing supplement.
- Slides 11-12: two empty slides with the original background, reserved for current actual inference results.
- Original slide 3 now appears at position 13. All 20 original slide XML parts, original chart media, layouts, masters, themes and embedded fonts are unchanged.

The new slides clone the supplied deck's title, rule, body, table and callout templates. Charts use its blue/orange/green series palette, white graph panels, axes, grids and intervals. `chart_source_rows.json` holds every plotted data cell. Complete source paths and interpretation limits are in each slide's speaker notes. Graphs also have PNG, SVG and PDF exports here.

The static study used 90 independent requests/model, five grouped folds and 18 additional repeat captures/model. Coverage is routing coverage, not answer quality. R replay reroutes original recorded hidden states; it is not masked autoregressive inference or a serving speedup. The current 75% inference results are deliberately left for the two reserved slides after closeout.

`edit_record.json` records the source/output hashes and artifact checks. `contact_sheet.png` and the per-slide previews are authoring previews made with the original embedded fonts; they are not Office renders.

Rebuild from the repository root with `uv run --no-sync python docs/presentations/pruning_20261008_assets/build_static_pruning_slides.py`, then `uv run --no-sync python docs/presentations/pruning_20261008_assets/preview_slides.py`.
