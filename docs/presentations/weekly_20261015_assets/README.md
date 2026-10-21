# Weekly deck - 15 October 2026

Deck: [FreeToken_Weekly_Progress_15-10-2026.pptx](../FreeToken_Weekly_Progress_15-10-2026.pptx), 23 slides.

- Slides 1-2: last-week context and the two directions, same flow as the 8 October deck.
- Slides 3-8: Direction 1, the planned cross-architecture runtime-aware q study (exp_plan.pdf pages 3-4 only). Nothing is collected beyond the RTX 4070 reference cell.
- Slides 9-14: Direction 2, the m-reduction study plan (P1-P9, datasets, stages and gates) from `docs/PLAN.md`.
- Slides 15-23: Direction 2 results that exist: Stage 0 qualification, Stage 1 offline P3-P9 (all three models), P1 seeding replay, GPT-OSS live fidelity, and the GPT-OSS P7 arm 1 smoke (development evidence, labelled as such).

Every slide clones a template slide part from the 8 October deck (title block, rule, heading/body blocks, numbered items, table, figure, callout); masters, layouts, theme and embedded fonts are byte-identical. Speaker notes are bullet pointers plus a Sources line. Charts use the previous palette; `chart_source_rows.json` holds every plotted value and `edit_record.json` the source/output hashes.

Previews (`preview_slide*.png`, `contact_sheet.png`) are Pillow authoring previews with the embedded fonts, not Office renders.

Rebuild from the repository root:

    uv run --no-sync python docs/presentations/weekly_20261015_assets/build_weekly_20261015.py
    uv run --no-sync python docs/presentations/weekly_20261015_assets/preview_slides.py

The live-fidelity table reads the running Stage 2 study; a rebuild after more runs complete still lists only the accepted GPT-OSS fidelity batches.
