# Agent Notes

This package is organized as a fixed pipeline. Do not infer alternate input folders.

## Folder Contract

```text
01_原始PDF/法規資料
02_PDF轉Markdown程式
03_Markdown生成區/法規資料_md
04_Markdown精修區/法規資料_md
05_結構化程式
06_CleanTree生成區/法規資料_md_clean
07_CleanTree精修區/法規資料_md_clean
08_leaf_json_embedding_問答/法規資料_md_clean_leaf_json
```

Human-facing documentation should stay only in the root `README.md`.
Use `AGENTS.md` for technical agent instructions.

## Read / Write Rules

- `02_PDF轉Markdown程式/提md.py`
  - Reads `01_原始PDF/法規資料`
  - Writes `03_Markdown生成區/法規資料_md`
- `05_結構化程式/分類前處理_codex.py`
  - Reads `04_Markdown精修區/法規資料_md`
  - Rebuilds `06_CleanTree生成區/法規資料_md_clean`
- `08_leaf_json_embedding_問答/法規資料_md_clean_leaf_json/scripts/01_build_structured_json.py`
  - Reads `07_CleanTree精修區/法規資料_md_clean`
  - Writes `08_leaf_json_embedding_問答/法規資料_md_clean_leaf_json/data/json` and `data/summary`
- `08_leaf_json_embedding_問答/法規資料_md_clean_leaf_json/scripts/02_embed_structured_modes.py`
  - Reads `data/summary`
  - Writes `data/embeddings/embedding_bge_m3_<mode>`
  - Mode is selected by editing `DEFAULT_MODE` in the script.
- `08_leaf_json_embedding_問答/法規資料_md_clean_leaf_json/scripts/03_embed_fixed_800_200_baseline.py`
  - Reads `04_Markdown精修區/法規資料_md`
  - Writes `data/embeddings/embedding_bge_m3_800200`
  - This is independent of `01_build_structured_json.py` and does not read `data/summary`.

## Main Script Order

1. Run `02_PDF轉Markdown程式/提md.py` only when source PDFs changed.
2. Manually sync selected generated markdown from `03` into `04`.
3. Run `05_結構化程式/分類前處理_codex.py`.
4. Manually sync selected generated clean/tree/tables from `06` into `07`.
5. Run `08.../scripts/01_build_structured_json.py`.
6. Run `08.../scripts/02_embed_structured_modes.py` for structured embeddings.
7. Optionally run `08.../scripts/03_embed_fixed_800_200_baseline.py` for the fixed-size baseline.
8. Run `08.../app.py` or `08.../scripts/04_batch_rag_answers.py`.

## Embedding Modes

`01_build_structured_json.py` has no mode switch. It always creates all structured JSON:

- `data/summary/all_nodes.json`
- `data/summary/all_leaf_nodes.json`
- `data/summary/all_table_cells.json`
- `data/summary/all_table_chunks.json`
- `data/summary/file_summary.json`

`02_embed_structured_modes.py` is the structured embedding mode switch. Set `DEFAULT_MODE` before running:

- `all_nodes`: embeds every tree node from `all_nodes.json`.
- `leaf`: embeds leaf nodes from `all_leaf_nodes.json`; text includes ancestor context.
- `table`: embeds table chunks from `all_table_chunks.json`.
- `hybrid`: embeds text nodes plus table chunks.

The output folder is:

```text
08_leaf_json_embedding_問答/法規資料_md_clean_leaf_json/data/embeddings/embedding_bge_m3_<mode>
```

`03_embed_fixed_800_200_baseline.py` is a separate baseline:

- Reads `04_Markdown精修區/法規資料_md` directly.
- Splits text into 800-character chunks with 200-character overlap.
- Writes `data/embeddings/embedding_bge_m3_800200`.
- Does not require `data/summary`, leaf JSON, tree, or table chunks.

## Cautions

- Do not edit generated zones directly unless the user explicitly asks.
- Do not recreate nested README files. Keep user-facing docs in root `README.md`.
- `02_embed_structured_modes.py` uses top-level constants, especially `DEFAULT_MODE`.
- `分類前處理_codex.py` deletes and rebuilds `06_CleanTree生成區/法規資料_md_clean`.
