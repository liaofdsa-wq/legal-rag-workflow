# 資料來源與使用說明

## 資料內容

本專案整理的是金融資訊安全、電子金融、資通安全、作業韌性、供應鏈風險、新興科技與相關內控規範資料。

資料夾中的原始資料位於：

```text
01_原始PDF/法規資料
```

這些 PDF 主要作為課程專題 / RAG 檢索展示資料使用。後續 Markdown、clean/tree、leaf JSON、embedding 都是由這批 PDF 轉換或衍生而來。

## 衍生資料

```text
03_Markdown生成區
```

由 `02_PDF轉Markdown程式/提md.py` 從 PDF 粗轉為 Markdown。

```text
04_Markdown手修區
```

人工修正後的 Markdown，作為結構化前處理的主要輸入。

```text
06_CleanTree生成區
```

由 `05_結構化程式/分類前處理_codex.py` 產生，包含 clean Markdown、tables、headings、structure、tree。

```text
07_CleanTree手修區
```

人工修正後的 clean Markdown、tables、tree，作為 leaf JSON 的主要輸入。

```text
08_leaf_json_embedding_問答/法規資料_md_clean_leaf_json/data
```

由 leaf JSON 與 embedding 腳本產生，供檢索和問答使用。

## Embedding

Embedding 預設使用：

```text
BAAI/bge-m3
```

已產生的 embedding 位於：

```text
08_leaf_json_embedding_問答/法規資料_md_clean_leaf_json/data/embeddings
```

包含 structured embedding 模式：

- `all_nodes`
- `leaf`
- `table`
- `hybrid`

以及固定長度 baseline：

- `800200`：每段 800 字，重疊 200 字。

## 問答模型

互動 App 和批次問答可以使用本機 Ollama 模型生成答案。預設或常用模型包含：

```text
qwen2.5:3b
llama3.1:latest
```

若只測試檢索結果，可以不啟動 Ollama；若要 LLM 生成回答，請先啟動 Ollama 並確認模型已下載。

## 使用限制

本整理版主要用於課程專題、研究展示與檢索流程驗證。若公開散布或商業使用，請自行確認原始法規資料來源、授權與引用規範。
