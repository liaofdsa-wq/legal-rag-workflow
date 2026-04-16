

# 簡介

這份資料夾照工作流排成 `01` 到 `08`。程式輸出到生成區，人工修改放精修區，後續程式直接讀精修區。若需節省時間，直接複製生成區檔案即可。

## 快速啟動

第一次使用現在資料夾先安裝依賴：

```powershell

pip install -r requirements.txt
```

如果只要看問答介面，已附 embedding，可以直接啟動：

```powershell
cd 08_leaf_json_embedding_問答\法規資料_md_clean_leaf_json
python app.py
```

若要讓 App 產生 LLM 回答，請先啟動 Ollama，並確認本機已有模型，例如 `qwen2.5:3b` 或 `llama3.1:latest`。如果只看檢索結果，可以先不用 Ollama。

## 流程順序(見 流程圖html)

```text
01_原始PDF
02_PDF轉Markdown程式
03_Markdown生成區
04_Markdown精修區
05_結構化程式
06_CleanTree生成區
07_CleanTree精修區
08_leaf_json_embedding_問答
```

## 怎麼跑

```powershell
cd C:\Users\user\Desktop\大三下\EY\法規資料_整理保留_20260412

# PDF -> Markdown，輸出到 03
python 02_PDF轉Markdown程式\提md.py

# 把 03 想採用的 Markdown 複製到 04，然後在 04 精修

# 精修 Markdown -> clean md / tables / tree，輸出到 06
python 05_結構化程式\分類前處理_codex.py

# 把 06 想採用的 clean/tree/tables 複製到 07，然後在 07 精修

# 精修 clean/tree/tables -> leaf JSON，輸出到 08 的 data
python 08_leaf_json_embedding_問答\法規資料_md_clean_leaf_json\scripts\01_build_structured_json.py

# 產生 embedding
cd 08_leaf_json_embedding_問答\法規資料_md_clean_leaf_json
python scripts\02_embed_structured_modes.py

# 可選：固定 800 字、重疊 200 字 baseline embedding
python scripts\03_embed_fixed_800_200_baseline.py

# 開問答 App
python app.py
```

## 08scripts 執行順序

`08_leaf_json_embedding_問答/法規資料_md_clean_leaf_json/scripts` 的主線順序是：

1. `01_build_structured_json.py`
2. `02_embed_structured_modes.py`
3. `04_batch_rag_answers.py` 或 `app.py`

可選 baseline：

1. `03_embed_fixed_800_200_baseline.py`
2. `app.py` 選 `800200` mode

## embedding 模式

### 結構化 embedding

結構化 embedding 用這支：

```text
08_leaf_json_embedding_問答/法規資料_md_clean_leaf_json/scripts/02_embed_structured_modes.py
```

它讀取：

```text
08_leaf_json_embedding_問答/法規資料_md_clean_leaf_json/data/summary
```

並輸出到：

```text
08_leaf_json_embedding_問答/法規資料_md_clean_leaf_json/data/embeddings/embedding_bge_m3_<mode>
```

模式由程式開頭的 `DEFAULT_MODE` 控制：

```python
DEFAULT_MODE = "all_nodes"
```

可用模式：

- `all_nodes`：把所有 tree 節點都拿去 embedding，包含章、節、條、款、項等各層。
- `leaf`：只拿最末端 leaf node embedding，並把上層 ancestor context 一起組進文字。
- `table`：只拿表格 chunk embedding，來源是 `all_table_chunks.json`。
- `hybrid`：同時包含文字節點與表格 chunk。

切換模式的方法是先改 `DEFAULT_MODE`，再執行：

```powershell
python scripts\02_embed_structured_modes.py
```

例如要產生 leaf embedding：

```python
DEFAULT_MODE = "leaf"
```

輸出資料夾會變成：

```text
data/embeddings/embedding_bge_m3_leaf
```

### 固定長度 baseline：800200

`800200` 用這支：

```text
08_leaf_json_embedding_問答/法規資料_md_clean_leaf_json/scripts/03_embed_fixed_800_200_baseline.py
```

它不讀 leaf JSON，也不讀 tree。它直接讀：

```text
04_Markdown精修區/法規資料_md
```

切段方式：

- 每段 800 字。
- 相鄰段落重疊 200 字。
- 實際步長是 600 字。

輸出到：

```text
08_leaf_json_embedding_問答/法規資料_md_clean_leaf_json/data/embeddings/embedding_bge_m3_800200
```

用途是當 baseline，用來比較「固定字數切段」和「法規結構化切段」的檢索效果。

## 資料來源

原始 PDF 放在：

```text
01_原始PDF/法規資料
```

衍生資料包含 Markdown、clean/tree、leaf JSON、embedding。更完整的資料來源與使用限制請看：

```text
DATA_SOURCES.md
```

## 精修位置

- 修 PDF 粗轉 Markdown：`04_Markdown精修區/法規資料_md`
- 修 clean Markdown、tree、tables：`07_CleanTree精修區/法規資料_md_clean`

不要直接改 `03_Markdown生成區` 或 `06_CleanTree生成區`，因為重跑程式會重建它們。
