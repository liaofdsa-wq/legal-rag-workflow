from __future__ import annotations

import json
import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any

import tempfile
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

from evaluator import evaluate_file, export_evaluation_result


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
EMBEDDINGS_ROOT = DATA_ROOT / "embeddings"
EVALUATION_RUNS_ROOT = DATA_ROOT / "evaluation_runs"
DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_OLLAMA_MODEL = "llama3.1:latest"
DEFAULT_BAD_CASE_THRESHOLD = 0.55
DEFAULT_DEDUP_FILE_NAME = "answers_rich_dedup.jsonl"
DEFAULT_REGISTRY_FILE_NAME = "run_registry.json"
EVALUATOR_OUTPUT_DIR_NAME = "evaluator_output"
APP_MODES = ("問答模式", "自動指標機")
EVALUATOR_VIEW_MODES = ("單一 run", "比較多個 runs")
AVAILABLE_MODES = ("hybrid", "leaf", "table", "all_nodes", "800200")
HYBRID_TEXT_OPTIONS = ("leaf", "all_nodes")


def find_available_port(start_port: int = 8501, max_attempts: int = 20) -> int:
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(
        f"No available port found between {start_port} and {start_port + max_attempts - 1}"
    )


def open_browser_delayed(port: int, delay_seconds: float = 1.5) -> None:
    url = f"http://localhost:{port}"

    def _open() -> None:
        try:
            if os.name == "nt":
                os.startfile(url)
            else:
                webbrowser.open(url)
        except Exception:
            webbrowser.open(url)

    threading.Timer(delay_seconds, _open).start()


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        get_script_run_ctx = None

    if get_script_run_ctx is None or get_script_run_ctx() is None:
        from streamlit.web import cli as stcli

        port = find_available_port()
        print(f"Opening Streamlit at http://localhost:{port}")
        open_browser_delayed(port)
        sys.argv = [
            "streamlit",
            "run",
            str(Path(__file__).resolve()),
            "--server.port",
            str(port),
        ]
        raise SystemExit(stcli.main())


def load_metadata(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@st.cache_resource(show_spinner=False)
def load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner=False)
def load_single_embedding_data(mode: str) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    if mode not in AVAILABLE_MODES:
        raise ValueError(f"Unsupported mode: {mode}")

    embedding_dir = EMBEDDINGS_ROOT / f"embedding_bge_m3_{mode}"
    embedding_path = embedding_dir / "embeddings.npy"
    metadata_path = embedding_dir / "metadata.jsonl"
    summary_path = embedding_dir / "embedding_summary.json"

    if not embedding_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {embedding_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    embeddings = np.load(embedding_path)
    metadata = load_metadata(metadata_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))

    if len(embeddings) != len(metadata):
        raise RuntimeError(
            f"Embedding count {len(embeddings)} does not match metadata count {len(metadata)}"
        )

    return embeddings, metadata, summary


@st.cache_resource(show_spinner=False)
def load_embedding_data(mode: str, hybrid_text_mode: str) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    if mode != "hybrid":
        return load_single_embedding_data(mode)

    if hybrid_text_mode not in HYBRID_TEXT_OPTIONS:
        raise ValueError(f"Unsupported hybrid text mode: {hybrid_text_mode}")

    text_embeddings, text_metadata, text_summary = load_single_embedding_data(hybrid_text_mode)
    table_embeddings, table_metadata, table_summary = load_single_embedding_data("table")

    merged_embeddings = np.concatenate([text_embeddings, table_embeddings], axis=0)
    merged_metadata = [*text_metadata, *table_metadata]
    merged_summary = {
        "mode": "hybrid",
        "hybrid_text_mode": hybrid_text_mode,
        "record_count": len(merged_metadata),
        "embedding_dim": int(merged_embeddings.shape[1]) if merged_embeddings.ndim == 2 else None,
        "doc_type_counts": {
            "all_node": sum(1 for row in merged_metadata if row.get("doc_type") == "all_node"),
            "leaf": sum(1 for row in merged_metadata if row.get("doc_type") == "leaf"),
            "table_chunk": sum(1 for row in merged_metadata if row.get("doc_type") == "table_chunk"),
        },
        "sources": {
            "text_mode": hybrid_text_mode,
            "text_metadata": text_summary.get("files", {}).get("metadata"),
            "table_metadata": table_summary.get("files", {}).get("metadata"),
        },
    }
    return merged_embeddings, merged_metadata, merged_summary


def cosine_search(query_embedding: np.ndarray, doc_embeddings: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    scores = doc_embeddings @ query_embedding
    top_k = min(top_k, len(scores))
    indices = np.argsort(-scores)[:top_k]
    return indices, scores[indices]


def load_ollama_models() -> list[str]:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []
    return [item["name"] for item in data.get("models", []) if item.get("name")]


def build_prompt(question: str, contexts: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for idx, item in enumerate(contexts, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Context {idx}]",
                    f"類型: {item.get('doc_type', '')}",
                    f"檔案: {item.get('file_name', '')}",
                    f"路徑: {item.get('path_text', '')}",
                    f"頁碼: {item.get('page_start', '')} - {item.get('page_end', '')}",
                    "內容:",
                    str(item.get("text", "")),
                ]
            )
        )
    return f"""你是一個法規問答助理，只能根據使用者提供的法規檢索內容回答問題。

回答規則：
1. 只能依據提供的檢索內容作答，不可自行補充外部知識。
2. 若檢索內容不足以回答，請直接回答：
「抱歉，我們的法規庫中沒有相關資訊，無法提供答案。」
3. 回答必須完全使用繁體中文，不可使用簡體中文。
4. 請先直接回答問題，再列出你引用的依據重點。
5. 不要編造法條、主管機關名稱、期限、罰則或程序。
6. 若參考資料彼此內容不一致，請明確指出差異，不可自行推定正確答案。
7. 若問題與提供的法規資料無關，也請直接回答：
「抱歉，我們的法規庫中沒有相關資訊，無法提供答案。」
問題:
{question}

檢索內容:
{chr(10).join(blocks)}
"""


def generate_with_ollama(prompt: str, model_name: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model_name, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    if "response" not in data:
        raise RuntimeError(f"Unexpected Ollama response: {data}")
    return str(data["response"]).strip()


def render_table_location(payload: dict[str, Any]) -> str:
    return (
        f"{payload.get('table_id', '')} / "
        f"r{payload.get('row_index', '')} / "
        f"c{payload.get('col_index', '')} / "
        f"k{payload.get('chunk_index', '')}"
    )


def render_extra_info(item: dict[str, Any]) -> str:
    payload = item.get("payload", {})
    if item.get("doc_type") == "fixed_chunk_800_200":
        return "\n".join(
            [
                f"file_name: {item.get('file_name', '')}",
                f"chunk_index: {payload.get('chunk_index', '')}",
                f"char_range: {payload.get('char_start', '')} - {payload.get('char_end', '')}",
                f"page_range: {item.get('page_start', '')} - {item.get('page_end', '')}",
            ]
        )
    if item.get("doc_type") == "table_chunk":
        return "\n".join(
            [
                f"檔案: {payload.get('file_name', '')}",
                f"掛載路徑: {payload.get('under_path_key', '')}",
                f"表格定位: {render_table_location(payload)}",
                f"頁碼: {item.get('page_start', '')} - {item.get('page_end', '')}",
                f"原始 cell: {payload.get('original_cell_text', '')}",
            ]
        )
    if item.get("doc_type") == "all_node":
        return "\n".join(
            [
                f"檔案: {item.get('file_name', '')}",
                f"節點名稱: {payload.get('node_name', '')}",
                f"path_key: {payload.get('path_key', '')}",
                f"路徑文字: {item.get('path_text', '')}",
                f"頁碼: {item.get('page_start', '')} - {item.get('page_end', '')}",
            ]
        )

    context_chain = payload.get("context_chain", [])
    title_chain = " > ".join(
        part
        for row in context_chain
        if isinstance(row, dict)
        for part in [str(row.get("node_name", "")).strip()]
        if part
    )
    return "\n".join(
        [
            f"檔案: {item.get('file_name', '')}",
            f"上下文鏈: {title_chain}",
            f"path_key: {payload.get('path_key', '')}",
            f"頁碼: {item.get('page_start', '')} - {item.get('page_end', '')}",
        ]
    )


def run_search(
    question: str,
    model: SentenceTransformer,
    doc_embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    query_embedding = model.encode(
        [question.strip()],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(np.float32)
    indices, scores = cosine_search(query_embedding, doc_embeddings, top_k)

    results: list[dict[str, Any]] = []
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        item = metadata[int(idx)]
        results.append({"rank": rank, "score": float(score), **item})
    return results


def resolve_registry_path() -> Path:
    return EVALUATION_RUNS_ROOT / DEFAULT_REGISTRY_FILE_NAME


def list_evaluation_runs() -> list[Path]:
    if not EVALUATION_RUNS_ROOT.exists():
        return []
    return sorted(
        [
            path
            for path in EVALUATION_RUNS_ROOT.iterdir()
            if path.is_dir()
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def resolve_run_input_path(run_dir: Path) -> Path:
    return run_dir / DEFAULT_DEDUP_FILE_NAME


def resolve_run_output_dir(run_dir: Path) -> Path:
    return run_dir / EVALUATOR_OUTPUT_DIR_NAME


def evaluator_output_paths(run_dir: Path) -> dict[str, Path]:
    output_dir = resolve_run_output_dir(run_dir)
    return {
        "summary": output_dir / "summary.json",
        "per_question_metrics": output_dir / "per_question_metrics.csv",
        "bad_cases": output_dir / "bad_cases.json",
    }


def has_evaluator_output(run_dir: Path) -> bool:
    return all(path.exists() for path in evaluator_output_paths(run_dir).values())


def load_run_registry() -> dict[str, Any]:
    registry_path = resolve_registry_path()
    if not registry_path.exists():
        return {"version": 1, "runs": []}
    return json.loads(registry_path.read_text(encoding="utf-8-sig"))


def load_run_metadata(run_dir: Path) -> dict[str, Any]:
    input_path = resolve_run_input_path(run_dir)
    if not input_path.exists():
        return {}
    with input_path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            return {
                "retrieval": row.get("retrieval", {}),
                "generation": row.get("generation", {}),
            }
    return {}


def infer_run_config_from_existing_run(run_dir: Path) -> dict[str, Any]:
    metadata = load_run_metadata(run_dir)
    retrieval = metadata.get("retrieval", {})
    generation = metadata.get("generation", {})
    return {
        "run_name": run_dir.name,
        "mode": retrieval.get("mode"),
        "hybrid_text_mode": retrieval.get("hybrid_text_mode"),
        "top_k": retrieval.get("top_k"),
        "ollama_model": generation.get("model"),
        "prompt_context_count": generation.get("prompt_context_count"),
        "description": "由既有 run 自動推斷",
    }


def determine_run_status(run_dir: Path | None) -> str:
    if run_dir is None:
        return "未執行"
    if has_evaluator_output(run_dir):
        return "evaluator ready"
    if resolve_run_input_path(run_dir).exists():
        return "已完成"
    return "已建立"


def build_run_catalog() -> list[dict[str, Any]]:
    registry = load_run_registry()
    registry_rows = registry.get("runs", [])

    existing_dirs = {run_dir.name: run_dir for run_dir in list_evaluation_runs()}
    catalog: list[dict[str, Any]] = []
    seen: set[str] = set()

    # 先把 registry 裡規劃好的版本都放進 catalog
    for row in registry_rows:
        run_name = str(row.get("run_name", "")).strip()
        if not run_name:
            continue

        run_dir = existing_dirs.get(run_name)

        catalog.append(
            {
                "run_name": run_name,
                "mode": row.get("mode"),
                "hybrid_text_mode": row.get("hybrid_text_mode"),
                "top_k": row.get("top_k"),
                "ollama_model": row.get("ollama_model"),
                "prompt_context_count": row.get("prompt_context_count"),
                "description": row.get("description", ""),
                "status": determine_run_status(run_dir),
                "run_dir": run_dir,
                "registry_source": "planned",
            }
        )
        seen.add(run_name)

    # 再把那些已存在資料夾、但沒有寫進 registry 的 run 補進來
    for run_name, run_dir in existing_dirs.items():
        if run_name in seen:
            continue

        inferred = infer_run_config_from_existing_run(run_dir)
        catalog.append(
            {
                **inferred,
                "status": determine_run_status(run_dir),
                "run_dir": run_dir,
                "registry_source": "existing_only",
            }
        )

    return catalog


def generate_run_commands(run_config: dict[str, Any]) -> str:
    run_name = run_config["run_name"]
    mode = run_config.get("mode") or "hybrid"
    top_k = run_config.get("top_k") or 5
    ollama_model = run_config.get("ollama_model") or DEFAULT_OLLAMA_MODEL
    prompt_context_count = run_config.get("prompt_context_count") or 3
    question_file = run_config.get("question_file") or "data/evaluation_inputs/capstone_questions_normalized.csv"
    embedding_model = run_config.get("embedding_model") or DEFAULT_MODEL

    root = r"C:\Users\ruby0\Desktop\EY_RAG\legal-rag-workflow-main\08_leaf_json_embedding_問答\法規資料_md_clean_leaf_json"
    question_path = question_file.replace("/", "\\")

    command = f"""$root = "{root}"
$questions = Join-Path $root "{question_path}"
$runDir = Join-Path $root "data\\evaluation_runs\\{run_name}"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

python (Join-Path $root "scripts\\04_batch_rag_answers.py") `
  --questions $questions `
  --output (Join-Path $runDir "answers.csv") `
  --rich-output (Join-Path $runDir "answers_rich.jsonl") `
  --embedding-model {embedding_model} `
  --mode {mode}"""

    if mode == "hybrid":
        hybrid_text_mode = run_config.get("hybrid_text_mode") or "leaf"
        command += f""" `
  --hybrid-text-mode {hybrid_text_mode}"""

    command += f""" `
  --top-k {top_k} `
  --ollama-model {ollama_model} `
  --prompt-context-count {prompt_context_count}

$rich = Join-Path $runDir "answers_rich.jsonl"
$dedup = Join-Path $runDir "answers_rich_dedup.jsonl"
$lines = Get-Content -LiteralPath $rich
$map = [ordered]@{{}}
foreach ($line in $lines) {{
  if ([string]::IsNullOrWhiteSpace($line)) {{ continue }}
  $obj = $line | ConvertFrom-Json
  $map[$obj.question] = $obj
}}
if (Test-Path $dedup) {{ Remove-Item -Force $dedup }}
foreach ($value in $map.Values) {{
  $value | ConvertTo-Json -Depth 8 -Compress | Add-Content -LiteralPath $dedup -Encoding UTF8
}}

python (Join-Path $root "scripts\\05_evaluate_rag_outputs.py") `
  --input (Join-Path $runDir "answers_rich_dedup.jsonl") `
  --output-dir (Join-Path $runDir "evaluator_output") `
  --model {embedding_model}
"""
    return command


def render_run_manager(catalog: list[dict[str, Any]]) -> None:
    st.subheader("Run Manager")

    manager_df = pd.DataFrame(
        [
            {
                "run_name": row["run_name"],
                "mode": row["mode"],
                "hybrid_text_mode": row["hybrid_text_mode"],
                "top_k": row["top_k"],
                "ollama_model": row["ollama_model"],
                "prompt_context_count": row["prompt_context_count"],
                "status": row["status"],
                "description": row["description"],
                "source": row["registry_source"],
            }
            for row in catalog
        ]
    )
    st.dataframe(manager_df, use_container_width=True, height=260)

    run_options = {row["run_name"]: row for row in catalog}
    selected_run_name = st.selectbox(
        "選擇一個版本查看設定 / 指令",
        options=list(run_options.keys()),
        index=0 if run_options else None,
        key="run_manager_select",
    )
    selected_run = run_options[selected_run_name]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"**目前狀態：** {selected_run['status']}")
        st.markdown(f"**版本說明：** {selected_run.get('description') or '(未填寫)'}")

    with col2:
        import subprocess
        import tempfile

        if st.button("🚀 Run this version", key=f"run_btn_{selected_run_name}"):
            st.info("開始執行，請稍候（可能需要幾分鐘）...")

            try:
                cmd = generate_run_commands(selected_run)

                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".ps1",
                    delete=False,
                    encoding="utf-8-sig"
                ) as temp_script:
                    temp_script.write(cmd)
                    temp_script_path = temp_script.name

                subprocess.run(
                    [
                        "powershell",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        temp_script_path,
                    ],
                    check=True,
                )

                st.success("✅ 執行完成！請重新整理頁面")

            except Exception as e:
                st.error(f"❌ 執行失敗：{e}")

    with st.expander("顯示執行指令", expanded=False):
        st.code(generate_run_commands(selected_run), language="powershell")

def load_existing_evaluator_output(run_dir: Path) -> dict[str, Any]:
    paths = evaluator_output_paths(run_dir)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8-sig"))
    per_question_df = pd.read_csv(paths["per_question_metrics"], encoding="utf-8-sig")
    bad_cases = json.loads(paths["bad_cases"].read_text(encoding="utf-8-sig"))
    return {
        "summary": summary,
        "per_question_df": per_question_df,
        "bad_cases": bad_cases,
    }


def compute_and_persist_evaluator_output(
    run_dir: Path,
    model_name: str,
    bad_case_threshold: float,
) -> dict[str, Any]:
    input_path = resolve_run_input_path(run_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing dedup input file: {input_path}")
    result = evaluate_file(
        input_path=input_path,
        model_name=model_name,
        bad_case_threshold=bad_case_threshold,
    )
    export_evaluation_result(result, resolve_run_output_dir(run_dir))
    return load_existing_evaluator_output(run_dir)


def get_run_evaluator_payload(
    run_dir: Path,
    model_name: str,
    bad_case_threshold: float,
    force_recompute: bool = False,
) -> dict[str, Any]:
    input_path = resolve_run_input_path(run_dir)
    if not input_path.exists():
        raise FileNotFoundError(
            "找不到 dedup rich output。"
            f" 請確認 run 目錄下存在 `{DEFAULT_DEDUP_FILE_NAME}`：{input_path}"
        )

    if force_recompute or not has_evaluator_output(run_dir):
        outputs = compute_and_persist_evaluator_output(
            run_dir=run_dir,
            model_name=model_name,
            bad_case_threshold=bad_case_threshold,
        )
        source = "剛重新計算完成"
    else:
        outputs = load_existing_evaluator_output(run_dir)
        source = "已讀取既有結果"

    return {
        **outputs,
        "source": source,
        "input_path": str(input_path),
        "metadata": load_run_metadata(run_dir),
    }


def render_metric_cards(metric_summary: dict[str, Any]) -> None:
    metrics_to_show = (
        ("response_relevancy", "Response Relevancy"),
        ("faithfulness", "Faithfulness"),
        ("context_precision", "Context Precision"),
        ("context_recall", "Context Recall"),
        ("overall_score", "Overall Score"),
    )
    cols = st.columns(len(metrics_to_show))
    for col, (metric_key, label) in zip(cols, metrics_to_show):
        mean_value = metric_summary[metric_key]["mean"]
        col.metric(label, f"{mean_value:.4f}" if mean_value is not None else "N/A")


def render_run_metadata(metadata: dict[str, Any]) -> None:
    retrieval = metadata.get("retrieval", {})
    generation = metadata.get("generation", {})
    rows = {
        "embedding_model": retrieval.get("embedding_model"),
        "retrieval_mode": retrieval.get("mode"),
        "hybrid_text_mode": retrieval.get("hybrid_text_mode"),
        "top_k": retrieval.get("top_k"),
        "ollama_model": generation.get("model"),
        "prompt_context_count": generation.get("prompt_context_count"),
    }
    st.json(rows)


def render_bad_cases(bad_cases: list[dict[str, Any]], limit: int = 5) -> None:
    st.subheader("Bad Cases")
    if not bad_cases:
        st.success("目前沒有低於 threshold 的 bad cases。")
        return

    for idx, row in enumerate(bad_cases[:limit], start=1):
        with st.expander(
            f"{idx}. overall={row['metrics'].get('overall_score', 0):.4f} | {row['question']}",
            expanded=idx == 1,
        ):
            st.write("Flags:", ", ".join(row.get("reasons", [])) or "None")
            st.write("Question:")
            st.code(row["question"], language="text")
            st.write("Generated Answer:")
            st.code(row.get("generated_answer_preview", ""), language="text")
            st.write("Metrics:")
            st.json(row["metrics"])


def render_downloads(per_question_df: pd.DataFrame, bad_cases: list[dict[str, Any]], run_name: str) -> None:
    st.subheader("下載")
    per_question_csv = per_question_df.to_csv(index=False)
    bad_cases_json = json.dumps(bad_cases, ensure_ascii=False, indent=2)
    col1, col2 = st.columns(2)
    col1.download_button(
        "下載 per_question_metrics.csv",
        data=per_question_csv.encode("utf-8-sig"),
        file_name=f"{run_name}_per_question_metrics.csv",
        mime="text/csv",
        use_container_width=True,
    )
    col2.download_button(
        "下載 bad_cases.json",
        data=bad_cases_json.encode("utf-8"),
        file_name=f"{run_name}_bad_cases.json",
        mime="application/json",
        use_container_width=True,
    )


def build_comparison_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    comparison_rows: list[dict[str, Any]] = []
    for row in rows:
        summary = row["summary"]
        metric_summary = summary.get("metric_summary", {})
        retrieval = row.get("metadata", {}).get("retrieval", {})
        generation = row.get("metadata", {}).get("generation", {})
        comparison_rows.append(
            {
                "run": row["run_name"],
                "source": row["source"],
                "case_count": summary.get("case_count"),
                "response_relevancy": metric_summary.get("response_relevancy", {}).get("mean"),
                "faithfulness": metric_summary.get("faithfulness", {}).get("mean"),
                "context_precision": metric_summary.get("context_precision", {}).get("mean"),
                "context_recall": metric_summary.get("context_recall", {}).get("mean"),
                "overall_score": metric_summary.get("overall_score", {}).get("mean"),
                "retrieval_mode": retrieval.get("mode"),
                "hybrid_text_mode": retrieval.get("hybrid_text_mode"),
                "top_k": retrieval.get("top_k"),
                "ollama_model": generation.get("model"),
                "prompt_context_count": generation.get("prompt_context_count"),
            }
        )
    return pd.DataFrame(comparison_rows)


def render_single_run_view(
    run_dir: Path,
    model_name: str,
    bad_case_threshold: float,
) -> None:
    st.write("預設輸入檔：")
    st.code(str(resolve_run_input_path(run_dir)), language="text")

    force_recompute = st.button("重新評估此 run", use_container_width=True)
    with st.spinner("載入 evaluator 結果中..."):
        try:
            payload = get_run_evaluator_payload(
                run_dir=run_dir,
                model_name=model_name,
                bad_case_threshold=bad_case_threshold,
                force_recompute=force_recompute,
            )
        except Exception as exc:
            st.error(f"載入 evaluator 失敗：{exc}")
            return

    summary = payload["summary"]
    metric_summary = summary["metric_summary"]
    per_question_df = payload["per_question_df"]
    bad_cases = payload["bad_cases"]

    st.info(f"目前資料來源：{payload['source']}")
    render_metric_cards(metric_summary)

    with st.expander("Run Metadata", expanded=False):
        render_run_metadata(payload["metadata"])

    with st.expander("Summary Details", expanded=False):
        st.json(summary)

    st.subheader("每題 Metrics")
    st.dataframe(per_question_df, use_container_width=True, height=360)

    render_bad_cases(bad_cases, limit=5)
    render_downloads(per_question_df, bad_cases, run_dir.name)


def render_comparison_view(
    run_dirs: list[Path],
    model_name: str,
    bad_case_threshold: float,
) -> None:
    run_options = {run_dir.name: run_dir for run_dir in run_dirs}
    selected_names = st.multiselect(
        "選擇要比較的 runs",
        options=list(run_options.keys()),
        default=list(run_options.keys())[: min(3, len(run_options))],
    )

    if not selected_names:
        st.info("請至少選擇一個 run。")
        return

    force_recompute = st.button("重新評估選取 runs", use_container_width=True)
    comparison_payloads: list[dict[str, Any]] = []

    with st.spinner("整理 comparison table 中..."):
        for run_name in selected_names:
            run_dir = run_options[run_name]
            try:
                payload = get_run_evaluator_payload(
                    run_dir=run_dir,
                    model_name=model_name,
                    bad_case_threshold=bad_case_threshold,
                    force_recompute=force_recompute,
                )
            except Exception as exc:
                comparison_payloads.append(
                    {
                        "run_name": run_name,
                        "source": f"載入失敗: {exc}",
                        "summary": {"case_count": None, "metric_summary": {}},
                        "metadata": {},
                    }
                )
                continue

            comparison_payloads.append(
                {
                    "run_name": run_name,
                    "source": payload["source"],
                    "summary": payload["summary"],
                    "metadata": payload["metadata"],
                }
            )

    comparison_df = build_comparison_table(comparison_payloads)
    st.subheader("Run Comparison")
    st.dataframe(comparison_df, use_container_width=True, height=320)


def render_evaluator_view(model_name: str) -> None:
    st.subheader("自動指標機")
    st.caption("先顯示版本規劃與狀態，再進行單一 run 或多 run 比較。")

    catalog = build_run_catalog()
    if not catalog:
        st.error(f"找不到 evaluation runs / registry：{EVALUATION_RUNS_ROOT}")
        return

    render_run_manager(catalog)

    available_run_dirs = [row["run_dir"] for row in catalog if row["run_dir"] is not None]
    if not available_run_dirs:
        st.warning("目前尚無已建立的 run，可先從 Run Manager 複製指令手動執行。")
        return

    view_mode = st.radio("檢視方式", options=list(EVALUATOR_VIEW_MODES), horizontal=True)
    bad_case_threshold = st.slider("Bad case threshold", 0.0, 1.0, DEFAULT_BAD_CASE_THRESHOLD, 0.01)

    if view_mode == "單一 run":
        run_options = {run_dir.name: run_dir for run_dir in available_run_dirs}
        selected_run_name = st.selectbox("選擇 evaluation run", options=list(run_options.keys()), index=0)      
        
        render_single_run_view(
            run_dir=run_options[selected_run_name],
            model_name=model_name,
            bad_case_threshold=bad_case_threshold,
        )
        return

    render_comparison_view(
        run_dirs=available_run_dirs,
        model_name=model_name,
        bad_case_threshold=bad_case_threshold,
    )


def render_search_view(
    model_name: str,
    mode: str,
    hybrid_text_mode: str,
    top_k: int,
    use_ollama: bool,
    ollama_model: str,
) -> None:
    try:
        doc_embeddings, metadata, summary = load_embedding_data(mode, hybrid_text_mode)
    except Exception as exc:
        st.error(f"載入 embedding 失敗: {exc}")
        st.stop()

    try:
        model = load_model(model_name)
    except Exception as exc:
        st.error(f"載入 embedding model 失敗: {exc}")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("文件數", f"{len(metadata):,}")
    col2.metric("向量維度", str(doc_embeddings.shape[1]))
    col3.metric("模式", summary.get("mode", mode))

    if mode == "hybrid":
        st.caption(f"Hybrid 文字來源模式: {hybrid_text_mode}")

    with st.expander("Embedding Summary"):
        st.json(summary)

    with st.form("search_form", clear_on_submit=False):
        st.text_area(
            "問題",
            key="question_input",
            height=100,
            placeholder="請輸入法規問題，例如：某項規範是否要求定期演練？",
        )
        submitted = st.form_submit_button("開始搜尋", type="primary")

    if submitted:
        submitted_query = st.session_state.question_input.strip()
        st.session_state.submitted_query = submitted_query
        st.session_state.search_question = submitted_query
        st.session_state.search_results = []
        st.session_state.ollama_answer = ""

        if submitted_query:
            with st.spinner("搜尋中..."):
                st.session_state.search_results = run_search(
                    submitted_query,
                    model,
                    doc_embeddings,
                    metadata,
                    top_k,
                )

            if use_ollama and st.session_state.search_results:
                with st.spinner("產生回答中..."):
                    try:
                        st.session_state.ollama_answer = generate_with_ollama(
                            #build_prompt(submitted_query, st.session_state.search_results[:3]),
                        # 有修======
                        build_prompt(
                            st.session_state.submitted_query,
                            st.session_state.search_results[:llm_context_k]
                        ),
                        # =========
                        ollama_model,
                    )
                        
                    except Exception as exc:
                        st.session_state.ollama_answer = ""
                        st.error(f"Ollama 回答失敗: {exc}")

    if st.session_state.search_results:
        st.subheader("檢索結果")
        st.caption(f"問題: {st.session_state.search_question}")
        query_key = st.session_state.submitted_query or "empty"
        for item in st.session_state.search_results:
            title = f"{item['rank']}. score={item['score']:.4f} | {item.get('file_name', '')} | {item.get('doc_type', '')}"
            source_key = str(item.get("source_id", item["rank"]))
            widget_key = f"{query_key}_{item['rank']}_{source_key}"
            with st.expander(title, expanded=item["rank"] == 1):
                st.text_area(
                    f"內容 #{item['rank']}",
                    value=str(item.get("text", "")),
                    height=180,
                    key=f"emb_{widget_key}",
                )
                st.text_area(
                    f"中繼資料 #{item['rank']}",
                    value=render_extra_info(item),
                    height=180,
                    key=f"meta_{widget_key}",
                )

    if st.session_state.ollama_answer:
        st.subheader("Ollama 回答")
        st.write(st.session_state.ollama_answer)


if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "search_question" not in st.session_state:
    st.session_state.search_question = ""
if "ollama_answer" not in st.session_state:
    st.session_state.ollama_answer = ""
if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "submitted_query" not in st.session_state:
    st.session_state.submitted_query = ""


st.set_page_config(page_title="法規檢索與自動指標機", layout="wide")
st.title("法規檢索與自動指標機")
st.caption("同一個 Streamlit UI 內提供問答模式、Run Manager 與 Evaluation Mode。")

app_mode = st.sidebar.selectbox("功能模式", options=list(APP_MODES), index=0)

with st.sidebar:
    model_name = st.text_input("Embedding 模型", value=DEFAULT_MODEL)
    if app_mode == "問答模式":
        mode = st.selectbox("Embedding 模式", options=list(AVAILABLE_MODES), index=0)
        hybrid_text_mode = st.selectbox(
            "Hybrid 文字來源",
            options=list(HYBRID_TEXT_OPTIONS),
            index=0,
            disabled=mode != "hybrid",
        )
        top_k = st.slider("Top K", 1, 20, 5)
        # 有修======
        llm_context_k = st.slider("Ollama 使用前幾筆結果", 1, 10, min(3, top_k))
        # ==========
        use_ollama = st.checkbox("使用 Ollama 產生回答", value=False)
        ollama_models = load_ollama_models()
        if ollama_models:
            default_model = DEFAULT_OLLAMA_MODEL if DEFAULT_OLLAMA_MODEL in ollama_models else ollama_models[0]
            ollama_model = st.selectbox(
                "Ollama 模型",
                ollama_models,
                index=ollama_models.index(default_model),
                disabled=not use_ollama,
            )
        else:
            ollama_model = st.text_input("Ollama 模型", value=DEFAULT_OLLAMA_MODEL, disabled=not use_ollama)
    else:
        mode = "hybrid"
        hybrid_text_mode = "leaf"
        top_k = 5
        use_ollama = False
        ollama_model = DEFAULT_OLLAMA_MODEL

if app_mode == "自動指標機":
    render_evaluator_view(model_name)
else:
    render_search_view(
        model_name=model_name,
        mode=mode,
        hybrid_text_mode=hybrid_text_mode,
        top_k=top_k,
        use_ollama=use_ollama,
        ollama_model=ollama_model,
    )
