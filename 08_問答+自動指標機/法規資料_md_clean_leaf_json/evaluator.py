from __future__ import annotations

import io
import json
import math
import re
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_BAD_CASE_THRESHOLD = 0.55
DEFAULT_CONTEXT_RELEVANCE_THRESHOLD = 0.52
CSV_ENCODING = "utf-8-sig"
QUESTION_KEYS = ("question", "問題")
GOLD_KEYS = ("gold_answer", "標準答案")
ANSWER_KEYS = ("generated_answer", "answer", "LLM回答", "LLM答案")
PATH_KEYS = ("retrieved_paths", "LLM回答引用路徑", "LLM答案引用路徑")
CONTEXT_KEYS = ("retrieved_contexts", "contexts", "contexts_json")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(part for item in value if (part := clean_text(item)))
    return str(value).replace("\r", " ").strip()


def split_sentences(text: str) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    parts = re.split(r"(?<=[。！？!?；;])\s+|(?<=[。！？!?；;])", cleaned)
    rows = [part.strip() for part in parts if part and part.strip()]
    return rows or [cleaned]


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    return float(np.dot(left, right))


def encode_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def first_present(record: dict[str, Any], candidates: tuple[str, ...]) -> Any:
    for key in candidates:
        if key in record:
            return record.get(key)
    return None


def normalize_contexts(raw_contexts: Any) -> list[dict[str, Any]]:
    if raw_contexts is None:
        return []
    if isinstance(raw_contexts, str):
        text = raw_contexts.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                return normalize_contexts(parsed)
            except json.JSONDecodeError:
                return [{"text": text}]
        return [{"text": text}]
    if isinstance(raw_contexts, list):
        normalized: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_contexts, start=1):
            if isinstance(item, dict):
                text = clean_text(item.get("text"))
                normalized.append({"rank": item.get("rank", idx), **item, "text": text})
            else:
                normalized.append({"rank": idx, "text": clean_text(item)})
        return [row for row in normalized if row.get("text")]
    return [{"text": clean_text(raw_contexts)}]


def normalize_record(raw_record: dict[str, Any]) -> dict[str, Any]:
    question = clean_text(first_present(raw_record, QUESTION_KEYS))
    gold_answer = clean_text(first_present(raw_record, GOLD_KEYS))
    generated_answer = clean_text(first_present(raw_record, ANSWER_KEYS))
    retrieved_paths = clean_text(first_present(raw_record, PATH_KEYS))
    retrieved_contexts = normalize_contexts(first_present(raw_record, CONTEXT_KEYS))

    record_id = clean_text(raw_record.get("id")) or question or f"row_{id(raw_record)}"
    normalized = {
        "id": record_id,
        "question": question,
        "gold_answer": gold_answer,
        "generated_answer": generated_answer,
        "retrieved_paths": retrieved_paths,
        "retrieved_contexts": retrieved_contexts,
        "metadata": {key: value for key, value in raw_record.items() if key not in set(QUESTION_KEYS + GOLD_KEYS + ANSWER_KEYS + PATH_KEYS + CONTEXT_KEYS)},
    }
    return normalized


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding=CSV_ENCODING) as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding=CSV_ENCODING))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("items"), list):
            return data["items"]
        return [data]
    raise ValueError(f"Unsupported JSON payload in {path}")


def load_csv(path: Path) -> list[dict[str, Any]]:
    return pd.read_csv(path, encoding=CSV_ENCODING).to_dict(orient="records")


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".jsonlines"}:
        rows = load_jsonl(path)
    elif suffix == ".json":
        rows = load_json(path)
    elif suffix == ".csv":
        rows = load_csv(path)
    else:
        raise ValueError(f"Unsupported input type: {path.suffix}")
    return [normalize_record(row) for row in rows]


def load_records_from_uploaded_file(uploaded_file: BinaryIO, file_name: str) -> list[dict[str, Any]]:
    suffix = Path(file_name).suffix.lower()
    raw = uploaded_file.read()
    text = raw.decode("utf-8-sig")
    if suffix in {".jsonl", ".jsonlines"}:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif suffix == ".json":
        data = json.loads(text)
        rows = data if isinstance(data, list) else data.get("items", [data])
    elif suffix == ".csv":
        rows = pd.read_csv(io.StringIO(text)).to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported upload type: {suffix}")
    return [normalize_record(row) for row in rows]


def score_response_relevancy(
    question_embedding: np.ndarray,
    answer_embedding: np.ndarray | None,
) -> float | None:
    if answer_embedding is None or question_embedding.size == 0:
        return None
    return cosine_similarity(question_embedding, answer_embedding)


def score_faithfulness(
    model: SentenceTransformer,
    answer: str,
    context_texts: list[str],
) -> tuple[float | None, list[dict[str, Any]]]:
    sentences = split_sentences(answer)
    if not sentences or not context_texts:
        return None, []

    sentence_embeddings = encode_texts(model, sentences)
    context_embeddings = encode_texts(model, context_texts)
    details: list[dict[str, Any]] = []
    scores: list[float] = []

    for sentence, sentence_embedding in zip(sentences, sentence_embeddings):
        similarities = context_embeddings @ sentence_embedding
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        scores.append(best_score)
        details.append(
            {
                "sentence": sentence,
                "best_context_rank": best_idx + 1,
                "best_context_score": best_score,
                "best_context_preview": context_texts[best_idx][:180],
            }
        )
    return safe_mean(scores), details


def context_relevance_flags(
    model: SentenceTransformer,
    reference_answer: str,
    context_texts: list[str],
    threshold: float = DEFAULT_CONTEXT_RELEVANCE_THRESHOLD,
) -> tuple[list[bool], list[float]]:
    if not reference_answer or not context_texts:
        return [], []
    reference_embedding = encode_texts(model, [reference_answer])[0]
    context_embeddings = encode_texts(model, context_texts)
    similarities = (context_embeddings @ reference_embedding).tolist()
    flags = [float(score) >= threshold for score in similarities]
    return flags, [float(score) for score in similarities]


def score_context_precision(
    relevance_flags: list[bool],
) -> float | None:
    if not relevance_flags:
        return None
    relevant_count = sum(1 for flag in relevance_flags if flag)
    if relevant_count == 0:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for idx, flag in enumerate(relevance_flags, start=1):
        if not flag:
            continue
        hits += 1
        precision_sum += hits / idx
    return float(precision_sum / relevant_count)


def score_context_recall(
    model: SentenceTransformer,
    reference_answer: str,
    context_texts: list[str],
) -> float | None:
    reference_sentences = split_sentences(reference_answer)
    if not reference_sentences or not context_texts:
        return None

    reference_embeddings = encode_texts(model, reference_sentences)
    context_embeddings = encode_texts(model, context_texts)
    scores: list[float] = []
    for sentence_embedding in reference_embeddings:
        similarities = context_embeddings @ sentence_embedding
        scores.append(float(np.max(similarities)))
    return safe_mean(scores)


def compute_case_metrics(model: SentenceTransformer, record: dict[str, Any]) -> dict[str, Any]:
    question = record["question"]
    gold_answer = record["gold_answer"]
    generated_answer = record["generated_answer"]
    contexts = record["retrieved_contexts"]
    context_texts = [clean_text(item.get("text")) for item in contexts if clean_text(item.get("text"))]

    question_embedding = encode_texts(model, [question])[0] if question else np.empty((0,), dtype=np.float32)
    answer_embedding = encode_texts(model, [generated_answer])[0] if generated_answer else None

    response_relevancy = score_response_relevancy(question_embedding, answer_embedding)
    faithfulness, faithfulness_details = score_faithfulness(model, generated_answer, context_texts)
    relevance_flags, relevance_scores = context_relevance_flags(model, gold_answer, context_texts)
    context_precision = score_context_precision(relevance_flags)
    context_recall = score_context_recall(model, gold_answer, context_texts)

    missing_inputs: list[str] = []
    if not generated_answer:
        missing_inputs.append("generated_answer")
    if not context_texts:
        missing_inputs.append("retrieved_contexts")
    if not gold_answer:
        missing_inputs.append("gold_answer")

    available_metrics = [
        score
        for score in (
            response_relevancy,
            faithfulness,
            context_precision,
            context_recall,
        )
        if score is not None
    ]
    overall_score = safe_mean(available_metrics)

    return {
        "id": record["id"],
        "question": question,
        "gold_answer": gold_answer,
        "generated_answer": generated_answer,
        "retrieved_paths": record["retrieved_paths"],
        "retrieved_context_count": len(context_texts),
        "metrics": {
            "response_relevancy": response_relevancy,
            "faithfulness": faithfulness,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "consistency": None,
            "overall_score": overall_score,
        },
        "metric_inputs": {
            "has_question": bool(question),
            "has_gold_answer": bool(gold_answer),
            "has_generated_answer": bool(generated_answer),
            "has_retrieved_contexts": bool(context_texts),
        },
        "metric_details": {
            "faithfulness": faithfulness_details,
            "context_precision": [
                {
                    "rank": idx,
                    "is_relevant_to_gold": flag,
                    "similarity_to_gold": score,
                    "context_preview": context_texts[idx - 1][:180],
                }
                for idx, (flag, score) in enumerate(zip(relevance_flags, relevance_scores), start=1)
            ],
        },
        "missing_inputs": missing_inputs,
        "retrieved_contexts": contexts,
        "metadata": record.get("metadata", {}),
    }


def pick_bad_cases(case_rows: list[dict[str, Any]], threshold: float = DEFAULT_BAD_CASE_THRESHOLD) -> list[dict[str, Any]]:
    bad_cases: list[dict[str, Any]] = []
    for row in case_rows:
        metrics = row["metrics"]
        reasons: list[str] = []
        for metric_name in ("response_relevancy", "faithfulness", "context_precision", "context_recall"):
            value = metrics.get(metric_name)
            if value is not None and value < threshold:
                reasons.append(f"{metric_name}<{threshold:.2f}")
        if row["missing_inputs"]:
            reasons.append("missing_inputs")
        if reasons:
            bad_cases.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "metrics": metrics,
                    "reasons": reasons,
                    "missing_inputs": row["missing_inputs"],
                    "retrieved_paths": row["retrieved_paths"],
                    "generated_answer_preview": row["generated_answer"][:280],
                }
            )

    bad_cases.sort(
        key=lambda row: (
            999.0 if row["metrics"].get("overall_score") is None else row["metrics"]["overall_score"],
            row["question"],
        )
    )
    return bad_cases


def build_summary(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = ("response_relevancy", "faithfulness", "context_precision", "context_recall", "consistency", "overall_score")
    metric_summary: dict[str, Any] = {}
    for metric_name in metric_names:
        values = [
            row["metrics"][metric_name]
            for row in case_rows
            if row["metrics"].get(metric_name) is not None
        ]
        metric_summary[metric_name] = {
            "mean": safe_mean(values),
            "count": len(values),
            "missing_count": len(case_rows) - len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    return {
        "case_count": len(case_rows),
        "metric_summary": metric_summary,
        "missing_input_counts": {
            "missing_gold_answer": sum(1 for row in case_rows if not row["metric_inputs"]["has_gold_answer"]),
            "missing_generated_answer": sum(1 for row in case_rows if not row["metric_inputs"]["has_generated_answer"]),
            "missing_retrieved_contexts": sum(1 for row in case_rows if not row["metric_inputs"]["has_retrieved_contexts"]),
        },
    }


def evaluate_records(
    records: list[dict[str, Any]],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    bad_case_threshold: float = DEFAULT_BAD_CASE_THRESHOLD,
) -> dict[str, Any]:
    model = SentenceTransformer(model_name)
    case_rows = [compute_case_metrics(model, record) for record in records]
    summary = build_summary(case_rows)
    bad_cases = pick_bad_cases(case_rows, threshold=bad_case_threshold)
    return {
        "summary": {
            **summary,
            "model_name": model_name,
            "bad_case_threshold": bad_case_threshold,
        },
        "per_question": case_rows,
        "bad_cases": bad_cases,
    }


def evaluate_file(
    input_path: Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    bad_case_threshold: float = DEFAULT_BAD_CASE_THRESHOLD,
) -> dict[str, Any]:
    records = load_records(input_path)
    result = evaluate_records(
        records,
        model_name=model_name,
        bad_case_threshold=bad_case_threshold,
    )
    result["summary"]["input_path"] = str(input_path)
    return result


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding=CSV_ENCODING)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=CSV_ENCODING) as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_evaluation_result(result: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    per_question_jsonl_path = output_dir / "per_question.jsonl"
    per_question_csv_path = output_dir / "per_question_metrics.csv"
    bad_cases_path = output_dir / "bad_cases.json"

    write_json(summary_path, result["summary"])
    write_jsonl(per_question_jsonl_path, result["per_question"])
    pd.DataFrame(
        [
            {
                "id": row["id"],
                "question": row["question"],
                "response_relevancy": row["metrics"]["response_relevancy"],
                "faithfulness": row["metrics"]["faithfulness"],
                "context_precision": row["metrics"]["context_precision"],
                "context_recall": row["metrics"]["context_recall"],
                "consistency": row["metrics"]["consistency"],
                "overall_score": row["metrics"]["overall_score"],
                "retrieved_context_count": row["retrieved_context_count"],
                "missing_inputs": " | ".join(row["missing_inputs"]),
            }
            for row in result["per_question"]
        ]
    ).to_csv(per_question_csv_path, index=False, encoding=CSV_ENCODING)
    write_json(bad_cases_path, result["bad_cases"])

    return {
        "summary": summary_path,
        "per_question_jsonl": per_question_jsonl_path,
        "per_question_csv": per_question_csv_path,
        "bad_cases": bad_cases_path,
    }
