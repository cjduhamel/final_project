#!/usr/bin/env python3
"""
Inference tool

Usage:
    python cite.py example_paper.json \
        --model-dir saved_model \
        --agg mean \
        --top-k 3 \
        --out-json example_paper.predicted.json \
        --out-dir example_paper_reports
"""

import argparse
import json
import os
import re
from typing import List, Dict, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Sentence splitting

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
def split_into_sentences(text: str) -> List[str]:
    """
    Very simple sentence splitter.
    
    """
    text = text.strip()
    if not text:
        return []   
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# Model loading
def load_model_and_tokenizer(model_dir: str, device: torch.device):
    """
    Load the trained model (in our case fine-tuned SciBERT classifier) and its tokenizer,
    plus inference config (threshold, max_length, label mapping).
    """
    print(f"Loading model from {model_dir} ...")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model.to(device)
    model.eval()

    infer_cfg_path = os.path.join(model_dir, "inference_config.json")
    with open(infer_cfg_path, "r") as f:
        infer_cfg = json.load(f)

    max_length = int(infer_cfg.get("max_length", 512))
    id2label = {int(k): v for k, v in infer_cfg["id2label"].items()}
    label2id = infer_cfg["label2id"]

    # We dont actually use the threshold for paragraphs anyways
    threshold = float(infer_cfg.get("threshold", 0.5))

    print("Loaded inference config:")
    print("  max_length:", max_length)
    print("  id2label:", id2label)
    print("  threshold (unused for ranking):", threshold)

    return model, tokenizer, max_length, id2label, label2id, threshold


# Text helpers

def build_ref_block(ref: Dict[str, Any]) -> str:
    """
    Build ref_block from a reference dict.
    """
    pieces = []

    for key in ["title", "ref_paper_title"]:
        v = ref.get(key, "")
        if isinstance(v, str) and v.strip():
            pieces.append(v.strip())

    for key in ["authors", "ref_paper_authors"]:
        v = ref.get(key, "")
        if isinstance(v, str) and v.strip():
            pieces.append(v.strip())

    for key in ["text", "ref_paper_text"]:
        v = ref.get(key, "")
        if isinstance(v, str) and v.strip():
            pieces.append(v.strip())

    return " ".join(pieces).strip()



# Inference helpers

def predict_probs_for_sentences(
    sentences: List[str],
    ref_block: str,
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
) -> List[float]:
    """
    For each sentence, return P(CITATION | sentence, ref_block).
    """
    probs = []
    if not sentences:
        return probs

    for s in sentences:
        s = s.strip()
        if not s:
            probs.append(0.0)
            continue

        inputs = tokenizer(
            s,
            ref_block,
            truncation="only_second",
            max_length=max_length,
            return_tensors="pt",
        )
        # move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # [1, 2]
            # softmax to get probs
            prob = torch.softmax(logits, dim=-1)[0, 1].item()  # class 1 = CITATION

        probs.append(float(prob))

    return probs


def aggregate_probs(probs: List[float], method: str = "mean") -> float:
    """
    Aggregate sentence-level probabilities into a paragraph score.
    we supposed the following 3 methods: mean (sentences in para added), max(highest sentence score), noisy_or.
    """
    if not probs:
        return 0.0

    p = np.array(probs, dtype=float)

    if method == "mean":
        return float(p.mean())
    elif method == "max":
        return float(p.max())
    elif method == "noisy_or":
        # 1 - product(1 - p_i)
        return float(1.0 - np.prod(1.0 - p))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def run_paragraph_scoring_for_ref(
    ref_idx: int,
    ref: Dict[str, Any],
    paragraphs: List[str],
    paragraph_sentences: List[List[str]],
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    agg_method: str,
    top_k: int,
):

    ref_block = build_ref_block(ref)
    gold_pars = ref.get("referenced_in_paragraphs", []) or []
    num_pars = len(paragraphs)

    sentence_probs_by_paragraph: List[List[float]] = []
    scores: List[float] = []

    for p_idx in range(num_pars):
        sents = paragraph_sentences[p_idx]
        probs = predict_probs_for_sentences(
            sents, ref_block, tokenizer, model, device, max_length
        )
        score = aggregate_probs(probs, method=agg_method)
        sentence_probs_by_paragraph.append(probs)
        scores.append(score)

    scores_arr = np.array(scores, dtype=float)
    # top-k paragraph indices sorted by score descin
    k = min(top_k, num_pars)
    ordering = np.argsort(-scores_arr)
    top_indices = ordering[:k].tolist()
    best_idx = int(top_indices[0]) if k > 0 else -1

    return {
        "ref_idx": ref_idx,
        "scores": scores_arr,
        "best_idx": best_idx,
        "top_indices": top_indices,
        "sentence_probs_by_paragraph": sentence_probs_by_paragraph,
        "gold_pars": gold_pars,
        "ref_block": ref_block,
    }



# Reporting (per-reference .txt)

def build_text_report_for_reference(
    ref_idx: int,
    ref: Dict[str, Any],
    paragraphs: List[str],
    paragraph_sentences: List[List[str]],
    scoring_result: Dict[str, Any],
    agg_method: str,
) -> str:
   
    lines: List[str] = []

    gold_pars = scoring_result["gold_pars"]
    scores = scoring_result["scores"]
    best_idx = scoring_result["best_idx"]
    top_indices = scoring_result["top_indices"]
    sentence_probs_by_paragraph = scoring_result["sentence_probs_by_paragraph"]

    title = ref.get("title", ref.get("ref_paper_title", "[no title]"))
    authors = ref.get("authors", ref.get("ref_paper_authors", "[no authors]"))
    text = ref.get("text", ref.get("ref_paper_text", "[no text]"))

    lines.append("=" * 80)
    lines.append(f"Reference #{ref_idx}")
    lines.append(f"Title:   {title}")
    lines.append(f"Authors: {authors}")
    lines.append(f"TEXT: {text}")
    lines.append(f"Gold paragraph indices: {gold_pars}")
    lines.append(f"Aggregation method: {agg_method}")
    lines.append("=" * 80)
    lines.append("")

    if best_idx >= 0:
        best_score = float(scores[best_idx])
        is_correct = bool(best_idx in gold_pars) if gold_pars else None
        lines.append(
            f"Predicted best paragraph index: {best_idx} (score = {best_score:.3f})"
        )
        if is_correct is not None:
            lines.append(f"Is prediction correct? {is_correct}")
        else:
            lines.append("Is prediction correct? (no gold annotations)")
    else:
        lines.append("No paragraphs to score.")
        return "\n".join(lines)

    lines.append("")
    lines.append("Top paragraphs by score:")
    lines.append("idx  score    is_gold  is_predicted_best")
    for idx in top_indices:
        score = float(scores[idx])
        is_gold = idx in gold_pars
        is_pred = idx == best_idx
        lines.append(
            f"{idx:3d}  {score:0.3f}   {str(is_gold):5s}   {str(is_pred):5s}"
        )
    lines.append("")


    lines.append("-" * 80)
    lines.append("Sentence-level probabilities for the PREDICTED paragraph")
    lines.append("")
    pred_paragraph = paragraphs[best_idx]
    pred_sents = paragraph_sentences[best_idx]
    pred_probs = sentence_probs_by_paragraph[best_idx]

    for i, (s, p) in enumerate(zip(pred_sents, pred_probs)):
        lines.append(f"[pred sent {i}] prob_citation = {p:.3f}")
        lines.append(f"    {s}")
        lines.append("")
    lines.append("")


    if gold_pars:
        lines.append("-" * 80)
        lines.append("Sentence-level probabilities for GOLD paragraphs")
        lines.append("")

        num_pars = len(paragraphs)
        for gp in gold_pars:
            if not (0 <= gp < num_pars):
                lines.append(f"[gold paragraph {gp}] (index out of range)")
                lines.append("")
                continue

            tag = " (also PREDICTED)" if gp == best_idx else ""
            lines.append(f"[GOLD paragraph idx={gp}]{tag}")
            gold_sents = paragraph_sentences[gp]
            gold_probs = sentence_probs_by_paragraph[gp]

            for i, (s, p) in enumerate(zip(gold_sents, gold_probs)):
                lines.append(f"  [gold sent {i}] prob_citation = {p:.3f}")
                lines.append(f"      {s}")
                lines.append("")
            lines.append("")

    return "\n".join(lines)



# JSON output


def attach_predictions_to_json(
    paper: Dict[str, Any],
    all_scoring_results: Dict[int, Dict[str, Any]],
    agg_method: str,
    top_k: int,
):
   
    references = paper.get("references", [])
    for ref_idx, ref in enumerate(references):
        result = all_scoring_results.get(ref_idx)
        if result is None:
            continue

        scores = result["scores"]
        top_indices = result["top_indices"]

        pred_paragraphs = []
        for idx in top_indices:
            pred_paragraphs.append(
                {
                    "paragraph_idx": int(idx),
                    "score": float(scores[idx]),
                }
            )

        ref["predicted_citations"] = {
            "agg_method": agg_method,
            "top_k": int(top_k),
            "paragraphs": pred_paragraphs,
        }



# Main


def main():
    parser = argparse.ArgumentParser(
        description="Run paragraph-level citation inference on a paper JSON."
    )
    parser.add_argument("input_json", help="Path to input paper JSON")
    parser.add_argument(
        "--model-dir",
        default="saved_model_t_0_45",
        help="Directory with fine-tuned SciBERT model and inference_config.json",
    )
    parser.add_argument(
        "--agg",
        default="mean",
        choices=["mean", "max", "noisy_or"],
        help="Sentence-to-paragraph aggregation method",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="How many top paragraphs to keep per reference",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Path for augmented JSON output (defaults to <input>.predicted.json)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write per-reference .txt reports "
             "(defaults to <input>_reports)",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load model + tokenizer
    model, tokenizer, max_length, id2label, label2id, threshold = load_model_and_tokenizer(
        args.model_dir, device
    )

    # Load paper JSON
    with open(args.input_json, "r") as f:
        paper = json.load(f)

    paragraphs = paper.get("paragraphs", [])
    references = paper.get("references", [])

    if not paragraphs:
        raise ValueError("No 'paragraphs' found in input JSON.")
    if not references:
        raise ValueError("No 'references' found in input JSON.")

    print(f"Loaded paper with {len(paragraphs)} paragraphs and {len(references)} references.")

    # Pre-split all paragraphs into sentences
    paragraph_sentences = [split_into_sentences(p) for p in paragraphs]

    # Prepare output paths
    base, _ = os.path.splitext(args.input_json)
    out_json_path = args.out_json or (base + ".predicted.json")
    out_dir = args.out_dir or (base + "_reports")
    os.makedirs(out_dir, exist_ok=True)

    # Run scoring for each reference
    all_scoring_results: Dict[int, Dict[str, Any]] = {}

    for ref_idx, ref in enumerate(references):
        print(f"\nScoring reference #{ref_idx} ...")
        result = run_paragraph_scoring_for_ref(
            ref_idx=ref_idx,
            ref=ref,
            paragraphs=paragraphs,
            paragraph_sentences=paragraph_sentences,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=max_length,
            agg_method=args.agg,
            top_k=args.top_k,
        )
        all_scoring_results[ref_idx] = result

        # Build text report and save
        report_text = build_text_report_for_reference(
            ref_idx=ref_idx,
            ref=ref,
            paragraphs=paragraphs,
            paragraph_sentences=paragraph_sentences,
            scoring_result=result,
            agg_method=args.agg,
        )
        report_filename = f"ref_{ref_idx:03d}.txt"
        report_path = os.path.join(out_dir, report_filename)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        print(f"  -> wrote report to {report_path}")

      
    # Compute top-k paragraph accuracy

    correct = 0
    total = 0

    for ref_idx, result in all_scoring_results.items():
        gold_pars = result.get("gold_pars") or []
        if not gold_pars:
            continue

        top_indices = result.get("top_indices", [])
        hit = any(gp in top_indices for gp in gold_pars)
        total += 1
        if hit:
            correct += 1

    if total > 0:
        acc = correct / total
        print(
            f"\nTop-{args.top_k} paragraph accuracy "
            f"(on {total} refs with gold annotations): "
            f"{correct}/{total} = {acc:.3f}"
        )
    else:
        print("\nNo references with gold annotations found; accuracy not computed.")




    attach_predictions_to_json(
        paper=paper,
        all_scoring_results=all_scoring_results,
        agg_method=args.agg,
        top_k=args.top_k,
    )

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(paper, f, indent=2, ensure_ascii=False)

    print(f"\nWrote augmented JSON with predictions to {out_json_path}")
    print(f"Per-reference reports in directory: {out_dir}")


if __name__ == "__main__":
    main()
