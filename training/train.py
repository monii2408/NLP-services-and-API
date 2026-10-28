"""Fine-tune a base encoder for extractive QA on SQuAD 2.0.

Owner: Mauricio.

Pipeline:
1. Load SQuAD 2.0 (Hugging Face `rajpurkar/squad_v2`) and optionally subsample the train split,
   keeping the answerable/unanswerable ratio of the full set.
2. Tokenize question + context into windows of `max_seq_length` tokens that overlap by
   `doc_stride` tokens. Each window is labeled with the token positions of the answer,
   or with the [CLS] token (index 0) when the answer is not inside that window or the
   question is unanswerable.
3. Fine-tune with the Hugging Face Trainer, evaluating the validation loss periodically.
4. Save the weights to models/ (not committed) and the logs, loss curve, config and
   validation metrics to logs/<run_name>/ (committed).

Validation EM/F1 here use a fixed no-answer threshold of 0 (null wins if its score is
higher than the best span). Tuning that threshold is done in evaluation/.

Examples:
    # quick smoke test (a few minutes)
    python training/train.py --run_name smoke --train_samples 2000 --eval_samples 500 --epochs 1
    # full run
    python training/train.py --run_name distilbert-full
"""

import argparse
import collections
import json
import logging
import platform
import re
import string
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("train")
DEFAULT_BASE_CHECKPOINT = "distilbert-base-uncased"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_BASE_CHECKPOINT, help="Base checkpoint (must NOT be fine-tuned on SQuAD).")
    p.add_argument("--run_name", default="distilbert-squad2")
    p.add_argument("--output_dir", default=None, help="Where to save the final model. Default: models/<run_name>.")
    p.add_argument("--max_seq_length", type=int, default=384)
    p.add_argument("--doc_stride", type=int, default=128)
    p.add_argument("--train_samples", type=int, default=None, help="Subsample the train split (stratified). Default: all.")
    p.add_argument("--eval_samples", type=int, default=None, help="Subsample the validation split (stratified). Default: all.")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--n_best", type=int, default=20)
    p.add_argument("--max_answer_length", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / "train.log", mode="w", encoding="utf-8")):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    # The Trainer logs through the transformers logger; send it to the same handlers.
    import transformers

    transformers.logging.set_verbosity_info()
    transformers.logging.disable_default_handler()
    transformers.logging.enable_propagation()
    for noisy in ("httpx", "httpx2", "huggingface_hub", "filelock", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def stratified_subsample(ds, n, seed):
    """Take n examples keeping the answerable/unanswerable ratio of ds."""
    if n is None or n >= len(ds):
        return ds
    rng = np.random.default_rng(seed)
    is_impossible = np.array([len(a["text"]) == 0 for a in ds["answers"]])
    impossible_idx = np.flatnonzero(is_impossible)
    possible_idx = np.flatnonzero(~is_impossible)
    n_impossible = round(n * len(impossible_idx) / len(ds))
    # Keep both classes represented whenever the requested sample can do so.
    if n >= 2 and len(impossible_idx) and len(possible_idx):
        n_impossible = min(max(n_impossible, 1), n - 1)
    picked = np.concatenate([
        rng.choice(impossible_idx, n_impossible, replace=False),
        rng.choice(possible_idx, n - n_impossible, replace=False),
    ])
    rng.shuffle(picked)
    return ds.select(picked.tolist())


def count_unanswerable(ds):
    return sum(len(a["text"]) == 0 for a in ds["answers"])


def tokenize_windows(tokenizer, examples, max_seq_length, doc_stride):
    # Questions sometimes start with whitespace, which only wastes window space.
    questions = [q.lstrip() for q in examples["question"]]
    return tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )


def prepare_train_features(examples, tokenizer, max_seq_length, doc_stride):
    """Split each example into windows and label start/end token positions (0 = [CLS] = no answer)."""
    enc = tokenize_windows(tokenizer, examples, max_seq_length, doc_stride)
    sample_map = enc.pop("overflow_to_sample_mapping")
    offsets_all = enc.pop("offset_mapping")
    start_positions, end_positions = [], []

    for i, offsets in enumerate(offsets_all):
        answers = examples["answers"][sample_map[i]]
        if len(answers["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])
        seq_ids = enc.sequence_ids(i)
        ctx_start = seq_ids.index(1)
        ctx_end = len(seq_ids) - 1 - seq_ids[::-1].index(1)

        # Answer not fully inside this window -> label as no answer for this window.
        if offsets[ctx_start][0] > start_char or offsets[ctx_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
            continue

        tok_start = ctx_start
        while tok_start <= ctx_end and offsets[tok_start][0] <= start_char:
            tok_start += 1
        tok_end = ctx_end
        while tok_end >= ctx_start and offsets[tok_end][1] >= end_char:
            tok_end -= 1
        start_positions.append(tok_start - 1)
        end_positions.append(tok_end + 1)

    enc["start_positions"] = start_positions
    enc["end_positions"] = end_positions
    return enc


def prepare_eval_features(examples, tokenizer, max_seq_length, doc_stride):
    """Windows for prediction: keep example id and context-only offsets for post-processing."""
    enc = tokenize_windows(tokenizer, examples, max_seq_length, doc_stride)
    sample_map = enc.pop("overflow_to_sample_mapping")
    enc["example_id"] = []
    for i in range(len(enc["input_ids"])):
        seq_ids = enc.sequence_ids(i)
        enc["example_id"].append(examples["id"][sample_map[i]])
        enc["offset_mapping"][i] = [o if seq_ids[k] == 1 else None for k, o in enumerate(enc["offset_mapping"][i])]
    return enc


def postprocess_predictions(examples, features, start_logits, end_logits, n_best, max_answer_length):
    """Best span per example across its windows; predict "" (no answer) if the null score is higher.

    Null score of an example = min over its windows of start_logit[CLS] + end_logit[CLS].
    Returns {id: answer_text} and {id: best_span_score - null_score} (useful for threshold tuning).
    """
    features_per_example = collections.defaultdict(list)
    for idx, ex_id in enumerate(features["example_id"]):
        features_per_example[ex_id].append(idx)
    all_offsets = features["offset_mapping"]

    predictions, score_diffs = {}, {}
    for ex in examples:
        context = ex["context"]
        null_score = None
        best_text, best_score = "", None
        for fi in features_per_example[ex["id"]]:
            s_log, e_log = start_logits[fi], end_logits[fi]
            offsets = all_offsets[fi]
            cls_score = s_log[0] + e_log[0]
            null_score = cls_score if null_score is None else min(null_score, cls_score)
            for s in np.argsort(s_log)[-1 : -n_best - 1 : -1]:
                for e in np.argsort(e_log)[-1 : -n_best - 1 : -1]:
                    if s >= len(offsets) or e >= len(offsets) or offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s or e - s + 1 > max_answer_length:
                        continue
                    score = s_log[s] + e_log[e]
                    if best_score is None or score > best_score:
                        best_score = score
                        best_text = context[offsets[s][0] : offsets[e][1]]
        if best_score is None:
            predictions[ex["id"]], score_diffs[ex["id"]] = "", float("-inf")
        else:
            diff = float(best_score - null_score)
            predictions[ex["id"]] = best_text if diff > 0 else ""
            score_diffs[ex["id"]] = diff
    return predictions, score_diffs


def normalize_answer(text):
    """Official SQuAD normalization: lowercase, drop punctuation, articles and extra whitespace."""
    text = "".join(ch for ch in text.lower() if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1_score(prediction, reference):
    pred_tokens, ref_tokens = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = collections.Counter(pred_tokens) & collections.Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred_tokens), overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def squad_v2_metrics(examples, predictions):
    """EM and F1 (in %) overall, on answerable (HasAns) and unanswerable (NoAns) questions.

    The prediction "" means no answer; for unanswerable questions the only reference is "".
    """
    scores = {"all": [], "HasAns": [], "NoAns": []}
    for ex in examples:
        refs = [t for t in ex["answers"]["text"] if normalize_answer(t)] or [""]
        pred = predictions[ex["id"]]
        em = max(float(normalize_answer(pred) == normalize_answer(r)) for r in refs)
        f1 = max(f1_score(pred, r) for r in refs)
        group = "HasAns" if ex["answers"]["text"] else "NoAns"
        scores["all"].append((em, f1))
        scores[group].append((em, f1))
    metrics = {}
    for group, values in scores.items():
        prefix = "" if group == "all" else f"{group}_"
        if values:
            metrics[f"{prefix}exact"] = 100 * float(np.mean([v[0] for v in values]))
            metrics[f"{prefix}f1"] = 100 * float(np.mean([v[1] for v in values]))
        metrics[f"{prefix}total"] = len(values)
    return metrics


def plot_loss(log_history, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = [(h["step"], h["loss"]) for h in log_history if "loss" in h]
    evals = [(h["step"], h["eval_loss"]) for h in log_history if "eval_loss" in h]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if train:
        ax.plot(*zip(*train), label="train loss", linewidth=1)
    if evals:
        ax.plot(*zip(*evals), "o-", label="validation loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (mean of start/end cross-entropy)")
    ax.set_title(path.parent.name)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    if args.max_seq_length < 3:
        raise SystemExit("--max_seq_length must be at least 3 tokens.")
    if not 0 <= args.doc_stride < args.max_seq_length:
        raise SystemExit("--doc_stride must be non-negative and smaller than --max_seq_length.")
    if args.train_samples is not None and args.train_samples < 1:
        raise SystemExit("--train_samples must be positive when provided.")
    if args.eval_samples is not None and args.eval_samples < 1:
        raise SystemExit("--eval_samples must be positive when provided.")
    log_dir = REPO_ROOT / "logs" / args.run_name
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "models" / args.run_name
    setup_logging(log_dir)
    set_seed(args.seed)

    if "squad" in args.model.lower():
        raise SystemExit(f"{args.model} looks like a SQuAD fine-tuned checkpoint; the rules require a base checkpoint.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    hardware = torch.cuda.get_device_name(0) if device == "cuda" else platform.processor()
    logger.info("Device: %s (%s), bf16=%s", device, hardware, use_bf16)

    raw = load_dataset("rajpurkar/squad_v2")
    train_raw = stratified_subsample(raw["train"], args.train_samples, args.seed)
    eval_raw = stratified_subsample(raw["validation"], args.eval_samples, args.seed)
    logger.info("Train examples: %d (%d unanswerable) of %d", len(train_raw), count_unanswerable(train_raw), len(raw["train"]))
    logger.info("Validation examples: %d (%d unanswerable) of %d", len(eval_raw), count_unanswerable(eval_raw), len(raw["validation"]))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.padding_side != "right":
        raise SystemExit("This script assumes right padding (question first, context second).")

    fn_kwargs = dict(tokenizer=tokenizer, max_seq_length=args.max_seq_length, doc_stride=args.doc_stride)
    train_features = train_raw.map(prepare_train_features, batched=True, remove_columns=train_raw.column_names, fn_kwargs=fn_kwargs, desc="Tokenizing train")
    eval_loss_features = eval_raw.map(prepare_train_features, batched=True, remove_columns=eval_raw.column_names, fn_kwargs=fn_kwargs, desc="Tokenizing validation (loss)")
    eval_pred_features = eval_raw.map(prepare_eval_features, batched=True, remove_columns=eval_raw.column_names, fn_kwargs=fn_kwargs, desc="Tokenizing validation (predict)")
    logger.info("Train windows: %d, validation windows: %d", len(train_features), len(eval_loss_features))

    model = AutoModelForQuestionAnswering.from_pretrained(args.model)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model %s, %d parameters", args.model, n_params)

    training_args = TrainingArguments(
        output_dir=str(REPO_ROOT / "checkpoints" / args.run_name),
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_steps=args.warmup_ratio,  # a float in [0, 1) is a ratio of total steps
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        bf16=use_bf16,
        fp16=device == "cuda" and not use_bf16,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_features,
        eval_dataset=eval_loss_features,
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
    )

    t0 = time.time()
    train_result = trainer.train()
    train_seconds = time.time() - t0
    final_eval = trainer.evaluate()
    logger.info("Training done in %.1f min; final eval loss %.4f", train_seconds / 60, final_eval["eval_loss"])

    # Validation EM/F1 (threshold 0).
    pred_inputs = eval_pred_features.remove_columns(["example_id", "offset_mapping"])
    start_logits, end_logits = trainer.predict(pred_inputs).predictions
    predictions, score_diffs = postprocess_predictions(eval_raw, eval_pred_features, start_logits, end_logits, args.n_best, args.max_answer_length)

    metrics = squad_v2_metrics(eval_raw, predictions)
    logger.info("Validation metrics (threshold 0): %s", json.dumps(metrics, indent=2))

    # Save model (not committed) and metadata for the service (/info).
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metadata = {
        "model": args.model,
        "params": n_params,
        "max_seq_length": args.max_seq_length,
        "doc_stride": args.doc_stride,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Logs (committed).
    config = {
        **vars(args),
        "output_dir": str(output_dir.relative_to(REPO_ROOT)) if output_dir.is_relative_to(REPO_ROOT) else str(output_dir),
        "train_examples": len(train_raw),
        "train_unanswerable": count_unanswerable(train_raw),
        "train_windows": len(train_features),
        "eval_examples": len(eval_raw),
        "eval_unanswerable": count_unanswerable(eval_raw),
        "eval_windows": len(eval_loss_features),
        "params": n_params,
        "hardware": hardware,
        "precision": "bf16" if use_bf16 else ("fp16" if device == "cuda" else "fp32"),
        "torch": torch.__version__,
        "train_minutes": round(train_seconds / 60, 2),
        "global_steps": train_result.global_step,
    }
    (log_dir / "config.json").write_text(json.dumps(config, indent=2))
    (log_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    (log_dir / "metrics.json").write_text(json.dumps({"final_eval_loss": final_eval["eval_loss"], "threshold_0": metrics}, indent=2))
    (log_dir / "val_score_diffs.json").write_text(json.dumps(score_diffs))
    plot_loss(trainer.state.log_history, log_dir / "loss.png")
    logger.info("Model saved to %s; logs saved to %s", output_dir, log_dir)


if __name__ == "__main__":
    main()
