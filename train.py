"""
PCL Detection Training Script
-------------------------------
Setup: RoBERTa-large with optional Layer-wise Learning Rate Decay and Focal Loss.
Early stopping and model selection by monitoring the validation AUPRC.
Threshold optimization based on the validation F1.
"""

import os
import sys
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainerCallback,
    set_seed,
    EarlyStoppingCallback
)
from transformers.trainer_callback import TrainerControl, TrainerState
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, average_precision_score, f1_score

# (1) Logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class JSONLLoggingCallback(TrainerCallback):
    """Appends structured training logs to a JSONL file for observability."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Optional[dict] = None, **kwargs):
        if state.is_world_process_zero and logs:
            with open(self.log_path, "a") as f:
                entry = {**logs, "step": state.global_step, "epoch": state.epoch}
                f.write(json.dumps(entry) + "\n")

# (2) Configuration Schemas
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="roberta-base")
    num_labels: int = field(default=1)

@dataclass
class DataArguments:
    train_file: str = field(default="data/processed/train.csv")
    validation_file: str = field(default="data/processed/dev_internal.csv")
    max_seq_length: int = field(default=256)

@dataclass
class PCLTrainingArguments(TrainingArguments):
    """Extends Hugging Face TrainingArguments with task-specific hyperparameters."""

    # Core
    output_dir: str = field(default="experiments/output")
    num_train_epochs: float = field(default=10.)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=4)
    bf16: bool = field(default=True)
    dataloader_pin_memory: bool = field(default=False)
    report_to: str = field(default="none")
    seed: int = field(default=42)

    # Optimizer & Scheduler
    learning_rate: float = field(default=1e-5)
    weight_decay: float = field(default=0.1)
    warmup_ratio: float = field(default=0.1)

    # Evaluation & Checkpointing
    eval_strategy: str = field(default="epoch")
    save_strategy: str = field(default="epoch")
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="auprc")
    greater_is_better: bool = field(default=True)
    save_total_limit: int = field(default=1)

    # Layer-wise Learning Rate Decay
    use_llrd: bool = field(default=False, metadata={"help": "Enable Layer-wise Learning Rate Decay."})
    llrd_decay: float = field(default=0.95, metadata={"help": "Decay factor applied per transformer layer."})

    # Alpha-Balanced Focal Loss
    use_focal_loss: bool = field(default=False, metadata={"help": "Enable Alpha-Balanced Focal Loss."})
    focal_gamma: float = field(default=1.0, metadata={"help": "Focusing parameter to down-weight easy examples."})
    focal_alpha: float = field(default=0.75, metadata={"help": "Weight for minority class (PCL)."})


# (3) Custom Trainer
class PCLTrainer(Trainer):
    """Custom Trainer overriding standard behaviors to handle LLRD and the focal loss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args: PCLTrainingArguments = self.args

    def create_optimizer(self):
        """Injects LLRD for the RoBERTa architecture."""
        if not getattr(self.args, "use_llrd", False) or self.optimizer is not None:
            return super().create_optimizer()

        logger.info(f"Applying LLRD with decay factor: {self.args.llrd_decay}...")
        opt_model = self.model
        lr = self.args.learning_rate
        decay = self.args.llrd_decay
        weight_decay = self.args.weight_decay
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped_parameters = list()

        # Separate the layers. Reverse to start applying decay from the top down.
        layers = [opt_model.roberta.embeddings] + list(opt_model.roberta.encoder.layer)
        layers.reverse()

        # Classification head receives the highest (base) learning rate
        optimizer_grouped_parameters += [
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if ("classifier" in n or "pooler" in n) and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
                "lr": lr
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if ("classifier" in n or "pooler" in n) and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.,
                "lr": lr
            }
        ]

        # Progressively decay the LR for lower Transformer layers
        current_lr = lr
        for layer in layers:
            current_lr *= decay
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                    "lr": current_lr
                },
                {
                    "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": current_lr
                }
            ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Applies standard BCE or focal loss."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if getattr(self.args, "use_focal_loss", False):
            bce_loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.float().view(-1), reduction='none')
            with torch.no_grad():
                pt = torch.clamp(torch.exp(-bce_loss), min=1e-8, max=1.0 - 1e-8)
            y = labels.float().view(-1)
            alpha_t = y * self.args.focal_alpha + (1 - y) * (1 - self.args.focal_alpha)
            cls_loss = (alpha_t * (1 - pt) ** self.args.focal_gamma * bce_loss).mean()
        else:
            cls_loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.float().view(-1))

        return (cls_loss, outputs) if return_outputs else cls_loss

# (4) Metrics & Optimization
def find_optimal_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Vectorized sweep via precision-recall curve to find the max F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)

    numerator = 2 * (precisions[:-1] * recalls[:-1])
    denominator = precisions[:-1] + recalls[:-1]
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx])

def compute_metrics(p) -> dict[str, float]:
    """Sweeps thresholds during evaluation to guarantee optimal epoch tracking."""
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    probs = (1 / (1 + np.exp(-logits))).squeeze(-1)
    labels = p.label_ids

    # (A) Model selection metric (threshold-agnostic)
    auprc = float(average_precision_score(labels, probs))

    # (B) Ceiling F1 — observability only, not for selection
    best_thresh, ceiling_f1 = find_optimal_threshold(labels, probs)

    # (C) Conservative F1 at fixed threshold
    fixed_preds = (probs >= 0.5).astype(int)
    fixed_f1 = float(f1_score(labels, fixed_preds, zero_division=0))

    best_preds = (probs >= best_thresh).astype(int)

    return {
        "auprc": auprc,
        "ceiling_f1": ceiling_f1,
        "ceiling_threshold": float(best_thresh),
        "fixed_f1": fixed_f1,
        "precision": float(precision_score(labels, best_preds, zero_division=0)),
        "recall": float(recall_score(labels, best_preds, zero_division=0)),
    }

# (5) Main Pipeline
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, PCLTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    logger.info(f"Loading datasets from {data_args.train_file} & {data_args.validation_file}...")
    raw_datasets = load_dataset("csv", data_files={"train": data_args.train_file, "validation": data_args.validation_file})

    if "label" in raw_datasets["train"].column_names:
        raw_datasets = raw_datasets.rename_column("label", "labels")

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # Inject contextual information
    special_tokens_dict = {"additional_special_tokens": ["[KEYWORD]", "[COUNTRY]", "[SEP]"]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"Added {num_added_toks} special tokens to the tokenizer.")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=data_args.max_seq_length)

    with training_args.main_process_first(desc="Tokenizing"):
        tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

    # Ensure text metadata doesn't crash the forward pass
    model_columns = ["input_ids", "attention_mask", "labels", "token_type_ids"]
    columns_to_remove = [c for c in tokenized_datasets["train"].column_names if c not in model_columns]
    tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=model_args.num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)

    # Resize embeddings to accommodate the new tokens
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Resized model embeddings to {len(tokenizer)}.")

    prior_prob = sum(1 for x in raw_datasets["train"]["labels"] if x == 1) / len(raw_datasets["train"])
    bias_value = -np.log((1 - prior_prob) / prior_prob)
    model.classifier.out_proj.bias.data.fill_(bias_value)
    logger.info(f"Initialized classification bias to {bias_value:.4f} (prior_prob={prior_prob:.4f})")

    trainer = PCLTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[
            JSONLLoggingCallback(os.path.join(training_args.output_dir, "metrics.jsonl")),
            EarlyStoppingCallback(early_stopping_patience=3)
        ]
    )

    logger.info("*** Starting Training ***")
    train_result = trainer.train()

    logger.info("*** Running Final Evaluation ***")
    val_output = trainer.predict(tokenized_datasets["validation"])
    best_threshold = val_output.metrics["test_ceiling_threshold"]
    best_f1 = val_output.metrics["test_ceiling_f1"]

    logger.info(f"Validation Results -> Threshold: {best_threshold:.4f} | F1: {best_f1:.4f}")

    trainer.save_model()
    trainer.log_metrics(split="train", metrics=train_result.metrics)
    trainer.save_metrics(split="train", metrics=train_result.metrics)
    trainer.save_state()

    final_config = {
        "model_args": asdict(model_args),
        "data_args": asdict(data_args),
        "training_args": training_args.to_dict(),
        "optimization": {
            "best_threshold": best_threshold,
            "best_val_f1": best_f1
        }
    }

    with open(os.path.join(training_args.output_dir, "run_config.json"), "w") as f:
        json.dump(final_config, f, indent=4, default=str)

    logger.info(f"Training complete. Artifacts saved to {training_args.output_dir}")

if __name__ == "__main__":
    main()