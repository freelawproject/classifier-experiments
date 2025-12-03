from typing import ClassVar

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForMaskedLM, PreTrainedModel, Trainer

from .mlm_run import MLMRun


class DistillMLMTrainer(Trainer):
    """Trainer for logit distillation from a teacher model."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        training_run = self.training_run
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        student_logits = outputs.logits
        vocab_size = student_logits.size(-1)

        loss_fct = CrossEntropyLoss(ignore_index=-100)
        mlm_loss = loss_fct(
            student_logits.view(-1, vocab_size),
            labels.view(-1),
        )

        if not model.training:
            return (mlm_loss, outputs) if return_outputs else mlm_loss

        with torch.no_grad():
            teacher_model = training_run.teacher_model
            teacher_device = next(teacher_model.parameters()).device
            if teacher_device != student_logits.device:
                teacher_model.to(student_logits.device)
            teacher_outputs = teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        T = training_run.distill_temperature
        alpha_ce = training_run.distill_alpha_ce
        alpha_kl = training_run.distill_alpha_kl

        teacher_probs_T = F.softmax(teacher_logits / T, dim=-1)
        student_log_probs_T = F.log_softmax(student_logits / T, dim=-1)

        kl_loss = F.kl_div(
            student_log_probs_T.view(-1, vocab_size),
            teacher_probs_T.view(-1, vocab_size),
            reduction="batchmean",
        ) * (T**2)

        loss = alpha_ce * mlm_loss + alpha_kl * kl_loss
        return (loss, outputs) if return_outputs else loss


class DistillMLMRun(MLMRun):
    """MLM run with logit distillation from a teacher model."""

    task = "distill-mlm"
    trainer_class: Trainer = DistillMLMTrainer
    add_config_attrs: ClassVar[list[str]] = MLMRun.add_config_attrs + [
        "teacher_model_name",
        "teacher_model_args",
        "distill_temperature",
        "distill_alpha_ce",
        "distill_alpha_kl",
    ]

    def __init__(
        self,
        *args,
        teacher_model_name: str | None = None,
        teacher_model_args: dict | None = None,
        distill_temperature: float = 2.0,
        distill_alpha_ce: float = 0.5,
        distill_alpha_kl: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model_name = teacher_model_name or self.base_model_name
        self.teacher_model_args = teacher_model_args or {}
        self.distill_temperature = distill_temperature
        self.distill_alpha_ce = distill_alpha_ce
        self.distill_alpha_kl = distill_alpha_kl
        self._teacher_model = None

    @property
    def teacher_model(self) -> PreTrainedModel:
        """Teacher model for distillation."""
        if self._teacher_model is None:
            self._teacher_model = AutoModelForMaskedLM.from_pretrained(
                self.teacher_model_name,
                **self.teacher_model_args,
            )
            self._teacher_model.eval()
        return self._teacher_model
