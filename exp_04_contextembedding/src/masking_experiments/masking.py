from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MaskingConfig, SoftColumnMaskConfig, TaskConfig
from .prompting import PromptVectorProvider


@dataclass
class Mask:
    """Mask applied to the sparse SNN."""

    learning_rate_scale: torch.Tensor
    threshold_shift: torch.Tensor
    prompt_vector: torch.Tensor
    column_mask: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "Mask":
        column_mask = self.column_mask.to(device) if self.column_mask is not None else None
        return Mask(
            learning_rate_scale=self.learning_rate_scale.to(device),
            threshold_shift=self.threshold_shift.to(device),
            prompt_vector=self.prompt_vector.to(device),
            column_mask=column_mask,
        )


class MaskController:
    """Build masks based on task prompts."""

    def __init__(
        self,
        tasks: List[TaskConfig],
        config: MaskingConfig,
        task_input_stats: Optional[Dict[str, torch.Tensor]] = None,
        input_dim: int | None = None,
    ) -> None:
        self._task_by_prompt: Dict[str, TaskConfig] = {task.prompt: task for task in tasks}
        self._tasks_by_name: Dict[str, TaskConfig] = {task.name: task for task in tasks}
        self._tasks = list(tasks)
        self._config = config
        self._task_input_stats: Dict[str, torch.Tensor] = {}
        if task_input_stats:
            for name, stats in task_input_stats.items():
                self._task_input_stats[name] = stats.clone().detach()
        num_embeddings = None
        if config.prompt_mode.lower() == "learned_task_id":
            task_ids = [task.task_id for task in tasks]
            if any(identifier is None for identifier in task_ids):
                raise ValueError("All tasks must define task_id when prompt_mode='learned_task_id'.")
            max_id = max(task_ids) if task_ids else -1
            num_embeddings = max_id + 1
        self._prompt_vectors = PromptVectorProvider(
            dimension=config.prompt_vector_dim,
            mode=config.prompt_mode,
            num_embeddings=num_embeddings,
            learning_rate=config.prompt_learning_rate,
            include_input=config.include_input_in_prompt,
            input_dim=input_dim if config.include_input_in_prompt else None,
            input_scale=config.prompt_input_scale,
        )
        self.num_columns = max(task.column_index for task in tasks) + 1

    def resolve_task(self, key: str) -> TaskConfig:
        if key in self._task_by_prompt:
            return self._task_by_prompt[key]
        if key in self._tasks_by_name:
            return self._tasks_by_name[key]
        raise KeyError(f"Unknown task or prompt '{key}'.")

    def _active_columns(self, task: TaskConfig) -> List[int]:
        columns = {task.column_index}
        if task.residual_from:
            base_task = self._tasks_by_name.get(task.residual_from)
            if base_task is None:
                raise KeyError(
                    f"Task '{task.name}' references residual_from='{task.residual_from}' which was not found."
                )
            columns.add(base_task.column_index)
        return sorted(columns)

    def _column_mask_tensor(self, columns: List[int]) -> torch.Tensor:
        mask = torch.zeros(self.num_columns, dtype=torch.float32)
        mask[columns] = 1.0
        return mask

    def build_mask(self, context: str) -> Mask:
        task = self.resolve_task(context)
        lr_scale = torch.full((self.num_columns,), self._config.learning_rate_scale, dtype=torch.float32)
        threshold_shift = torch.full((self.num_columns,), self._config.threshold_shift, dtype=torch.float32)

        active_columns = self._active_columns(task)
        for idx in active_columns:
            threshold_shift[idx] = 0.0

        lr_scale[task.column_index] = 1.0
        for idx in active_columns:
            if idx != task.column_index:
                lr_scale[idx] = 0.0

        prompt_vec = self._context_vector(task)
        column_mask = self._column_mask_tensor(active_columns)

        return Mask(
            learning_rate_scale=lr_scale,
            threshold_shift=threshold_shift,
            prompt_vector=prompt_vec,
            column_mask=column_mask,
        )

    def regularization_loss(self, task_name: str) -> torch.Tensor | None:
        return None

    def apply_gradients(self, task_name: str | None = None) -> None:
        self._prompt_vectors.apply_gradients()

    def zero_grad(self, task_name: str | None = None) -> None:
        self._prompt_vectors.zero_grad()

    def get_masks_dict(self, prompt_overrides: Optional[Dict[str, str]] = None) -> Dict[str, torch.Tensor]:
        masks: Dict[str, torch.Tensor] = {}
        for task in self._tasks:
            masks[task.name] = self._column_mask_tensor(self._active_columns(task))
        return masks

    def column_occupancy(self, prompt_overrides: Optional[Dict[str, str]] = None) -> Dict[str, torch.Tensor]:
        if not self._tasks:
            return {}
        masks = torch.stack([self._column_mask_tensor(self._active_columns(task)) for task in self._tasks])
        return {"columns": masks.mean(dim=0)}

    def to(self, device: torch.device) -> "MaskController":
        self._prompt_vectors.to(device)
        return self

    def promote_to_base(self, task_names: Sequence[str]) -> None:
        return

    def context_vector(self, task_name: str) -> torch.Tensor:
        task = self._tasks_by_name[task_name]
        return self._context_vector(task).detach()

    def _context_vector(self, task: TaskConfig) -> torch.Tensor:
        stats = self._task_input_stats.get(task.name)
        if self._config.include_input_in_prompt and stats is None:
            raise ValueError(f"No input statistics registered for task '{task.name}'.")
        return self._prompt_vectors.vector(task.prompt, task.task_id, stats)


class SoftColumnMaskController(nn.Module):
    """Learnable soft mask controller with base/novel column banks."""

    def __init__(
        self,
        tasks: List[TaskConfig],
        config: MaskingConfig,
        task_input_stats: Optional[Dict[str, torch.Tensor]] = None,
        input_dim: int | None = None,
    ) -> None:
        super().__init__()
        self._tasks_by_name: Dict[str, TaskConfig] = {task.name: task for task in tasks}
        self._tasks_by_prompt: Dict[str, TaskConfig] = {task.prompt: task for task in tasks}
        self._config = config
        self._task_input_stats: Dict[str, torch.Tensor] = {}
        if task_input_stats:
            for name, stats in task_input_stats.items():
                self._task_input_stats[name] = stats.clone().detach()
        num_embeddings = None
        if config.prompt_mode.lower() == "learned_task_id":
            task_ids = [task.task_id for task in tasks]
            if any(identifier is None for identifier in task_ids):
                raise ValueError("All tasks must define task_id when prompt_mode='learned_task_id'.")
            max_id = max(task_ids) if task_ids else -1
            num_embeddings = max_id + 1
        self.prompt_vectors = PromptVectorProvider(
            dimension=config.prompt_vector_dim,
            mode=config.prompt_mode,
            num_embeddings=num_embeddings,
            learning_rate=config.prompt_learning_rate,
            include_input=config.include_input_in_prompt,
            input_dim=input_dim if config.include_input_in_prompt else None,
            input_scale=config.prompt_input_scale,
        )
        self.soft_cfg: SoftColumnMaskConfig = config.soft_columns
        if self.soft_cfg.total_columns <= 0:
            raise ValueError("soft_columns.total_columns must be > 0 when mask.mode='soft_columns'.")
        self.total_columns = self.soft_cfg.total_columns
        base_columns = self.soft_cfg.base_columns if self.soft_cfg.base_columns is not None else self.total_columns
        if base_columns <= 0 or base_columns > self.total_columns:
            raise ValueError("soft_columns.base_columns must be > 0 and <= total_columns.")
        self.base_columns = base_columns
        self.novel_columns = self.total_columns - self.base_columns
        self.temperature = max(self.soft_cfg.temperature, 1e-3)
        self.lambda_entropy = self.soft_cfg.lambda_entropy
        self.lambda_balance = self.soft_cfg.lambda_balance
        self.lambda_overlap_novel = self.soft_cfg.lambda_overlap_novel
        self.mask_learning_rate = self.soft_cfg.mask_learning_rate
        self.num_columns = self.total_columns
        self._device = torch.device("cpu")

        base_tasks = set(self.soft_cfg.base_tasks or [task.name for task in tasks])
        novel_tasks = set(self.soft_cfg.novel_tasks)
        self._task_to_bank: Dict[str, str] = {}
        for task in tasks:
            if task.name in novel_tasks:
                if self.novel_columns <= 0:
                    raise ValueError("novel tasks configured but no novel columns available.")
                bank = "novel"
            else:
                bank = "base"
            self._task_to_bank[task.name] = bank
        for task_name in novel_tasks:
            if task_name not in self._task_to_bank:
                raise KeyError(f"Novel task '{task_name}' not found in configured tasks.")

        self._bank_to_tasks: Dict[str, List[str]] = {"base": [], "novel": []}
        for name, bank in self._task_to_bank.items():
            self._bank_to_tasks[bank].append(name)

        self.use_key_query = config.key_query.enabled
        self.logits = nn.ParameterDict()
        if self.use_key_query:
            self.key_temperature = max(config.key_query.temperature, 1e-3)
            self.locked_temperature = max(config.key_query.locked_temperature, 1e-3)
            self.assignment_threshold = config.key_query.assignment_threshold
            self.occupied_penalty = config.key_query.occupied_penalty
            self.lock_bonus = config.key_query.lock_bonus
            self.margin = config.key_query.margin
            self.margin_weight = config.key_query.margin_weight
            self.column_keys = nn.Parameter(
                torch.randn(self.total_columns, config.prompt_vector_dim) * config.key_query.init_scale
            )
            self._task_assignment: Dict[str, int] = {}
            self._column_assignments: Dict[int, str] = {}
        else:
            for name, bank in self._task_to_bank.items():
                size = self._bank_size(bank)
                param = nn.Parameter(torch.zeros(size))
                self.logits[name] = param

    def _bank_size(self, bank: str) -> int:
        return self.base_columns if bank == "base" else self.novel_columns

    def resolve_task(self, key: str) -> TaskConfig:
        if key in self._tasks_by_prompt:
            return self._tasks_by_prompt[key]
        if key in self._tasks_by_name:
            return self._tasks_by_name[key]
        raise KeyError(f"Unknown task or prompt '{key}'.")

    def _bank_slice(self, bank: str) -> slice:
        if bank == "base":
            return slice(0, self.base_columns)
        return slice(self.base_columns, self.total_columns)

    def _distribution(self, task_name: str, detach: bool) -> torch.Tensor:
        logits = self.logits[task_name]
        scaled = logits / self.temperature
        weights = F.softmax(scaled, dim=-1)
        return weights.detach() if detach else weights

    def _full_mask(self, task_name: str, detach: bool) -> torch.Tensor:
        if self.use_key_query:
            raise RuntimeError("full_mask should not be called when key-query routing is enabled.")
        bank = self._task_to_bank[task_name]
        param = self.logits[task_name]
        mask = torch.zeros(self.total_columns, device=param.device, dtype=param.dtype)
        bank_mask = self._distribution(task_name, detach=detach)
        mask[self._bank_slice(bank)] = bank_mask
        return mask

    def _key_query_mask(self, task: TaskConfig, prompt_vec: torch.Tensor, detach: bool) -> torch.Tensor:
        scores = torch.matmul(self.column_keys, prompt_vec)
        bank = self._task_to_bank[task.name]
        bank_slice = self._bank_slice(bank)
        # Apply occupied penalty to columns already claimed by other tasks within this bank.
        if self.occupied_penalty > 0.0:
            for col_idx, owner in self._column_assignments.items():
                if owner != task.name and bank_slice.start <= col_idx < bank_slice.stop:
                    scores[col_idx] -= self.occupied_penalty
        assigned_idx = self._task_assignment.get(task.name)
        temperature = self.key_temperature
        if assigned_idx is not None:
            temperature = self.locked_temperature
            scores[assigned_idx] += self.lock_bonus
        mask = torch.full_like(scores, float("-inf"))
        mask[bank_slice] = scores[bank_slice] / temperature
        weights = F.softmax(mask, dim=-1)
        if not detach:
            with torch.no_grad():
                slice_weights = weights[bank_slice]
                best_value, best_idx = torch.max(slice_weights, dim=0)
                if float(best_value.item()) >= self.assignment_threshold:
                    global_idx = bank_slice.start + int(best_idx.item())
                    owner = self._column_assignments.get(global_idx)
                    if owner in (None, task.name):
                        self._column_assignments[global_idx] = task.name
                        self._task_assignment[task.name] = global_idx
        return weights.detach() if detach else weights

    def build_mask(self, context: str) -> Mask:
        task = self.resolve_task(context)
        if self.use_key_query:
            prompt_vec = self._context_vector(task).to(self.column_keys.device)
            full_mask = self._key_query_mask(task, prompt_vec, detach=False)
            lr_mask = full_mask.detach()
            threshold_shift = torch.zeros(self.total_columns, device=full_mask.device, dtype=full_mask.dtype)
            return Mask(
                learning_rate_scale=lr_mask,
                threshold_shift=threshold_shift,
                prompt_vector=prompt_vec.to(full_mask.device),
                column_mask=full_mask,
            )
        prompt_vec = self._context_vector(task)
        full_mask = self._full_mask(task.name, detach=False)
        lr_mask = full_mask.detach()
        threshold_shift = torch.zeros(self.total_columns, device=full_mask.device, dtype=full_mask.dtype)
        return Mask(
            learning_rate_scale=lr_mask,
            threshold_shift=threshold_shift,
            prompt_vector=prompt_vec.to(full_mask.device),
            column_mask=full_mask,
        )

    def regularization_loss(self, task_name: str) -> torch.Tensor | None:
        penalties: Optional[torch.Tensor] = None
        bank = self._task_to_bank[task_name]
        if self.use_key_query:
            task = self._tasks_by_name[task_name]
            full = self._key_query_mask(task, self._context_vector(task).to(self.column_keys.device), detach=False)
            bank_mask = full[self._bank_slice(bank)]
        else:
            bank_mask = self._distribution(task_name, detach=False)
        if self.lambda_entropy > 0.0:
            entropy = -(bank_mask * (bank_mask + 1e-8).log()).sum()
            penalties = entropy * self.lambda_entropy
        if self.lambda_balance > 0.0:
            uniform = torch.full_like(bank_mask, 1.0 / max(1, bank_mask.numel()))
            balance = F.mse_loss(bank_mask, uniform)
            penalties = balance * self.lambda_balance if penalties is None else penalties + balance * self.lambda_balance
        if self.lambda_overlap_novel > 0.0 and bank == "novel" and not self.use_key_query:
            overlaps = []
            for other_name in self._bank_to_tasks.get("novel", []):
                if other_name == task_name:
                    continue
                other_mask = self._distribution(other_name, detach=True)
                overlaps.append(torch.dot(bank_mask, other_mask))
            if overlaps:
                overlap_value = torch.stack(overlaps).mean()
                overlap_term = overlap_value * self.lambda_overlap_novel
                penalties = overlap_term if penalties is None else penalties + overlap_term
        if self.use_key_query and self.margin > 0.0 and task_name in self._task_assignment:
            assigned_idx = self._task_assignment.get(task_name)
            if assigned_idx is not None:
                task = self._tasks_by_name[task_name]
                vec = self._context_vector(task).to(self.column_keys.device)
                scores = torch.matmul(self.column_keys, vec)
                bank_slice = self._bank_slice(bank)
                if bank_slice.start <= assigned_idx < bank_slice.stop:
                    local_scores = scores[bank_slice]
                    assigned_local = assigned_idx - bank_slice.start
                    if local_scores.numel() > 1:
                        mask = torch.ones_like(local_scores, dtype=torch.bool)
                        mask[assigned_local] = False
                        max_other = local_scores[mask].max()
                        diff = local_scores[assigned_local] - max_other
                        hinge = torch.relu(self.margin - diff)
                        if hinge.item() > 0:
                            penalties = hinge * self.margin_weight if penalties is None else penalties + hinge * self.margin_weight
        return penalties

    def apply_gradients(self, task_name: str | None = None) -> None:
        if self.mask_learning_rate <= 0.0:
            self.prompt_vectors.apply_gradients()
            return
        if self.use_key_query:
            if self.column_keys.grad is not None:
                self.column_keys.data -= self.mask_learning_rate * self.column_keys.grad
        else:
            names = [task_name] if task_name else list(self.logits.keys())
            for name in names:
                param = self.logits[name]
                if param.grad is None:
                    continue
                param.data -= self.mask_learning_rate * param.grad
        self.prompt_vectors.apply_gradients()

    def zero_grad(self, task_name: str | None = None) -> None:
        if self.use_key_query:
            if self.column_keys.grad is not None:
                self.column_keys.grad.detach_()
                self.column_keys.grad.zero_()
        else:
            names = [task_name] if task_name else list(self.logits.keys())
            for name in names:
                param = self.logits[name]
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()
        self.prompt_vectors.zero_grad()

    def get_masks_dict(self, prompt_overrides: Optional[Dict[str, str]] = None) -> Dict[str, torch.Tensor]:
        overrides = prompt_overrides or {}
        masks: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for task_name, task in self._tasks_by_name.items():
                context = overrides.get(task_name, task.prompt)
                context_task = self.resolve_task(context)
                if self.use_key_query:
                    vec = self._context_vector(context_task).to(self.column_keys.device)
                    mask = self._key_query_mask(context_task, vec, detach=True)
                else:
                    mask = self._full_mask(context_task.name, detach=True)
                masks[task_name] = mask.cpu()
        return masks

    def column_occupancy(self, prompt_overrides: Optional[Dict[str, str]] = None) -> Dict[str, torch.Tensor]:
        overrides = prompt_overrides or {}
        occupancy: Dict[str, List[torch.Tensor]] = {"base": [], "novel": []}
        with torch.no_grad():
            for task_name, task in self._tasks_by_name.items():
                context = overrides.get(task_name, task.prompt)
                context_task = self.resolve_task(context)
                bank = self._task_to_bank[context_task.name]
                if self.use_key_query:
                    vec = self._context_vector(context_task).to(self.column_keys.device)
                    bank_mask = self._key_query_mask(context_task, vec, detach=True)[self._bank_slice(bank)]
                else:
                    bank_mask = self._distribution(context_task.name, detach=True)
                occupancy[bank].append(bank_mask.cpu())
        averaged: Dict[str, torch.Tensor] = {}
        for bank, entries in occupancy.items():
            if entries:
                stacked = torch.stack(entries, dim=0)
                averaged[bank] = stacked.mean(dim=0)
        return averaged

    def to(self, device: torch.device) -> "SoftColumnMaskController":  # type: ignore[override]
        super().to(device)
        self._device = device
        return self

    def context_vector(self, task_name: str) -> torch.Tensor:
        task = self._tasks_by_name[task_name]
        return self._context_vector(task).detach()

    def _context_vector(self, task: TaskConfig) -> torch.Tensor:
        stats = self._task_input_stats.get(task.name)
        if self._config.include_input_in_prompt and stats is None:
            raise ValueError(f"No input statistics registered for task '{task.name}'.")
        return self.prompt_vectors.vector(task.prompt, task.task_id, stats)

    def promote_to_base(self, task_names: Sequence[str]) -> None:
        for name in task_names:
            if name not in self._task_to_bank:
                continue
            current_bank = self._task_to_bank[name]
            if current_bank == "base":
                continue
            try:
                self._bank_to_tasks[current_bank].remove(name)
            except (KeyError, ValueError):
                pass
            self._bank_to_tasks["base"].append(name)
            self._task_to_bank[name] = "base"
            if not self.use_key_query:
                new_size = self._bank_size("base")
                device = self.logits[name].device
                self.logits[name] = nn.Parameter(torch.zeros(new_size, device=device))
