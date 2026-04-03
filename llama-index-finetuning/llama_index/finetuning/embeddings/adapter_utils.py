"""Adapter utils."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import torch
import transformers
from llama_index.core.utils import print_text
from llama_index.embeddings.adapter import BaseAdapter
from sentence_transformers.util import cos_sim
from torch import Tensor, nn
from torch.optim import Optimizer
from tqdm.autonotebook import trange


class MyMultipleNegativesRankingLoss(nn.Module):
    """
    Multiple negatives ranking loss.

    This loss is similar to the one in sentence_transformers,
    but optimized for our own embeddings.

    """

    def __init__(
        self,
        model: BaseAdapter,
        scale: float = 20.0,
        similarity_fct: Optional[Callable] = None,
    ):
        """Define ranking loss."""
        super().__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = cos_sim if similarity_fct is None else similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, query_embeds: Tensor, context_embeds: Tensor) -> Tensor:
        """Forward pass."""
        # transform context embeds
        # context_embeds_2 = self.model.forward(context_embeds)
        query_embeds_2 = self.model.forward(query_embeds)

        scores = self.similarity_fct(query_embeds_2, context_embeds) * self.scale
        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )
        return self.cross_entropy_loss(scores, labels)


def train_model(
    model: BaseAdapter,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 1,
    steps_per_epoch: Optional[int] = None,
    warmup_steps: int = 10000,
    optimizer_class: Type[Optimizer] = torch.optim.AdamW,
    optimizer_params: Dict[str, Any] = {"lr": 2e-5},
    output_path: str = "model_output",
    max_grad_norm: float = 1,
    show_progress_bar: bool = True,
    verbose: bool = False,
    # callback: Callable[[float, int, int], None] = None,
    # scheduler: str = "WarmupLinear",
    # weight_decay: float = 0.01,
    # evaluation_steps: int = 0,
    # save_best_model: bool = True,
    # use_amp: bool = False,  # disable this option for now
    checkpoint_path: Optional[str] = None,
    checkpoint_save_steps: int = 500,
    # checkpoint_save_total_limit: int = 0,
) -> None:
    """Train model."""
    model.to(device)
    # TODO: hardcode loss now, make customizable later
    loss_model = MyMultipleNegativesRankingLoss(model=model)
    loss_model.to(device)

    # prepare optimizer/scheduler
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters: List[Dict[str, Any]] = [
        {
            "params": [p for n, p in param_optimizer],
        },
    ]
    optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
    if steps_per_epoch is None or steps_per_epoch == 0:
        steps_per_epoch = len(data_loader)
    num_train_steps = int(steps_per_epoch * epochs)
    scheduler_obj = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps
    )

    if verbose:
        print_text("> Prepared optimizer, scheduler, and loss model.\n", color="blue")

    global_step = 0
    data_iterator = iter(data_loader)

    # if checkpoint_path is specified, create if doesn't exist
    if checkpoint_path is not None:
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
        training_steps = 0
        loss_model.zero_grad()
        loss_model.train()
        for _ in trange(
            steps_per_epoch,
            desc="Iteration",
            smoothing=0.05,
            disable=not show_progress_bar,
        ):
            try:
                data = next(data_iterator)
            except StopIteration:
                data_iterator = iter(data_loader)
                data = next(data_iterator)

            query, context = data
            context = context.to(device)
            query = query.to(device)

            loss_value = loss_model(query, context)
            if verbose:
                print_text(
                    f"> [Epoch {epoch}] Current loss: {loss_value}\n", color="blue"
                )
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
            optimizer.step()

            optimizer.zero_grad()

            scheduler_obj.step()

            training_steps += 1
            global_step += 1

            # TODO: skip eval for now
            if checkpoint_path is not None and global_step % checkpoint_save_steps == 0:
                full_ck_path = Path(checkpoint_path) / f"step_{global_step}"
                model.save(str(full_ck_path))

    if verbose:
        print_text(f"> Finished training, saving to {output_path}\n", color="blue")

    # save model
    model.save(output_path)
