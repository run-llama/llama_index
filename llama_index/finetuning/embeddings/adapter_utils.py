"""Adapter utils."""

import torch
from torch import nn, Tensor
from typing import Optional, Dict, Type, Callable
from torch.optim import Optimizer
import transformers
import os
import shutil
from tqdm.autonotebook import trange
import json
from llama_index.embeddings.adapter import LinearLayer
from llama_index.bridge.langchain import print_text

from sentence_transformers.util import cos_sim


class MyMultipleNegativesRankingLoss(nn.Module):
    """This loss is similar to the one in sentence_transformers, but optimized for our own embeddings."""

    def __init__(
        self,
        model: LinearLayer,
        scale: float = 20.0,
        similarity_fct: Optional[Callable] = None,
    ):
        """Define ranking loss."""

        super(MyMultipleNegativesRankingLoss, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = cos_sim if similarity_fct is None else similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, query_embeds: Tensor, context_embeds: Tensor) -> Tensor:
        """Forward pass."""

        # transform context embeds
        context_embeds_2 = self.model.forward(context_embeds)

        scores = self.similarity_fct(query_embeds, context_embeds_2) * self.scale
        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )
        return self.cross_entropy_loss(scores, labels)


def train_model(
    model: LinearLayer,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 1,
    steps_per_epoch: Optional[int] = None,
    # scheduler: str = "WarmupLinear",
    warmup_steps: int = 10000,
    optimizer_class: Type[Optimizer] = torch.optim.AdamW,
    optimizer_params: Dict[str, object] = {"lr": 2e-5},
    # weight_decay: float = 0.01,
    # evaluation_steps: int = 0,
    output_path: str = "model_output",
    # save_best_model: bool = True,
    max_grad_norm: float = 1,
    # use_amp: bool = False,  # disable this option for now
    callback: Callable[[float, int, int], None] = None,
    show_progress_bar: bool = True,
    verbose: bool = False
    # checkpoint_path: str = None,
    # checkpoint_save_steps: int = 500,
    # checkpoint_save_total_limit: int = 0,
) -> None:
    """Train model."""

    model.to(device)
    # NOTE: hardcode loss
    loss_model = MyMultipleNegativesRankingLoss(model=model)
    loss_model.to(device)

    # prepare optimizer/scheduler
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer],
            # "weight_decay": weight_decay,
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
        print_text(f"> Prepared optimizer, scheduler, and loss model.\n", color="blue")

    global_step = 0
    data_iterator = iter(data_loader)

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

            # TODO: skip saving checkpoint

    if verbose:
        print_text(f"> Finished training, saving to {output_path}\n", color="blue")

    # save model
    model.save(output_path)
