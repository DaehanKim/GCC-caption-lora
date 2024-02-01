from transformers import Trainer 
import inspect
import random
import warnings
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from transformers import GenerationConfig


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
import wandb
import deepspeed



class MyTrainer(Trainer):
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        # Sample and save to game log if requested (for one batch to save time)
        # Generate random indices within the range of the total number of samples
        num_samples = len(dataloader.dataset)
        random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

        # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
        random_batch_dataset = dataloader.dataset.select(random_indices)
        captions = random_batch_dataset["caption"][:16]
        random_batch = self.data_collator(random_batch_dataset)
        # random_batch = self._prepare_inputs(random_batch)

        # reduce to 16 samples
        random_batch["pixel_values"] = random_batch["pixel_values"][:16]
        random_batch["labels"] = random_batch["labels"][:16]

        generation_config = GenerationConfig(**{
            "max_new_tokens": 128,
            "top_p": 0.9,
            "temperature": 0.5,
            "repetition_penalty": 1.2,
            "do_sample" : True
        })
        eval_model = self.model.eval()
        txts = eval_model.generate(pixel_values=random_batch["pixel_values"].to(eval_model.text_model.device), generation_config=generation_config)

        self.log(
            {
                "Generated Captions": wandb.Table(columns=["pred_caption", "gt_caption", "image"], data=list(zip(txts, captions, [wandb.Image(e) for e in random_batch["pixel_values"]])))
            }
        )
        self.state.log_history.pop()

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output
    
    # def log(self, logs: Dict[str, float]) -> None:
    #     """
    #     Log `logs` on the various objects watching training, including stored metrics.

    #     Args:
    #         logs (`Dict[str, float]`):
    #             The values to log.
    #     """
    #     # logs either has 'loss' or 'eval_loss'
    #     train_eval = "train" if "loss" in logs else "eval"
    #     # Add averaged stored metrics to logs
    #     for key, metrics in self._stored_metrics[train_eval].items():
    #         logs[key] = torch.tensor(metrics).mean().item()
    #     del self._stored_metrics[train_eval]
    #     return super().log(logs)