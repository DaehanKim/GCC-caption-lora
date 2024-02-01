from model import Captioner
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from typing import Any, Dict, List, Optional, Union
from transformers.tokenization_utils_base import PaddingStrategy
import torch
import torch.nn as nn
from transformers import DataCollatorWithPadding
from dataclasses import dataclass
import logging
import sys
import os
from transformers import CLIPProcessor
from PIL import Image
import requests
from transformers import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_utils import set_seed
import datasets 
import transformers
from transformers import T5Tokenizer
import math
import time 
from PIL import UnidentifiedImageError
from dataclasses import dataclass, field
import multiprocessing as mp

mp.set_start_method('spawn', force=True)

@dataclass
class ScriptArguments:
    """
    custom arguments
    """

    # data parameters
    base_decoder_type: str = field(default="google/flan-t5-large", metadata={"help": "base decoder type"})
    custom_debug: bool = field(default=False, metadata={"help": "debug mode"})
    lora_targets: str = field(default="attn", metadata={"help": "lora targets. should contain subset of [attn, ffn]."})

@dataclass
class DataCollatorForImageCaptioningWithPadding(DataCollatorWithPadding):
    image_processor: CLIPProcessor = None
    tokenizer: PreTrainedTokenizerBase = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        # 성공적으로 처리된 이미지들만을 사용하여 labels 생성
        labels = [torch.tensor(feature["labels"], dtype=torch.long) for feature in features]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        pixel_values = torch.cat([torch.tensor(feature["pixel_values"], dtype=torch.float) for feature in features], dim=0)

        batch = {
            "pixel_values": pixel_values,
            "labels": labels,
        }

        return batch

def main():
    parser = HfArgumentParser(
        (TrainingArguments, ScriptArguments)
    )
    training_args, script_args = parser.parse_args_into_dataclasses()

    # setting remove_unused_column to false to avoid removing necessary columns prior to colating batch.
    training_args.remove_unused_columns = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # get lora patterns
    lora_patterns = []
    if "attn" in script_args.lora_targets:
        lora_patterns += ["decoder.*.(SelfAttention|EncDecAttention).[qkvo]"]
    if "ffn" in script_args.lora_targets:
        lora_patterns += ["decoder.*.DenseReluDense.(wi_0|wi_1|wo)"]
    
    assert lora_patterns, "lora patterns should not be empty"

    model = Captioner(lora_patterns=lora_patterns, base_decoder_type=script_args.base_decoder_type)
    tokenizer = T5Tokenizer.from_pretrained(script_args.base_decoder_type)
    image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16").image_processor

    def data_tokenize(examples):
        tokenized = tokenizer(
            examples['caption'],
            return_tensors="pt",
            return_length=True,
            add_special_tokens=True,
            truncation=True,
            max_length=tokenizer.model_max_length
        )
        examples["labels"] = tokenized["input_ids"][0]
        examples["length"] = tokenized.length.item()
        try:
            examples["pixel_values"] = image_processor(images=[examples['image']], return_tensors="pt", padding=True).pixel_values
        except:
            examples["pixel_values"] = None
        return examples
    
    # import functools
    # data_tokenize_image_processor = functools.partial(data_tokenize, image_processor=image_processor)

    def filter_none(examples):
        return examples["pixel_values"] is not None

    with training_args.main_process_first():
        if script_args.custom_debug:
            valid_dataset = load_dataset(
                "imagefolder",
                data_dir="dataset/valid",
                split="train"
            )
            valid_dataset = valid_dataset.map(
                data_tokenize, load_from_cache_file=True, num_proc = 1, batch_size=1, writer_batch_size=100
            )
            valid_dataset = valid_dataset.remove_columns(["image"])
            valid_dataset = valid_dataset.filter(
                filter_none, num_proc=10
            )
            train_dataset = valid_dataset
        else:            
            train_dataset = load_dataset(
                "imagefolder",
                data_dir="dataset/train",
                split="train"
            )
            valid_dataset = load_dataset(
                "imagefolder",
                data_dir="dataset/valid",
                split="train"
            )
            train_dataset = train_dataset.map(
                data_tokenize, load_from_cache_file=True, batch_size=1, num_proc = 1
            )
            valid_dataset = valid_dataset.map(
                data_tokenize, load_from_cache_file=True, batch_size=1, num_proc = 1
            )
            train_dataset = train_dataset.remove_columns(["image"])
            valid_dataset = valid_dataset.remove_columns(["image"])
            train_dataset = train_dataset.filter(
                filter_none, num_proc=10
            )
            valid_dataset = valid_dataset.filter(
                filter_none, num_proc=10
            )
            

    from custom_trainer import MyTrainer
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForImageCaptioningWithPadding(tokenizer=tokenizer, image_processor=image_processor),
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = len(valid_dataset)
        metrics["eval_samples"] = max_eval_samples
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()