import fire
from transformers import CLIPProcessor
from transformers import CLIPModel
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from typing import Any, Dict, List, Optional, Union
from transformers.tokenization_utils_base import PaddingStrategy
import torch
from dataclasses import dataclass
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch
import numpy as np
from collections import defaultdict
import json
from pathlib import Path

@dataclass
class DataCollatorForClipScore(DataCollatorWithPadding):
    tokenizer: PreTrainedTokenizerBase = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [torch.tensor(feature["input_ids"], dtype=torch.long).squeeze() for feature in features]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        preds_input_ids = [torch.tensor(feature["preds_input_ids"], dtype=torch.long).squeeze() for feature in features]
        preds_input_ids = torch.nn.utils.rnn.pad_sequence(preds_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        pixel_values = torch.cat([torch.tensor(feature["pixel_values"], dtype=torch.float) for feature in features], dim=0)

        batch = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "preds_input_ids": preds_input_ids
        }

        return batch

def clipscore(hyp_text="inference_results/large-attlora-lr5e-4-epoch4_beam_hyp.txt"):
    # load clip-large model
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    # load text file
    with open(hyp_text, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    num_hyp = len(lines)
    
    # to keep ordering of valid dataset after filtering
    # load reference dataset
    ref_dataset = load_dataset("parquet", data_files="dataset/filtered_valid_for_eval.parquet")["train"]
    refs = set(ref_dataset["caption"])

    # load and process dataset
    def data_tokenize(examples):
        try:
            out = clip_processor(images=[examples['image']], text=[examples['caption']] ,return_tensors="pt", padding=True)
            examples["pixel_values"], examples["input_ids"] = out.pixel_values, out.input_ids
            examples["preds_input_ids"] = clip_processor(text=examples["preds"], return_tensors="pt", padding=True).input_ids
        except:
            examples["pixel_values"] = None
            examples["input_ids"] = None
            examples["preds_input_ids"] = None
        return examples
    
    def filter_none(examples):
        return examples["pixel_values"] is not None and examples["input_ids"] is not None
    def filter_caption(examples):
        return examples["caption"] in refs

    valid_dataset = load_dataset(
        "imagefolder",
        data_dir="dataset/valid",
        split="train"
    )
    valid_dataset = valid_dataset.filter(
        filter_caption, num_proc=10
    )

    # add predictions to dataset
    valid_dataset = valid_dataset.add_column("preds", lines)
    # debug
    # valid_dataset = valid_dataset.select(range(100))
    # lines = lines[:100]
    valid_dataset = valid_dataset.map(
        data_tokenize, load_from_cache_file=True, num_proc = 1, batch_size=1, writer_batch_size=100
    )
    valid_dataset = valid_dataset.remove_columns(["image"])
    valid_dataset = valid_dataset.filter(
        filter_none, num_proc=10
    )

    assert num_hyp == len(valid_dataset), "number of hypothesis and number of images are not equal"
    
    # make loader
    loader = DataLoader(valid_dataset, batch_size=32, collate_fn=DataCollatorForClipScore(tokenizer=clip_processor.tokenizer))

    # get score
    scores = defaultdict(list)
    for batch in loader:
        with torch.inference_mode():
            batch = {k: v.cuda() for k, v in batch.items()}
            gt_outputs = clip(**{
                "input_ids" : batch["input_ids"],
                "pixel_values" : batch["pixel_values"]
            })
            pred_outputs = clip(**{
                "input_ids" : batch["preds_input_ids"],
                "pixel_values" : batch["pixel_values"]
            })
            # compute cosine similarity 
            # between outputs.text_embeds, and outputs.image_embeds
            gt_score = torch.nn.functional.cosine_similarity(gt_outputs.text_embeds, gt_outputs.image_embeds)
            pred_score = torch.nn.functional.cosine_similarity(pred_outputs.text_embeds, pred_outputs.image_embeds)
            scores["gt"].extend(gt_score.squeeze().cpu().tolist())
            scores["pred"].extend(pred_score.squeeze().cpu().tolist())
        
    # average scores
    scores["gt_avg"] = np.array(scores["gt"]).mean()
    scores["pred_avg"] = np.array(scores["pred"]).mean()
    print(f"average gt clip score for {hyp_text}:\n{scores['gt_avg']:.3f}")
    print(f"average pred clip score for {hyp_text}:\n{scores['pred_avg']:.3f}")

    # save scores
    Path("inference_results").mkdir(exist_ok=True)
    with open(f"inference_results/{Path(hyp_text).stem}_clip_score.json", 'w') as f:
        json.dump(scores, f)

if __name__  == "__main__":
    fire.Fire(clipscore)