# given a checkpoint and a validation dataset, evaluate the model in terms of CIDEr, METEOR, and BLEU scores
import fire
from pathlib import Path
from model import Captioner
from safetensors import safe_open
from datasets import load_dataset
from transformers import T5Tokenizer
from transformers import CLIPProcessor
from torch.utils.data import DataLoader
from transformers import GenerationConfig
import torch
from transformers import set_seed
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from transformers import PreTrainedTokenizerBase
from typing import Any, Dict, List, Optional, Union
from transformers.tokenization_utils_base import PaddingStrategy
import torch
from dataclasses import dataclass
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader


@dataclass
class DataCollatorForBestOfNSampling(DataCollatorWithPadding):
    tokenizer: PreTrainedTokenizerBase = None
    clip_tokenizer : PreTrainedTokenizerBase = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = [torch.tensor(feature["labels"], dtype=torch.long).squeeze() for feature in features]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        clip_large_input_ids = [torch.tensor(feature["input_ids_for_clip_large"], dtype=torch.long).squeeze() for feature in features]
        clip_large_input_ids = torch.nn.utils.rnn.pad_sequence(clip_large_input_ids, batch_first=True, padding_value=self.clip_tokenizer.pad_token_id)
        pixel_values = torch.cat([torch.tensor(feature["pixel_values"], dtype=torch.float) for feature in features], dim=0)
        clip_large_pixel_values = torch.cat([torch.tensor(feature["pixel_values_for_clip_large"], dtype=torch.float) for feature in features], dim=0)

        batch = {
            "pixel_values": pixel_values,
            # "input_ids": input_ids,
            "labels": labels,
            "input_ids_for_clip_large": clip_large_input_ids,
            "pixel_values_for_clip_large": clip_large_pixel_values
        }

        return batch

def pick_best(candidates, pixel_values, clip_model, clip_processor):
    input_ids = clip_processor(text=candidates, return_tensors="pt", padding=True).input_ids
    with torch.inference_mode():
        out = clip_model(pixel_values=pixel_values.unsqueeze(0).cuda(), input_ids=input_ids.cuda())
        cos_sim = torch.nn.functional.cosine_similarity(out.text_embeds, out.image_embeds.repeat(len(candidates),1), dim=-1)
        idx = cos_sim.argmax(dim=-1).item()
    return candidates[idx]

def inference(checkpoint="/outputs/large-attn,ffnlora-lr5e-4-epoch4/checkpoint-5088", best_n=10, valid_dir="dataset/valid"):
    model_name = Path(checkpoint).parent.stem
    lora_patterns = []
    if "att" in model_name:
        lora_patterns += ["decoder.*.(SelfAttention|EncDecAttention).[qkvo]"]
    if "ffn" in model_name:
        lora_patterns += ["decoder.*.DenseReluDense.(wi_0|wi_1|wo)"]

    print(model_name, lora_patterns)

    model_type = "google/flan-t5-large" if "large" in model_name else "google/flan-t5-base"
    model = Captioner(lora_patterns=lora_patterns, base_decoder_type=model_type)
    tensors = {}
    with safe_open(Path(checkpoint)/"model.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    model.load_state_dict(tensors)

    # load and process dataset
    tokenizer = T5Tokenizer.from_pretrained(model_type)
    image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16").image_processor

    # load evaluation model : clip-large
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

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
            examples["pixel_values_for_clip_large"] = clip_processor(images=[examples['image']], return_tensors="pt", padding=True).pixel_values
            examples["input_ids_for_clip_large"] = clip_processor(text=[examples['caption']], return_tensors="pt", padding=True).input_ids
        except:
            examples["pixel_values"] = None
            examples["pixel_values_for_clip_large"] = None
            examples["input_ids_for_clip_large"] = None
        return examples
    
    def filter_none(examples):
        return examples["pixel_values"] is not None

    valid_dataset = load_dataset(
        "imagefolder",
        data_dir=valid_dir,
        split="train"
    )
    valid_dataset = valid_dataset.map(
        data_tokenize, load_from_cache_file=True, num_proc = 1, batch_size=1, writer_batch_size=100
    )
    valid_dataset = valid_dataset.remove_columns(["image"])
    valid_dataset = valid_dataset.filter(
        filter_none, num_proc=10
    )


    loader = DataLoader(valid_dataset, batch_size=64, collate_fn=DataCollatorForBestOfNSampling(tokenizer=tokenizer, clip_tokenizer=clip_processor.tokenizer))
    model = model.cuda().eval()
    # "beam_size" : 10,
    generation_config = GenerationConfig(**{
        "do_sample" : True,
        "top_p" : 0.9,
        "temperature" : 0.5,
        "repetition_penalty" : 1.2,
        "max_new_tokens" : 20,
        "num_return_sequences" : best_n
    })

    result_text = []
    gt_text = []
    set_seed(42) # fix generation seed
    for batch in tqdm(loader, total=len(loader)):
        with torch.inference_mode():
            txts = model.generate(pixel_values = batch["pixel_values"].cuda(), generation_config=generation_config)
            # chunk texts with best_n size 
            chunks = [txts[i:i+best_n] for i in range(0, len(txts), best_n)]
            # pick best from each chunk
            best_txts = [pick_best(chunk, batch["pixel_values_for_clip_large"][i], clip, clip_processor) for i, chunk in enumerate(chunks)]
            result_text += best_txts
            gt_text += tokenizer.batch_decode(torch.where(batch['labels']!=-100, batch['labels'], 0) , skip_special_tokens=True)

    # make a result directory
    Path("inference_results").mkdir(exist_ok=True)

    with open(f"inference_results/{model_name}_bestof{best_n}_hyp.txt", "w") as f:
        f.write("\n".join(result_text))
    with open(f"inference_results/{model_name}_bestof{best_n}_ref.txt", "w") as f:
        f.write("\n".join(gt_text))

if __name__ == "__main__":
    fire.Fire(inference)