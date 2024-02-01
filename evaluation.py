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

def inference(checkpoint="/outputs/large-attn,ffnlora-lr5e-4-epoch4/checkpoint-5088", decoding = "sample", valid_dir="dataset/valid"):
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


    from train import DataCollatorForImageCaptioningWithPadding
    loader = DataLoader(valid_dataset, batch_size=128 if decoding == "sample" else 48, collate_fn=DataCollatorForImageCaptioningWithPadding(tokenizer=tokenizer, image_processor=image_processor))
    model = model.cuda().eval()
    # "beam_size" : 10,
    if decoding == "sample":
        generation_config = GenerationConfig(**{
            "do_sample" : True,
            "top_p" : 0.9,
            "temperature" : 0.5,
            "repetition_penalty" : 1.2,
            "max_new_tokens" : 20
        })
    elif decoding=="beam":
        generation_config = GenerationConfig(**{
            "do_sample" : False,
            "num_beams" : 10,
            "max_length" : 20
        })

    result_text = []
    gt_text = []
    set_seed(42) # fix generation seed
    for batch in tqdm(loader, total=len(loader)):
        with torch.inference_mode():
            txts = model.generate(pixel_values = batch["pixel_values"].cuda(), generation_config=generation_config)
            # txts = tokenizer.batch_decode(gens, skip_special_tokens=True)
            result_text += txts
            gt_text += tokenizer.batch_decode(torch.where(batch['labels']!=-100, batch['labels'], 0) , skip_special_tokens=True)

    # make a result directory
    Path("inference_results").mkdir(exist_ok=True)

    with open(f"inference_results/{model_name}_{decoding}_hyp.txt", "w") as f:
        f.write("\n".join(result_text))
    with open(f"inference_results/{model_name}_{decoding}_ref.txt", "w") as f:
        f.write("\n".join(gt_text))

if __name__ == "__main__":
    fire.Fire(inference)