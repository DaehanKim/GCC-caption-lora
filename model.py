import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from modeling_t5 import T5ForConditionalGeneration
from transformers import T5Tokenizer
from lora import inject_lora_layers, freeze_except_lora
from types import SimpleNamespace

class ExtendedSimpleNamespace(SimpleNamespace):
    def to_dict(self):
        return self.__dict__.copy()

class Captioner(nn.Module):
    def __init__(self, lora_patterns=["decoder.*.(SelfAttention|EncDecAttention).[qkvo]"], base_decoder_type="google/flan-t5-large"):
        super(Captioner, self).__init__()
        self.config = ExtendedSimpleNamespace(hidden_size=768)
        self.lora_patterns = lora_patterns
        # vision model
        vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        # extract encoder part
        self.vision_model = vision_model.vision_model
        # txt model
        self.tokenizer = T5Tokenizer.from_pretrained(base_decoder_type)
        text_model = T5ForConditionalGeneration.from_pretrained(base_decoder_type)
        print(f"### using model type : {base_decoder_type}")
        # projection to match latent dim
        self.bridge_proj = nn.Linear(768, text_model.config.d_model)

        # inject lora & freeze
        text_model = inject_lora_layers(text_model, patterns=lora_patterns)    
        self.text_model = freeze_except_lora(text_model)
        self.vision_model = freeze_except_lora(self.vision_model)

    def forward(self, pixel_values, labels):
        # image & caption is processed by collator
        # pixel_values: [batch_size, 3, 224, 224]
        # labels: [batch_size, max_len]
        vision_output = self.vision_model(pixel_values = pixel_values)
        vision_output.last_hidden_state = self.bridge_proj(vision_output.last_hidden_state)
        text_output = self.text_model(encoder_outputs=vision_output, labels=labels)
        return text_output
    
    def generate(self, pixel_values, generation_config):
        vision_output = self.vision_model(pixel_values = pixel_values)
        vision_output.last_hidden_state = self.bridge_proj(vision_output.last_hidden_state)
        gens = self.text_model.generate(encoder_outputs=vision_output, generation_config=generation_config)
        txts = self.tokenizer.batch_decode(gens, skip_special_tokens=True)
        return txts
