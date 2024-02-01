import torch.nn as nn
import math
import re


class LoraLayer(nn.Module):
    '''
    lora layer for a linear layer
    '''
    def __init__(self, linear_layer : nn.Linear, config = None):
        super(LoraLayer, self).__init__()
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        self.alpha = 8
        self.r = 8
        self.lora_dropout = 0.1
        self.scaling = self.alpha / math.sqrt(self.r) # rslora : https://arxiv.org/abs/2312.03732
        self.dropout_layer = nn.Dropout(self.lora_dropout) if self.lora_dropout > 0 else nn.Identity()
        self.base_layer = linear_layer
        self.lora_a = nn.Linear(in_features, self.r, bias=False)
        self.lora_b = nn.Linear(self.r, out_features, bias=False)

        # initialization
        self.lora_a.weight.data.normal_(std=1/self.r)
        self.lora_b.weight.data.zero_()

    def forward(self, x):
        wx = self.base_layer(x)
        delta = self.lora_b(self.lora_a(self.dropout_layer(x))) * self.scaling
        return wx + delta


def inject_lora_layers(model, patterns=["decoder.*.(SelfAttention|EncDecAttention).[qkvo]"]):
    for name, module in model.named_modules():
        if any(re.search(pattern, name) for pattern in patterns):
            if isinstance(module, nn.Linear):
                parent_module = model
                name_components = name.split('.')
                for component in name_components[:-1]:  
                    parent_module = getattr(parent_module, component)

                setattr(parent_module, name_components[-1], LoraLayer(module))
    return model

def freeze_except_lora(model):
    for name, param in model.named_parameters():
        # `lora_a` 또는 `lora_b`를 이름에 포함하지 않는 파라미터를 동결합니다.
        if 'lora_a' not in name and 'lora_b' not in name:
            param.requires_grad = False
    return model


if __name__ == "__main__": 
    from PIL import Image
    import requests
    from transformers import CLIPProcessor, CLIPModel
    from modeling_t5 import T5ForConditionalGeneration
    from transformers import T5Tokenizer

    # vision model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    vision_model = model.vision_model
    # vision to text dim projection
    # latent_proj = nn.Linear(768, 512)
    # txt model
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    text_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

    # inject lora
    text_model = inject_lora_layers(text_model, patterns=["decoder.*.(SelfAttention|EncDecAttention).[qkvo]"])    
    text_model = freeze_except_lora(text_model)
    vision_model = inject_lora_layers(vision_model, patterns=["encoder.*.self_attn.(k_proj|q_proj|v_proj|o_proj)"])
    vision_model = freeze_except_lora(vision_model)

    # dummy data
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt", padding=True)
    vision_output = vision_model(**inputs) # B x 197 x 768
    # vision_output = latent_proj(vision_output.last_hidden_state) # B x 197 x 512

    input_text = ""
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    
    out = text_model(encoder_outputs = vision_output, decoder_input_ids=input_ids)
    # print(out)

    for n, p in vision_model.named_parameters():
        print(n, p.requires_grad)

    outputs = text_model.generate(inputs = input_ids, encoder_outputs=vision_output)
    print(tokenizer.decode(outputs[0]))
