import sys
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from transformers import T5Tokenizer
sys.path.append("..")
from modeling_t5 import T5ForConditionalGeneration
from lora import inject_lora_layers, freeze_except_lora

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

    # input_text = ""
    # input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    
    # out = text_model(encoder_outputs = vision_output, decoder_input_ids=input_ids)
    # print(out)

    for n, p in vision_model.named_parameters():
        print(n, p.requires_grad)

    outputs = text_model.generate(encoder_outputs=vision_output)
    print(tokenizer.decode(outputs[0]))