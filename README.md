# GCC-caption-lora
This repo contains codes to train an image captioning model using GCC dataset. You can download, process, train lora weights injected to LM decoder and evaluate the results.

# Model Architecture
Captioner model is composed of CLIP image encoder and Flan-T5 decoder. Each input image is injected via cross-attention to the decoder, after a linear projection to fit the size of embedding. decoder's qkvo projection and feed forword layer can be augmented with lora weights.

![image](architecture.png)

# Files

- ```lora.py``` : minimal interface to inject lora weights to FLAN-T5 decoder.
- ```modeling_t5.py``` : modified from huggingface's ```modeling_t5.py``` to function with ```lora.py```
- ```train.py``` : training script
- ```evaluation.py``` : captioning inference script
- ```model.py``` : definition of Captioner model.
- ```preprocess_data.py``` : downloading images for dataset
- ```best_of_n_sample.py``` : captioning inference script using best-of-n sampling strategy.
- ```custom_trainer.py``` : huggingface trainer that supports ar-generation during training
- ```clipscore.py``` : clipscore computation
- ```nlg-eval``` : see [nlg-eval's documentation](https://github.com/Maluuba/nlg-eval). used for BLEU and CIDEr computation. 
- ```tests``` : tests to make sure model is working
- ```scripts``` : shell scripts to run python scripts

