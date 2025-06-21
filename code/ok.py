from transformers import BertForMaskedLM, BertTokenizer, pipeline
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert", use_safetensors=True)

unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)

results = unmasker("D L I P T S S K L V V [MASK] D T S L Q V K K A F F A L V T")
print(results)
