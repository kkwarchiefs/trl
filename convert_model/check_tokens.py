from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
model_path = "./GLM"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
inputs = tokenizer("reward model's hash [gMASK]", return_tensors="pt")
for key in inputs:
    inputs[key] = inputs[key][:, :-1]
print(inputs)
print(tokenizer.decode(inputs['input_ids'][0]))
ge_inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=12)
ge_inputs2 = tokenizer.build_inputs_for_generation(inputs, targets='我也不知道', max_gen_length=12, padding=True)

print(tokenizer.decode(ge_inputs2['input_ids'][0]))
print(ge_inputs['input_ids'], ge_inputs['position_ids'], ge_inputs['generation_attention_mask'])
print(ge_inputs2['input_ids'], ge_inputs2['position_ids'], ge_inputs2['attention_mask'])
