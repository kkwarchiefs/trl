from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
model_path = "./GLM"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
inputs = tokenizer("reward model's hash [gMASK]", return_tensors="pt")
for key in inputs:
    inputs[key] = inputs[key][:, :-1]
print(inputs)
# print(tokenizer.decode(inputs['input_ids'][0]))
# ge_inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=12)
ge_inputs2 = tokenizer.build_inputs_for_generation(inputs, targets='我也不知道我也很想知道怎么处理', max_gen_length=32, padding=True)
print(inputs['input_ids'], inputs['input_ids'].shape)

ge_inputs_idx = ge_inputs2['input_ids'].tolist()[0]
print(ge_inputs_idx, len(ge_inputs_idx))
start_idx = ge_inputs_idx.index(50006)+1
print(inputs['input_ids'].shape[1])
end_idx = ge_inputs_idx.index(50007)
print('start_end:', start_idx, end_idx)
new_mask = [0]*start_idx + [1]*(end_idx-start_idx) + [0]*(ge_inputs2['input_ids'].shape[1]-end_idx)
print(new_mask, len(new_mask))
print(tokenizer.decode(ge_inputs2['input_ids'][0]))
# print(ge_inputs['input_ids'], ge_inputs['position_ids'], ge_inputs['generation_attention_mask'])
# print(ge_inputs2['input_ids'], ge_inputs2['position_ids'], ge_inputs2['attention_mask'])
