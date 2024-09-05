import torch
from transformers import LlamaForCausalLM, AutoTokenizer

# Load the model and tokenizer
# 定义模型路径
model_dir = "models/story/"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_dir)

model = LlamaForCausalLM.from_pretrained(model_dir)


input_text = "这是一个测试句子."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
# print(input_ids.shape)
# print(input_ids)

output = model(input_ids=input_ids, labels=input_ids)

print(output)


# def fn forward(input_ids):
#     # Run the model
#     output = model(input_ids=input_ids, labels=input_ids)
#     return output