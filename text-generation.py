# 文本生成示例

from transformers import pipeline

# 英文示例
# 初始化文本生成器
generator = pipeline("text-generation")
# 生成文本
resultsEN =generator("In this course, we will teach you how to")
print(resultsEN)


# 文本生成中文示例

# pip3 install jieba
from transformers import CpmAntTokenizer, CpmAntForCausalLM

texts = "今天天气不错，"
model = CpmAntForCausalLM.from_pretrained("openbmb/cpm-ant-10b", mirror='tuna')
tokenizer = CpmAntTokenizer.from_pretrained("openbmb/cpm-ant-10b", mirror='tuna')
input_ids = tokenizer(texts, return_tensors="pt")
outputs = model.generate(**input_ids)
output_texts = tokenizer.batch_decode(outputs)

print(output_texts)