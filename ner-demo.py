# 命名实体识别示例

from transformers import pipeline

# 分词 + 命名实体识别 + 组块 + 去除歧义
ner = pipeline("ner", grouped_entities=True) 

# 输入文本
results =ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")

print(results)