# 文本生成示例

# from transformers import pipeline

# 英文示例
# 初始化文本生成器
# generator = pipeline("text-generation")
# 生成文本
# resultsEN =generator("In this course, we will teach you how to")
# print(resultsEN)


# 文本生成中文示例

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("openbmb/cpm-bee-1b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("openbmb/cpm-bee-1b", trust_remote_code=True) # 
result = model.generate({"input": "今天天气不错，", "<ans>": ""}, tokenizer)
print(result)
