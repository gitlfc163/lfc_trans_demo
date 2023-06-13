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
from transformers import TextGenerationPipeline, AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("TsinghuaAI/CPM-Generate")
model = AutoModelWithLMHead.from_pretrained("TsinghuaAI/CPM-Generate")

text_generator = TextGenerationPipeline(model, tokenizer)
text_generator('清华大学', max_length=50, do_sample=True, top_p=0.9)