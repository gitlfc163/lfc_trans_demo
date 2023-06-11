# 文本生成示例

from transformers import pipeline

# 初始化文本生成器
generator = pipeline("text-generation")

# 生成文本
results =generator("In this course, we will teach you how to")

print(results)