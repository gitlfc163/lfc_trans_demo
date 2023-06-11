# 情感分析示例

# 导入pipeline
from transformers import pipeline

# 英文示例
# 加载pipeline, 并传入英文模型
classifier = pipeline("sentiment-analysis")

# classifier 接受一个字符串，并返回一个字典，其中包含标签和分数
results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
# 打印结果
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

# 中文示例
classifierCH = pipeline("sentiment-analysis","bert-base-chinese")

# classifier 接受一个字符串，并返回一个字典，其中包含标签和分数
resultsCH = classifierCH(["我们很高兴向您展示 🤗 Transformers资料库。", "我们希望您不要讨厌它。"])
# 打印结果
for result in resultsCH:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
    