# 零样本分类示例

# 导入pipeline函数
from transformers import pipeline

# 英文示例
# pipeline 接受一个字符串，并返回一个字典，其中包含预测的标签和概率。
# 标签和概率的排序是按概率从高到低的。
classifierEN = pipeline("zero-shot-classification")

# 预测标签
resultsEN = classifierEN(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
print(resultsEN)


# 中文示例
# pipeline 接受一个字符串，并返回一个字典，其中包含预测的标签和概率。
# 标签和概率的排序是按概率从高到低的。
classifierCH = pipeline('zero-shot-classification','bert-base-chinese')

# 预测标签
resultsCH = classifierCH(
    "这是一门关于Transformers的课程",
    candidate_labels=["教育", "政治", "商业"],
)
print(resultsCH)