# datasets示例

from datasets import load_dataset
# 加载数据集
# datasets = load_dataset("beyond/chinese_clean_passages_80m")
datasets = load_dataset("madao33/new-title-chinese")

# 数据查看
# print(datasets["train"][0])

print(datasets["train"][:2])