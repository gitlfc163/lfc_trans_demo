
# 文本填充示例

from transformers import pipeline

# 英文示例
# 初始化文本填充器
unmasker = pipeline("fill-mask") 
# 输入文本
results = unmasker("This course will teach you all about <mask> models.", top_k=2)
print(results)


# 中文示例
# 初始化文本填充器
generatorCH = pipeline('fill-mask','bert-base-chinese')
# 生成文本
resultsCH =generatorCH("生活的真谛是[MASK]。")
# 输出结果
print(resultsCH)