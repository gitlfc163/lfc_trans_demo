
## 文本填充示例

from transformers import pipeline

# 初始化文本填充器
unmasker = pipeline("fill-mask") 

# 输入文本
results = unmasker("This course will teach you all about <mask> models.", top_k=2)

print(results)