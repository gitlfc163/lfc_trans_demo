#问答系统示例

# 导入
from transformers import pipeline

# 定义模型
question_answerer = pipeline("question-answering") 

# 输入                        
results=question_answerer( question="Where do I work?", context="My name is Sylvain and I work at Hugging Face in Brooklyn", )

# 输出
print(results)