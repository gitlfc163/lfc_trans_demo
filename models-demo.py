
# 预训练模型加载与使用

# 导入BertModel
#from transformers import BertModel
# 加载bert-base-chinese模型
#model= BertModel.from_pretrained("bert-base-chinese")


# 便捷的模型加载方式-AutoModel
from transformers import AutoTokenizer,AutoModel
# 加载bert-base-chinese模型
model = AutoModel.from_pretrained("bert-base-chinese")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

model(**tokenizer("弱小的我也有大梦想", return_tensors="pt"))