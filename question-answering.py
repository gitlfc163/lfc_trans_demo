#问答系统英文示例

# 导入
from transformers import pipeline

# 定义模型
question_answerer = pipeline("question-answering") 
# 输入                        
results=question_answerer( question="Where do I work?", context="My name is Sylvain and I work at Hugging Face in Brooklyn", )
# 输出
print(results)

#问答系统中文示例

# 导入
from transformers import AutoModelForQuestionAnswering,AutoTokenizer,pipeline

# 定义模型
model = AutoModelForQuestionAnswering.from_pretrained('luhua/chinese_pretrain_mrc_roberta_wwm_ext_large')
# 定义分词器
tokenizer = AutoTokenizer.from_pretrained('luhua/chinese_pretrain_mrc_roberta_wwm_ext_large')
# 定义问答系统
question_answerer_zh = pipeline("question-answering", model=model, tokenizer=tokenizer)
# 输入
input_zh = {'question': "我住在哪里？",'context': "我叫沃尔夫冈，我住在柏林。"}
# 输入
results_zh=question_answerer_zh(input_zh)
# 输出
print(results_zh)
