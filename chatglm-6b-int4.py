
# pip install cpm_kernels

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True) # .half().cuda()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)