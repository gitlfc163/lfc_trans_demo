# 句子分词示例

# 单条数据的处理方式
# 加载分词器
from transformers  import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 输入句子
text = '今天天气不错'

# 分词
tokens = tokenizer.tokenize(text)
# 打印
print(tokens)

# 查看词表
# print(tokenizer.vocab)

# 词序列转数字序列
ids=tokenizer.convert_tokens_to_ids(tokens)
print(ids)

# 数字序列转词序列
tokens2=tokenizer.convert_ids_to_tokens(ids)
print(tokens2)


# 填充
ids2=tokenizer.encode(text,max_length=15,padding='max_length')
print(ids2)

# 裁剪
ids3=tokenizer.encode(text,max_length=5,padding='max_length',truncation=True)
print(ids3)

# attention_mask 与 token_type_id
ids = tokenizer.encode(text, padding="max_length", max_length=15)
attention_mask = [1 if idx != 0 else 0 for idx in ids]
token_type_ids = [0] * len(ids)
print(attention_mask)
print(token_type_ids)

# 快速调用方式
inputs = tokenizer.encode_plus(text, padding="max_length", max_length=15)
print(inputs)

# 最直接调用方式
inputs = tokenizer(text, padding="max_length", max_length=15)
print(inputs)

# 多条的数据同时处理
text = ["弱小的我也有大梦想",
        "有梦想谁都了不起",
        "追逐梦想的心，比梦想本身，更可贵"]
res = tokenizer(text, padding="max_length", max_length=15)
print(res)
