
# 多条的数据同时处理
text = ["弱小的我也有大梦想",
        "有梦想谁都了不起",
        "追逐梦想的心，比梦想本身，更可贵"]
res = tokenizer(text, padding="max_length", max_length=15)
print(inputs)