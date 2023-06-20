import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer

data_file = "./data/ChnSentiCorp_htl_all.csv" # 数据文件路径，数据需要提前下载
model_name = "hfl/rbt3" # 所使用模型

# 加载数据集
dataset = load_dataset("csv", data_files=data_file)
dataset = dataset.filter(lambda x: x["review"] is not None)
datasets = dataset["train"].train_test_split(0.2)

# 数据集处理
tokenizer = AutoTokenizer.from_pretrained(model_name)

def process_function(examples):
    tokenized_examples = tokenizer(examples["review"], max_length=64, truncation=True)
    tokenized_examples["labels"] = examples["label"]
    return tokenized_examples

tokenized_datasets = datasets.map(process_function, batched=True)

# 构建评估函数
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# 训练器配置
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

args = TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=128,
    num_train_epochs=5,
    weight_decay=0.01,
    output_dir="model_for_seqclassification",
    logging_steps=10,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True
)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

# 训练与评估
trainer.train()

trainer.evaluate()