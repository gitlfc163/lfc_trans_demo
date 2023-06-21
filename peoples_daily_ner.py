# 序列标注(命名实体识别)示例

# 导入相关包
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer

# 加载数据集
datasets = load_dataset("peoples_daily_ner")
label_list = datasets["train"].features["ner_tags"].feature.names

# 数据集处理
tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")

# 定义数据处理函数
def process_function(examples):
    tokenized_examples = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=64)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_examples.word_ids(batch_index=i)  
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
        labels.append(label_ids)
    tokenized_examples["labels"] = labels
    return tokenized_examples
tokenized_datasets = datasets.map(process_function, batched=True)

# 构建评估函数
seqeval_metric = evaluate.load("seqeval")

# 定义评估函数
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels, mode="strict", scheme="IOB2")
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 配置训练器-哈工大开源的rbt3分词器模型
model = AutoModelForTokenClassification.from_pretrained("hfl/rbt3", num_labels=len(label_list))

args = TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=128,
    num_train_epochs=5,
    weight_decay=0.01,
    output_dir="model_for_tokenclassification",
    logging_steps=10,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=DataCollatorForTokenClassification(tokenizer),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 训练与评估
trainer.train()
trainer.evaluate(tokenized_datasets["test"])