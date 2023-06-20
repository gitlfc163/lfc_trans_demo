# 导入evaluate包 
import evaluate
# pip install sklearn

# Load the accuracy metric
accuracy_metric = evaluate.load("accuracy")

# Evaluate the model
accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0])
