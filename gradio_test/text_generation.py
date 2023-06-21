
# 英文文本生成示例

# 导入gradio包
import gradio as gr
from transformers import pipeline

# pipeine设置为gpt2

generator = pipeline('text-generation', model='gpt2')

# generate
def generate(text):
    result = generator(text, max_length=30, num_return_sequences=1)
    return result[0]["generated_text"]

examples = [
    ["今天天气不错，"],
    ["The Moon's orbit around Earth has"],
    ["The smooth Borealis basin in the Northern Hemisphere covers 40%"],
]

demo = gr.Interface(
    fn=generate,
    inputs=gr.inputs.Textbox(lines=5, label="Input Text"),
    outputs=gr.outputs.Textbox(label="Generated Text"),
    examples=examples
)

demo.launch(server_port=9089)