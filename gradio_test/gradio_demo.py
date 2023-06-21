
# Hello World示例

# 导入gradio包
import gradio as gr

# 定位函数
def greet(name):
  return "Hello " + name + "!"
demo = gr.Interface(fn=greet, inputs="text", outputs="text")

# 指定gradio运行端口
demo.launch(server_port=9091)