import gradio as gr
def greet(name,intensity):
    return "Hi " + name + "!" * int(intensity)
demo = gr.Interface(
    fn= greet, inputs=["text","slider"],
    outputs=["text"]
)
demo.launch(server_name="127.0.0.1", server_port=7680)