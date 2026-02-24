import gradio as gr


def greet(name):
    return f"Hello, {name}!"


demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(label="Name", placeholder="Enter your name..."),
    outputs=gr.Textbox(label="Greeting"),
    title="Gradio Demo",
    description="A simple Gradio demo application.",
)

if __name__ == "__main__":
    demo.launch()
