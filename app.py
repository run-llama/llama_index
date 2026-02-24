import os

import gradio as gr
from huggingface_hub import InferenceClient

MODEL_ID = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
HF_TOKEN = os.getenv("HF_TOKEN", None)

client = InferenceClient(token=HF_TOKEN)


def generate_image(prompt, negative_prompt, guidance_scale, num_steps, width, height, seed):
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    kwargs = {
        "prompt": prompt,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_steps,
        "width": width,
        "height": height,
    }

    if negative_prompt.strip():
        kwargs["negative_prompt"] = negative_prompt

    if seed >= 0:
        kwargs["seed"] = seed

    image = client.text_to_image(model=MODEL_ID, **kwargs)
    return image


with gr.Blocks(theme=gr.themes.Soft(), title="Text-to-Image Generator") as demo:
    gr.Markdown("# Text-to-Image Generator")
    gr.Markdown(f"Using model: **{MODEL_ID}**")

    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe the image you want to generate...",
                lines=3,
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                placeholder="What to avoid in the image...",
                lines=2,
            )

            with gr.Row():
                guidance_scale = gr.Slider(
                    minimum=1.0, maximum=20.0, value=7.5, step=0.5,
                    label="Guidance Scale",
                )
                num_steps = gr.Slider(
                    minimum=1, maximum=100, value=30, step=1,
                    label="Inference Steps",
                )

            with gr.Row():
                width = gr.Slider(
                    minimum=256, maximum=1344, value=1024, step=64,
                    label="Width",
                )
                height = gr.Slider(
                    minimum=256, maximum=1344, value=1024, step=64,
                    label="Height",
                )

            seed = gr.Number(label="Seed", value=-1, precision=0)
            gr.Markdown("*Set seed to -1 for random generation.*")

            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=4):
            output_image = gr.Image(label="Generated Image", type="pil")

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, guidance_scale, num_steps, width, height, seed],
        outputs=output_image,
    )

if __name__ == "__main__":
    demo.launch()
