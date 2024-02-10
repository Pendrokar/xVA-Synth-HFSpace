import server as xvaserver
import gradio as gr

def predict(input):
	return ''

input_textbox = gr.Textbox(
    label="Input Text",
    lines=1,
    autofocus=True
)

gradio_app = gr.Interface(
    predict,
    input_textbox,
    title="xVASynth",
)

if __name__ == "__main__":
    gradio_app.launch()