import os
import requests
from subprocess import Popen, PIPE
import gradio as gr

try:
	# start the process without waiting for a response
	Popen(['python', 'server.py'], stdout=PIPE, stderr=PIPE)
except:
	import logging
	logging.error(f'Could not run xVASynth.')
	sys.exit(0)

def predict(input, pacing):
	model_type = 'xVAPitch'
	line = 'Test'
	pace = pacing if pacing else 1.0
	save_path = 'test.wav'
	language = 'en'
	base_speaker_emb = []
	use_sr = 0
	use_cleanup = 0

	data = {
	    'modelType': model_type,
	    'sequence': line,
	    'pace': pace,
	    'outfile': save_path,
	    'vocoder': 'n/a',
	    'base_lang': language,
	    'base_emb': base_speaker_emb,
	    'useSR': use_sr,
	    'useCleanup': use_cleanup,
	}
	requests.post('http://localhost:8008/synthesize', json=data)
	return 22100, os.open(save_path, "rb")

input_textbox = gr.Textbox(
    label="Input Text",
    lines=1,
    autofocus=True
)
slider = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Pacing")

gradio_app = gr.Interface(
    predict,
    [
    	input_textbox,
    	slider
	],
    outputs= "audio",
    title="xVASynth",
)

if __name__ == "__main__":
    gradio_app.launch()