import gradio as gr
import requests
from subprocess import Popen, PIPE

try:
	# start the process without waiting for a response
	Popen(['python', 'server.py'], stdout=PIPE, stderr=PIPE)
except:
	import logging
	logging.error(f'Could not run xVASynth.')
	sys.exit(0)

def predict(input):
	model_type = 'xVAPitch'
	line = 'Test'
	pace = 1.0
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
	return ''

input_textbox = gr.Textbox(
    label="Input Text",
    lines=1,
    autofocus=True
)

gradio_app = gr.Interface(
    predict,
    input_textbox,
    outputs="text",
    title="xVASynth",
)

if __name__ == "__main__":
    gradio_app.launch()