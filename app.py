import os
import sys
import time
import requests
from subprocess import Popen, PIPE
import threading
import gradio as gr


def run_xvaserver():
	try:
		# start the process without waiting for a response
		print('Running xVAServer subprocess...')
		xvaserver = Popen(['python', 'server.py'], stdout=PIPE, stderr=PIPE, universal_newlines=True)
	except:
		print('Could not run xVASynth.')
		sys.exit(0)

	# Wait for a moment to ensure the server starts up
	time.sleep(10)

	# Check if the server is running
	if xvaserver.poll() is not None:
		print("Web server failed to start.")
		sys.exit(0)

	# contact local xVASynth server; ~2 second timeout
	print('Attempting to connect to xVASynth...')
	response = requests.get('http://0.0.0.0:8008')
	response.raise_for_status()  # If the response contains an HTTP error status code, raise an exception

	print('xVAServer running on port 8008')

	# load default voice model
	# load_model()

	predicted = predict('test', 1.0)
	print(predicted)

	# Read and print stdout and stderr of the subprocess
	while True:
		output = xvaserver.stdout.readline()
		if output == '' and xvaserver.poll() is not None:
			break
		if output:
			print(output.strip())

		error = xvaserver.stderr.readline()
		if error == '' and xvaserver.poll() is not None:
			break
		if error:
			print(error.strip(), file=sys.stderr)

	# Wait for the process to exit
	xvaserver.wait()

def load_model():
	model_type = 'xVAPitch'
	language = 'en'

	data = {
		'outputs': None,
		'version': '3.0',
		'model': 'ccby/ccby_nvidia_hifi_6670_M',
		'modelType': model_type,
		'base_lang': language,
		'pluginsContext': '{}',
	}
	response = requests.post('http://0.0.0.0:8008/loadModel', json=data)
	response.raise_for_status()  # If the response contains an HTTP error status code, raise an exception
	return

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
	response = requests.post('http://0.0.0.0:8008/synthesize', json=data)
	response.raise_for_status()  # If the response contains an HTTP error status code, raise an exception
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
	title="xVASynth (WIP)",
)


if __name__ == "__main__":
	# Run the web server in a separate thread
	web_server_thread = threading.Thread(target=run_xvaserver)
	print('Starting xVAServer thread')
	web_server_thread.start()

	print('running Gradio interface')
	gradio_app.launch()

	# Wait for the web server thread to finish (shouldn't be reached in normal execution)
	web_server_thread.join()
