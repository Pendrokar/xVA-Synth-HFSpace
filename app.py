import os
import sys
import time
import requests
import json
from subprocess import Popen, PIPE
import threading
from huggingface_hub import hf_hub_download
import gradio as gr

hf_model_name = "Pendrokar/xvapitch_nvidia"
hf_cache_models_path = '/home/user/.cache/huggingface/hub/models--Pendrokar--xvapitch_nvidia/snapshots/61b10e60b22bc21c1e072f72f1108b9c2b21e94c/'
models_path = '/home/user/.cache/huggingface/hub/models--Pendrokar--xvapitch_nvidia/snapshots/61b10e60b22bc21c1e072f72f1108b9c2b21e94c/'

try:
	os.symlink('/home/user/.cache/huggingface/hub/models--Pendrokar--TorchMoji/snapshots/58217568daaf64d3621245dd5c88c94e651a08d6/', '/home/user/app/resources/app/plugins/deepmoji_plugings/model')
except:
	print('Failed to create symlink to DeepMoji model, may already be there.')

voice_models = [
	("Male #6671", "ccby_nvidia_hifi_6671_M"),
	("Male #6670", "ccby_nvidia_hifi_6670_M"),
	("Male #9017", "ccby_nvidia_hifi_9017_M"),
	("Male #6097", "ccby_nvidia_hifi_6097_M"),
	("Female #92", "ccby_nvidia_hifi_92_F"),
	("Female #11697", "ccby_nvidia_hifi_11697_F"),
	("Female #12787", "ccby_nvidia_hifi_12787_F"),
	("Female #11614", "ccby_nv_hifi_11614_F"),
	("Female #8051", "ccby_nvidia_hifi_8051_F"),
	("Female #9136", "ccby_nvidia_hifi_9136_F"),
]
current_voice_model = None
base_speaker_emb = ''

# order ranked by similarity to English due to the xVASynth's use of ARPAbet instead of IPA
languages = [
    ("🇬🇧 EN", "en"),
    ("🇩🇪 DE", "de"),
    ("🇪🇸 ES", "es"),
    ("🇮🇹 IT", "it"),
    ("🇳🇱 NL", "nl"),
    ("🇵🇹 PT", "pt"),
    ("🇵🇱 PL", "pl"),
    ("🇷🇴 RO", "ro"),
    ("🇸🇪 SV", "sv"),
    ("🇩🇰 DA", "da"),
    ("🇫🇮 FI", "fi"),
    ("🇭🇺 HU", "hu"),
    ("🇬🇷 EL", "el"),
    ("🇫🇷 FR", "fr"),
    ("🇷🇺 RU", "ru"),
    ("🇺🇦 UK", "uk"),
    ("🇹🇷 TR", "tr"),
    ("🇸🇦 AR", "ar"),
    ("🇮🇳 HI", "hi"),
    ("🇯🇵 JP", "jp"),
    ("🇰🇷 KO", "ko"),
    ("🇨🇳 ZH", "zh"),
    ("🇻🇳 VI", "vi"),
    ("🇻🇦 LA", "la"),
    ("HA", "ha"),
    ("SW", "sw"),
    ("🇳🇬 YO", "yo"),
    ("WO", "wo"),
]

# Translated from English by DeepMind's Gemini Pro
default_text = {
	"ar": "هذا هو صوتي.",
	"da": "Sådan lyder min stemme.",
	"de": "So klingt meine Stimme.",
	"el": "Έτσι ακούγεται η φωνή μου.",
	"en": "This is what my voice sounds like.",
	"es": "Así suena mi voz.",
	"fi": "Näin ääneni kuulostaa.",
	"fr": "Voici à quoi ressemble ma voix.",
	"ha": "Wannan ne muryata ke.",
	"hi": "यह मेरी आवाज़ कैसी लगती है।",
	"hu": "Így hangzik a hangom.",
	"it": "Così suona la mia voce.",
	"jp": "これが私の声です。",
	"ko": "여기 제 목소리가 어떤지 들어보세요.",
	"la": "Haec est vox mea sonans.",
	"nl": "Dit is hoe mijn stem klinkt.",
	"pl": "Tak brzmi mój głos.",
	"pt": "É assim que minha voz soa.",
	"ro": "Așa sună vocea mea.",
	"ru": "Вот как звучит мой голос.",
	"sv": "Såhär låter min röst.",
	"sw": "Sauti yangu inasikika hivi.",
	"tr": "Benim sesimin sesi böyle.",
	"uk": "Ось як звучить мій голос.",
	"vi": "Đây là giọng nói của tôi.",
	"wo": "Ndox li neen xewnaal ma.",
	"yo": "Ìyí ni ohùn mi ńlá.",
	"zh": "这是我的声音。",
}

def run_xvaserver():
	# start the process without waiting for a response
	print('Running xVAServer subprocess...\n')
	xvaserver = Popen(['python', f'{os.path.dirname(os.path.abspath(__file__))}/resources/app/server.py'], stdout=PIPE, stderr=PIPE, cwd=f'{os.path.dirname(os.path.abspath(__file__))}/resources/app/')

	# Wait for a moment to ensure the server starts up
	time.sleep(10)

	# Check if the server is running
	if xvaserver.poll() is not None:
		print("Web server failed to start.")
		sys.exit(0)

	# contact local xVASynth server
	print('Attempting to connect to xVASynth...')
	try:
		response = requests.get('http://0.0.0.0:8008')
		response.raise_for_status()  # If the response contains an HTTP error status code, raise an exception
	except requests.exceptions.RequestException as err:
		print('Failed to connect!')
		return

	print('xVAServer running on port 8008')

	# load default model
	load_model("ccby_nvidia_hifi_6671_M")

	# Wait for the process to exit
	xvaserver.wait()

def load_model(voice_model_name):
	model_path =  models_path + voice_model_name

	model_type = 'xVAPitch'
	language = 'en'

	data = {
		'outputs': None,
		'version': '3.0',
		'model': model_path,
		'modelType': model_type,
		'base_lang': language,
		'pluginsContext': '{}',
	}

	embs = base_speaker_emb

	try:
		response = requests.post('http://0.0.0.0:8008/loadModel', json=data, timeout=60)
		response.raise_for_status()  # If the response contains an HTTP error status code, raise an exception
		current_voice_model = voice_model_name

		with open(model_path + '.json', 'r', encoding='utf-8') as f:
		    voice_model_json = json.load(f)
		embs = voice_model_json['games'][0]['base_speaker_emb']
	except requests.exceptions.RequestException as err:
		print('Failed to load voice model!')

	return embs

def predict(
	input_text,
	voice,
	lang,
	pacing,
	pitch,
	energy,
	anger,
	happy,
	sad,
	surprise,
	use_deepmoji
):
	# grab only the first 1000 characters
	input_text = input_text[:1000]

	# load voice model if not the current model
	if (current_voice_model != voice):
		base_speaker_emb = load_model(voice)

	model_type = 'xVAPitch'
	pace = pacing if pacing else 1.0
	save_path = '/tmp/xvapitch_audio_sample.wav'
	language = lang
	use_sr = 0
	use_cleanup = 0

	pluginsContext = {}
	pluginsContext["mantella_settings"] = {
		"emAngry": (anger if anger > 0 else 0),
		"emHappy": (happy if happy > 0 else 0),
		"emSad": (sad if sad > 0 else 0),
		"emSurprise": (surprise if surprise > 0 else 0),
		"run_model": use_deepmoji
	}


	data = {
		'pluginsContext': json.dumps(pluginsContext),
		'modelType': model_type,
		# pad with whitespaces as a workaround to avoid cutoffs
		'sequence': input_text.center(len(input_text) + 2, ' '),
		'pace': pace,
		'outfile': save_path,
		'vocoder': 'n/a',
		'base_lang': language,
		'base_emb': base_speaker_emb,
		'useSR': use_sr,
		'useCleanup': use_cleanup,
	}

	try:
		response = requests.post('http://0.0.0.0:8008/synthesize', json=data, timeout=60)
		response.raise_for_status()  # If the response contains an HTTP error status code, raise an exception
		# response_data = json.loads(response.text)
	except requests.exceptions.RequestException as err:
		print('Failed to synthesize!')
		print('server.log contents:')
		with open('resources/app/server.log', 'r') as f:
			print(f.read())
		return ['', err]

	print('server.log contents:')
	with open('resources/app/server.log', 'r') as f:
		print(f.read())

	return [save_path, response.text]

input_textbox = gr.Textbox(
	label="Input Text",
	value="This is what my voice sounds like.",
	info="Also accepts ARPAbet symbols placed within {} brackets.",
	lines=1,
	max_lines=5,
	autofocus=True
)
pacing_slider = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Duration")
pitch_slider = gr.Slider(0, 1.0, value=0.5, step=0.05, label="Pitch", visible=False)
energy_slider = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Energy", visible=False)
anger_slider = gr.Slider(0, 1.0, value=0, step=0.05, label="😠 Anger", info="Tread lightly beyond 0.9")
happy_slider = gr.Slider(0, 1.0, value=0, step=0.05, label="😃 Happiness", info="Tread lightly beyond 0.7")
sad_slider = gr.Slider(0, 1.0, value=0, step=0.05, label="😭 Sadness", info="Duration increased when beyond 0.2")
surprise_slider = gr.Slider(0, 1.0, value=0, step=0.05, label="😮 Surprise", info="Does not play well with Happiness with either being beyond 0.3")
voice_radio = gr.Radio(
	voice_models,
	value="ccby_nvidia_hifi_6671_M",
	label="Voice",
	info="NVIDIA HIFI CC-BY-4.0 xVAPitch voice model"
)

def set_default_text(lang):
	input_textbox = gr.Textbox(
		label="Input Text",
		value=default_text[lang],
		lines=1,
		max_lines=5,
		autofocus=True
	)

language_radio = gr.Radio(
	languages,
	value="en",
	label="Language",
	info="Will be more monotone and have an English accent. Tested mostly by a native Briton."
)
# language_radio.change(set_default_text)
deepmoji_checkbox = gr.Checkbox(label="Use DeepMoji", info="Auto adjust emotional values")

gradio_app = gr.Interface(
	predict,
	[
		input_textbox,
		voice_radio,
		language_radio,
		pacing_slider,
		pitch_slider,
		energy_slider,
		anger_slider,
		happy_slider,
		sad_slider,
		surprise_slider,
		deepmoji_checkbox
	],
	outputs=[
		gr.Audio(label="22kHz audio output", type="filepath"),
		gr.Textbox(label="xVASynth Server Response")
	],
	title="xVASynth (WIP)",
	clear_btn=gr.Button(visible=False)
	# examples=[
	# 	["Once, I headed in much deeper. But I doubt I'll ever do that again.", 1],
	# 	["You love hurting me, huh?", 1.5],
	# 	["Ah, I see. Well, I'm afraid I can't help with that.", 1],
	# 	["Embrace your demise!", 1],
	# 	["Never come back!", 1]
	# ],
	# cache_examples=None
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
