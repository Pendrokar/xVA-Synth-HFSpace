import os
import sys
import time
import requests
import json
from subprocess import Popen, PIPE
import threading
from huggingface_hub import hf_hub_download
import gradio as gr

# try:
import resources.app.no_server as xvaserver
# except:
# 	print('server.log contents:')
# 	with open('resources/app/server.log', 'r') as f:
# 		print(f.read())


hf_model_name = "Pendrokar/xvapitch_nvidia"
hf_cache_models_path = '/home/user/.cache/huggingface/hub/models--Pendrokar--xvapitch_nvidia/snapshots/61b10e60b22bc21c1e072f72f1108b9c2b21e94c/'
models_path = '/home/user/.cache/huggingface/hub/models--Pendrokar--xvapitch_nvidia/snapshots/61b10e60b22bc21c1e072f72f1108b9c2b21e94c/'

# FIXME: currently hardcoded in DeepMoji code
# try:
# 	os.symlink('/home/user/.cache/huggingface/hub/models--Pendrokar--TorchMoji/snapshots/58217568daaf64d3621245dd5c88c94e651a08d6', '/home/user/app/resources/app/plugins/deepmoji_plugings/model', target_is_directory=True)
# except:
# 	print('Failed to create symlink to DeepMoji model, may already be there.')

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

	print('Loading voice model...')
	try:
		json_data = xvaserver.loadModel(data)
		current_voice_model = voice_model_name

		with open(model_path + '.json', 'r', encoding='utf-8') as f:
		    voice_model_json = json.load(f)
		embs = voice_model_json['games'][0]['base_speaker_emb']
	except requests.exceptions.RequestException as err:
		print(f'FAILED to load voice model: {err}')

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

	print('Synthesizing...')
	try:
		json_data = xvaserver.synthesize(data)
		# response = requests.post('http://0.0.0.0:8008/synthesize', json=data, timeout=60)
		# response.raise_for_status()  # If the response contains an HTTP error status code, raise an exception
		# json_data = json.loads(response.text)
	except requests.exceptions.RequestException as err:
		print('FAILED to synthesize: {err}')
		save_path = ''
		response = {'text': '{"message": "Failed"}'}
		json_data = {
			'arpabet': ['Failed'],
			'durations': [0],
			'em_anger': anger,
			'em_happy': happy,
			'em_sad': sad,
			'em_surprise': surprise,
		}

	# print('server.log contents:')
	# with open('resources/app/server.log', 'r') as f:
	# 	print(f.read())

	arpabet_html = '<h6>ARPAbet & Phoneme lengths</h6>'
	arpabet_symbols = json_data['arpabet'].split('|')
	utter_time = 0
	for symb_i in range(len(json_data['durations'])):
		# skip PAD symbol
		if (arpabet_symbols[symb_i] == '<PAD>'):
			continue

		length = float(json_data['durations'][symb_i])
		arpa_length = str(round(length/2, 1))
		arpabet_html += '<strong\
			class="arpabet"\
			style="padding: 0 '\
			+ str(arpa_length)\
			+'em"'\
			+f" title=\"{utter_time} + {length}\""\
			+'>'\
			+ arpabet_symbols[symb_i]\
			+ '</strong> '
		utter_time += round(length, 1)

	return [
		save_path,
		arpabet_html,
		round(json_data['em_angry'][0], 2),
		round(json_data['em_happy'][0], 2),
		round(json_data['em_sad'][0], 2),
		round(json_data['em_surprise'][0], 2),
		json_data
	]

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

def set_default_text(lang, deepmoji_checked):
	# DeepMoji only works on English Text
	# checkbox_enabled = True
	# if lang != 'en':
	# 	checkbox_enabled = False

	if lang == 'en':
		checkbox_enabled = gr.Checkbox(
			label="Use DeepMoji",
			info="Auto adjust emotional values",
			value=deepmoji_checked,
			interactive=True
		)
	else:
		checkbox_enabled = gr.Checkbox(
			label="Use DeepMoji",
			info="Works only with English!",
			value=False,
			interactive=False
		)

	return default_text[lang], checkbox_enabled  # Return the modified textbox (important for Blocks)

en_examples = [
	"This is what my voice sounds like.",
	"If there is anything else you need, feel free to ask.",
	"Amazing! Could you do that again?",
	"Why, I would be more than happy to help you!",
	"That was unexpected.",
	"How dare you! . You have no right.",
	"Ahh, well, you see. There is more to it.",
	"I can't believe she is gone.",
	"Stay out of my way!!!",
	# ARPAbet example
	"This { IH1 Z } { W AH1 T } { M AY1 } { V OY1 S } { S AW1 N D Z } like.",
]

def set_example_as_input(example_text):
	return example_text

def reset_em_sliders(
	deepmoji_enabled,
	anger,
	happy,
	sad,
	surprise
):
	if (deepmoji_enabled):
		return (0, 0, 0, 0)
	else:
		return (
			anger,
			happy,
			sad,
			surprise
		)

def toggle_deepmoji(
	checked,
	anger,
	happy,
	sad,
	surprise
):
	if checked:
		return (0, 0, 0, 0)
	else:
		return (
			anger,
			happy,
			sad,
			surprise
		)

language_radio = gr.Radio(
	languages,
	value="en",
	label="Language",
	info="Will be more monotone and have an English accent. Tested mostly by a native Briton."
)

_DESCRIPTION = '''
<div>
<a style="display:inline-block;" href="https://github.com/DanRuta/xVA-Synth"><img src='https://img.shields.io/github/stars/DanRuta/xVA-Synth?style=social'/></a>
<a style="display:inline-block; margin-left: .5em" href="https://discord.gg/nv7c6E2TzV"><img src='https://img.shields.io/discord/794590496202293278.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2'/></a>
<span style="display: inline-block;margin-left: .5em;vertical-align: top;"><a href="https://huggingface.co/spaces/Pendrokar/xVASynth?duplicate=true" style="" target="_blank"><img style="margin-bottom: 0em;display: inline;" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a> for a personal CPU-run</span>
</div>
'''

with gr.Blocks(css=".arpabet {display: inline-block; background-color: gray; border-radius: 5px; font-size: 120%; margin: 0.1em 0}") as demo:
	gr.Markdown("# xVASynth TTS")

	gr.HTML(label="description", value=_DESCRIPTION)

	with gr.Row():  # Main row for inputs and language selection
		with gr.Column():  # Input column
			input_textbox = gr.Textbox(
				label="Input Text",
				value="This is what my voice sounds like.",
				info="Also accepts ARPAbet symbols placed within {} brackets.",
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
			with gr.Row():
				with gr.Column():
					en_examples_dropdown = gr.Dropdown(
						en_examples,
						value=en_examples[0],
						label="Example dropdown",
						show_label=False,
						info="English Examples",
						visible=(language_radio.value == 'en')
					)
				with gr.Column():
					pacing_slider = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Duration")
		with gr.Column():  # Control column
			voice_radio = gr.Radio(
				voice_models,
				value="ccby_nvidia_hifi_6671_M",
				label="Voice",
				info="NVIDIA HIFI CC-BY-4.0 xVAPitch voice model"
			)
			pitch_slider = gr.Slider(0, 1.0, value=0.5, step=0.05, label="Pitch", visible=False)
			energy_slider = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Energy", visible=False)
			with gr.Row():  # Main row for inputs and language selection
				with gr.Column():  # Input column
					anger_slider = gr.Slider(0, 1.0, value=0, step=0.05, label="😠 Anger", info="Tread lightly beyond 0.9")
					sad_slider = gr.Slider(0, 1.0, value=0, step=0.05, label="😭 Sadness", info="Duration increased when beyond 0.2")
				with gr.Column():  # Input column
					happy_slider = gr.Slider(0, 1.0, value=0, step=0.05, label="😃 Happiness", info="Tread lightly beyond 0.7")
					surprise_slider = gr.Slider(0, 1.0, value=0, step=0.05, label="😮 Surprise", info="Can oversaturate Happiness")
			deepmoji_checkbox = gr.Checkbox(label="Use DeepMoji", info="Auto adjust emotional values", value=True)

	# Event handling using click
	btn = gr.Button("Generate", variant="primary")

	with gr.Row():  # Main row for inputs and language selection
		with gr.Column():  # Input column
			output_wav = gr.Audio(
				label="22kHz audio output",
				type="filepath",
				editable=False,
				autoplay=True
			)
		with gr.Column():  # Input column
			output_arpabet = gr.HTML(label="ARPAbet")

	btn.click(
		fn=predict,
		inputs=[
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
			output_wav,
			output_arpabet,
			anger_slider,
			happy_slider,
			sad_slider,
			surprise_slider,
			# xVAServer response
			gr.Textbox(visible=False)
		]
	)
	input_textbox.submit(
		fn=predict,
		inputs=[
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
			output_wav,
			output_arpabet,
			anger_slider,
			happy_slider,
			sad_slider,
			surprise_slider,
			# xVAServer response
			gr.Textbox(visible=False)
		]
	)

	language_radio.change(
		set_default_text,
		inputs=[language_radio, deepmoji_checkbox],
		outputs=[input_textbox, deepmoji_checkbox]
	)

	en_examples_dropdown.change(
		set_example_as_input,
		inputs=[en_examples_dropdown],
		outputs=[input_textbox]
	)

	deepmoji_checkbox.change(
		toggle_deepmoji,
		inputs=[
			deepmoji_checkbox,
			anger_slider,
			happy_slider,
			sad_slider,
			surprise_slider
		],
		outputs=[
			anger_slider,
			happy_slider,
			sad_slider,
			surprise_slider
		]
	)

	input_textbox.change(
		reset_em_sliders,
		inputs=[
			deepmoji_checkbox,
			anger_slider,
			happy_slider,
			sad_slider,
			surprise_slider
		],
		outputs=[
			anger_slider,
			happy_slider,
			sad_slider,
			surprise_slider
		]
	)

	voice_radio.change(
		reset_em_sliders,
		inputs=[
			deepmoji_checkbox,
			anger_slider,
			happy_slider,
			sad_slider,
			surprise_slider
		],
		outputs=[
			anger_slider,
			happy_slider,
			sad_slider,
			surprise_slider
		]
	)

if __name__ == "__main__":
	# Run the web server in a separate thread

	# print('Attempting to connect to local xVASynth server...')
	# try:
	# 	response = requests.get('http://0.0.0.0:8008')
	# 	response.raise_for_status()  # If the response contains an HTTP error status code, raise an exception
	# except requests.exceptions.RequestException as err:
	# 	print('Failed to connect to xVASynth!')
	# 	web_server_thread = threading.Thread(target=run_xvaserver)
	# 	print('Starting xVAServer thread')
	# 	web_server_thread.start()

	print('running Gradio interface')
	demo.launch()

	# Wait for the web server thread to finish (shouldn't be reached in normal execution)
	# web_server_thread.join()
