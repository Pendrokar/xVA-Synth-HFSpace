import os
import sys
import copy
import time
import requests
import json
from huggingface_hub import hf_hub_download
import gradio as gr
from gradio_client import Client

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
	deepmoji_checked
):
	wav_path, response = client.predict(
		input_text,	# str  in 'Input Text' Textbox component
		voice,	# Literal['ccby_nvidia_hifi_6670_M', 'ccby_nv_hifi_11614_F', 'ccby_nvidia_hifi_11697_F', 'ccby_nvidia_hifi_12787_F', 'ccby_nvidia_hifi_6097_M', 'ccby_nvidia_hifi_6671_M', 'ccby_nvidia_hifi_8051_F', 'ccby_nvidia_hifi_9017_M', 'ccby_nvidia_hifi_9136_F', 'ccby_nvidia_hifi_92_F']  in 'Voice' Radio component
		lang,	# Literal['en', 'de', 'es', 'it', 'fr', 'ru', 'tr', 'la', 'ro', 'da', 'vi', 'ha', 'nl', 'zh', 'ar', 'uk', 'hi', 'ko', 'pl', 'sw', 'fi', 'hu', 'pt', 'yo', 'sv', 'el', 'wo', 'jp']  in 'Language' Radio component
		pacing,	# float (numeric value between 0.5 and 2.0) in 'Duration' Slider component
		pitch,	# float (numeric value between 0 and 1.0) in 'Pitch' Slider component
		energy,	# float (numeric value between 0.1 and 1.0) in 'Energy' Slider component
		anger,	# float (numeric value between 0 and 1.0) in '😠 Anger' Slider component
		happy,	# float (numeric value between 0 and 1.0) in '😃 Happiness' Slider component
		sad,	# float (numeric value between 0 and 1.0) in '😭 Sadness' Slider component
		surprise,	# float (numeric value between 0 and 1.0) in '😮 Surprise' Slider component
		deepmoji_checked, # bool
		api_name="/predict"
	)
	json_data = json.loads(response)
	arpabet = json_data['arpabet'].replace('|<PAD>|', ' ')

	return [
		wav_path,
		arpabet,
		round(json_data['em_angry'][0], 2),
		round(json_data['em_happy'][0], 2),
		round(json_data['em_sad'][0], 2),
		round(json_data['em_surprise'][0], 2)
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

with gr.Blocks() as demo:
	gr.Markdown("# xVASynth TTS")

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
			deepmoji_checkbox = gr.Checkbox(label="Use DeepMoji", info="Auto adjust emotional values")

	# Event handling using click
	btn = gr.Button("Generate")

	with gr.Row():  # Main row for inputs and language selection
		with gr.Column():  # Input column
			output_wav = gr.Audio(label="22kHz audio output", type="filepath", editable=False)
		with gr.Column():  # Input column
			output_arpabet = gr.Textbox(label="ARPAbet", interactive=False)

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
			surprise_slider
		]
	)

	language_radio.change(
		set_default_text,
		inputs=[language_radio, deepmoji_checkbox],
		outputs=[input_textbox, deepmoji_checkbox]
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
	print('running Gradio interface')
	# gradio_app.launch()
	client = Client("Pendrokar/xVASynth")

	demo.launch()
