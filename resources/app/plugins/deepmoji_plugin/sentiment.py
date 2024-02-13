import os
from os.path import abspath, dirname

import configparser
import logging
import os
import sys

isDev = setupData["isDev"]
logger = setupData["logger"]

import sys
root_path = f'.' if isDev else f'./resources/app' # The root directories are different in dev/prod
if not isDev:
    sys.path.append("./resources/app")

# import DeepMoji/TorchMoji
sys.path.append(f"{root_path}/plugins/deepmoji_plugin/DeepMoji")

# TOP # emoticons out of 64 to take into account
emoji_count = 10
text_scores = []
# the plugin's default settings
plugin_settings = {}
isBatch = False
isXVAPitch = True
isEnglish = True

# previous sentence
prev_sentence = ''
# Previous emotional modifier values
last_em_angry = float(0)
last_em_happy = float(0)
last_em_sad = float(0)
last_em_surprise = float(0)

def scoreText(text):
	return text
from plugins.deepmoji_plugin.xvasynth_torchmoji import scoreText
import csv

def setup(data=None):
	logger.log(f'Setting up plugin. App version: {data["appVersion"]} | CPU only: {data["isCPUonly"]} | Development mode: {data["isDev"]}')
	# Show test emoji in console; can crash due to encoding issues
	try:
		print("DeepMoji Plugin - emoji smily console print test: \U0001F604")
	except:
		pass

def pre_load_model(data=None):
	# reset last em values
	global last_em_angry, last_em_happy, last_em_sad, last_em_surprise,\
		isBatch, isXVAPitch, isEnglish, prev_sentence,\
		plugin_settings, configparser, emoji_count

	# reload settings from INI
	config = configparser.ConfigParser()
	with open(f'{root_path}/plugins/deepmoji_plugin/deepmoji.ini', encoding='utf-8') as stream:
		# xVASynth saves without INI sections
	    config.read_string("[top]\n" + stream.read())
	plugin_settings = dict(config['top'])

	emoji_count = int(plugin_settings['emoji_count'])

	isBatch = False
	isXVAPitch = True
	isEnglish = True
	prev_sentence = ''
	last_em_angry = float(0)
	last_em_happy = float(0)
	last_em_sad = float(0)
	last_em_surprise = float(0)
	logger.log("last_em reset")

def fetch_text(data=None):
	global plugin_settings, emoji_count, text_scores, scoreText, isXVAPitch, isEnglish, prev_sentence
	isBatch = False

	text_scores = [data["sequence"]]
	try:
		# editor second generation test
		if len(data["pitch"]):
			logger.warning("DeepMoji analysis skipped due to customized editor values")
			return
	except:
		pass

	if (
		(
			plugin_settings["load_deepmoji_model"]=="false"
			or plugin_settings["load_deepmoji_model"]==False
		)
		and data["pluginsContext"]["mantella_settings"]["run_model"]==False
	):
		logger.log("DeepMoji model skipped")
		return

	if (data['modelType'] != 'xVAPitch'):
		logger.log("DeepMoji can affect only xVAPitch models")
		isXVAPitch = False
		return
	if (data['base_lang'] != 'en'):
		logger.log("DeepMoji works only with English text")
		isEnglish = False
		return

	if (
		plugin_settings["append_prev_sentence"]=="false"
		or plugin_settings["append_prev_sentence"]==False
	):
		prev_sentence = ''
	else:
		prev_sentence += ' '

	# add previous sentence for a better flow
	text_scores = scoreText(prev_sentence + data["sequence"], emoji_count)
	logger.log(text_scores)

	text_scores[0] = data["sequence"]

def fetch_text_batch(data=None):
	global isBatch, plugin_settings, emoji_count, text_scores, scoreText, isXVAPitch, isEnglish, prev_sentence
	isBatch = True

	text_scores = [data["linesBatch"][0][0]]
	if (
		plugin_settings["use_on_batch"]=="false"
		or plugin_settings["use_on_batch"]==False
	):
		logger.debug("DeepMoji Plugin skipped on batch")
		return

	if (
		plugin_settings["load_deepmoji_model"]=="false"
		or plugin_settings["load_deepmoji_model"]==False
	):
		logger.debug("DeepMoji model skipped")
		return

	if (data['modelType'] != 'xVAPitch'):
		logger.debug("DeepMoji can affect only xVAPitch models")
		isXVAPitch = False
		return

	if (data["linesBatch"][0][8] != 'en'):
		logger.debug("DeepMoji works only with English text")
		return

	if (
		plugin_settings["append_prev_sentence"]=="false"
		or plugin_settings["append_prev_sentence"]==False
	):
		prev_sentence = ''
	else:
		prev_sentence += ' '

	# logger.info(data)
	try:
		logger.debug(data["linesBatch"][0][0])
		text_scores = scoreText(prev_sentence + data["linesBatch"][0][0], emoji_count)
		logger.debug(text_scores)
	except:
		logger.error("Could not parse line")
		return

# text_scores
# (['Text', 'Top#%',
#                     'Emoji_1', 'Emoji_2', 'Emoji_3', 'Emoji_4', 'Emoji_5',
#                     'Pct_1', 'Pct_2', 'Pct_3', 'Pct_4', 'Pct_5'])

def adjust_values(data=None):
	global root_path, os, csv, example_helper,\
		isBatch, isXVAPitch, isEnglish, logger, emoji_count, text_scores, plugin_settings,\
		prev_sentence, last_em_angry, last_em_happy, last_em_sad, last_em_surprise

	if (
		isBatch
		and (
			plugin_settings["use_on_batch"] == "false"
			or plugin_settings["use_on_batch"] == False
		)
	):
		logger.debug("DeepMoji Plugin skipped on batch")
		return

	if (isXVAPitch == False):
		logger.log("DeepMoji can affect only xVAPitch models")
		return

	em_angry = float(0)
	em_happy = float(0)
	em_sad = float(0)
	em_surprise = float(0)
	emojis = ''

	if (
		isXVAPitch
		and isEnglish
		and len(text_scores) > 1
	):
		# DeepMoji works only with English text
		with open(f'{root_path}/plugins/deepmoji_plugin/emoji_unicode_emotions.csv', encoding='utf-8') as csvfile:
			reader = csv.DictReader(csvfile)
			index = 0
			for emoji_row in reader:
				for em_index in range(emoji_count):
					# emotion is not one of detected emotions?
					if (index != text_scores[2 + em_index]):
						# skip
						continue

					em_angry += float(emoji_row['anger']) * float(text_scores[2 + em_index + emoji_count])
					em_happy += float(emoji_row['happiness']) * float(text_scores[2 + em_index + emoji_count])
					em_sad += float(emoji_row['sadness']) * float(text_scores[2 + em_index + emoji_count])
					em_surprise += float(emoji_row['surprise']) * float(text_scores[2 + em_index + emoji_count])
					emojis += emoji_row['emoji']+' '
				index += 1

	# Show Top emojis in console
	try:
		# can crash on batch
		print(emojis)
	except:
		pass

	em_emotion_max = 0.8
	em_angry_max = 0.6
	try:
		em_angry += float(data["pluginsContext"]["mantella_settings"]["emAngry"]) * 100
		em_angry_max = 1
	except:
		pass
	try:
		em_happy += float(data["pluginsContext"]["mantella_settings"]["emHappy"]) * 100
	except:
		pass
	try:
		em_sad += float(data["pluginsContext"]["mantella_settings"]["emSad"]) * 100
	except:
		pass
	try:
		em_surprise += float(data["pluginsContext"]["mantella_settings"]["emSurprise"]) * 100
	except:
		pass

	# HF
	if (len(text_scores) > 1):
		# top_em highest wins all
		top_em = max(
			em_angry,
			em_happy,
			em_sad
		)
		em_angry = em_angry if (em_angry == top_em) else 0
		em_happy = em_happy if (em_happy == top_em) else 0
		# amplified sadness ratio
		em_sad = (em_sad * 3) if (em_sad == top_em) else 0

		# amplifier
		ratio = float(plugin_settings['amplifier_ratio'])
	else:
		ratio = 1.0
		em_emotion_max = 1
		em_angry_max = 1

	logger.log(f'Amplifier ratio: {ratio}')
	hasExcMark = False
	if ('!!!' in text_scores[0]):
		ratio += 2
		em_angry_max = max(em_angry_max, 0.92)
		logger.log(f"!!! detected => em_angry_max: {em_angry_max}")
		logger.log(f'!!! Ratio: {ratio}')
		hasExcMark = True
	elif (
		('!!' in text_scores[0])
		or ('!?!' in text_scores[0])
	):
		ratio += 1.5
		em_angry_max = max(em_angry_max, 0.82)
		logger.log(f"!! detected => em_angry_max: {em_angry_max}")
		logger.log(f'!! Ratio: {ratio}')
		hasExcMark = True
	elif ('!' in text_scores[0]):
		ratio += 1
		em_angry_max = max(em_angry_max, 0.7)
		logger.log(f"! detected => em_angry_max: {em_angry_max}")
		logger.log(f'! Ratio: {ratio}')
		hasExcMark = True

	# HF
	if (len(text_scores) <= 1):
		em_angry_max = 1
		ratio = 1

	# final values
	em_angry = min(em_angry_max, em_angry / 100 * ratio) if em_angry > 0 else 0
	em_happy = min(em_emotion_max, em_happy / 100 * ratio) if em_happy > 0 else 0
	em_sad = min(em_emotion_max, em_sad / 100 * ratio) if em_sad > 0 else 0
	em_surprise = min(em_emotion_max, em_surprise / 100 * ratio) if em_surprise > 0 else 0


	# do average of previous if above threshold and last_em is not higher
	last_top_em = max(last_em_angry, last_em_happy, last_em_sad)
	if (
		(em_angry > 0)
		and (last_top_em > 0.1)
		and (last_top_em < em_angry)
	):
		logger.log(f"em_angry before avg: {em_angry}")
		em_angry = (em_angry + last_top_em) / 2
	if (
		(em_happy > 0)
		and (last_top_em > 0.1)
		and (last_top_em < em_happy)
	):
		logger.log(f"em_happy before avg: {em_happy}")
		em_happy = (em_happy + last_top_em) / 2
	if (
		(em_sad > 0)
		and (last_top_em > 0.1)
		and (last_top_em < em_sad)
	):
		logger.log(f"em_sad before avg: {em_sad}")
		em_sad = (em_sad + last_top_em) / 2
	if (
		(em_surprise > 0)
		and (last_em_surprise > 0.05)
		and (last_em_surprise < em_surprise)
	):
		logger.log(f"em_surprise before avg: {em_surprise}")
		em_surprise = (em_surprise + last_em_surprise) / 2


	# adjust the values within data
	if (em_angry > 0):
		logger.log(f"Adjusting em_angry: {em_angry}")
		adjusted_pacing = False
		for line_i in range(len(data["emAngry"])):
			for char_i in range(len(data["emAngry"][line_i])):
				data["emAngry"][line_i][char_i] = em_angry
				data["hasDataChanged"] = True

		# slower the speech if above threshold
		# as the pacing goes way too fast when past this point
		if (em_angry >= 0.7 and hasExcMark):
			try:
				data["pace"] *= (1 + em_angry / 2)
				logger.log(f"Adjusting pacing: {1 + em_angry / 2}")
			except:
				pass

	if (
		em_happy > 0 and (
			em_happy >= em_surprise
			or em_surprise < 0.3
		)
	):
		# em_surprise & em_happy overamplify each other
		logger.log(f"Adjusting em_happy: {em_happy}")
		for line_i in range(len(data["emHappy"])):
			for char_i in range(len(data["emHappy"][line_i])):
				data["emHappy"][line_i][char_i] = em_happy
				data["hasDataChanged"] = True

	if (em_sad > 0):
		logger.log(f"Adjusting em_sad: {em_sad}")
		for line_i in range(len(data["emSad"])):
			for char_i in range(len(data["emSad"][line_i])):
				data["emSad"][line_i][char_i] = em_sad
				data["hasDataChanged"] = True

		# slower the speech if above threshold
		if (em_sad > 0.2):
			try:
				data["pace"] *= (1 + em_sad / 3)
				logger.log(f"Adjusting pacing: {1 + em_sad / 3}")
			except:
				pass

	if (
		em_surprise > 0 and (
			em_surprise > em_happy
			or em_happy < 0.3
		)
	):
		# em_surprise & em_happy overamplify each other
		logger.log(f"Adjusting em_surprise: {em_surprise}")
		for line_i in range(len(data["emSurprise"])):
			for char_i in range(len(data["emSurprise"][line_i])):
				data["emSurprise"][line_i][char_i] = em_surprise
				data["hasDataChanged"] = True

	prev_sentence = text_scores[0]
	last_em_angry = em_angry
	last_em_happy = em_happy
	last_em_sad = em_sad
	last_em_surprise = em_surprise

	return
