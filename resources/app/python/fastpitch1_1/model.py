import os
import re
import json
import codecs
import ffmpeg
import argparse

import torch
import torch.nn as nn
from python.fastpitch1_1 import models

from scipy.io.wavfile import write
from torch.nn.utils.rnn import pad_sequence
from python.common.text import text_to_sequence, sequence_to_text


ARPAbet_replacements_dict = {
    "YO": "IY0 UW0",
    "UH": "UH0",
    "AR": "R",
    "EY": "EY0",
    "A": "AA0",
    "AW": "AW0",
    "X": "K S",
    "CX": "K HH",
    "AO": "AO0",
    "PF": "P F",
    "AY": "AY0",
    "OE": "OW0 IY0",
    "IY": "IY0",
    "EH": "EH0",
    "OY": "OY0",
    "IH": "IH0",
    "H": "HH"
}


class FastPitch1_1(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(FastPitch1_1, self).__init__()

        self.logger = logger
        self.PROD = PROD
        self.models_manager = models_manager
        self.device = device
        self.ckpt_path = None

        self.arpabet_dict = {}

        torch.backends.cudnn.benchmark = True

        self.init_model("english_basic")
        self.isReady = True


    def init_model (self, symbols_alphabet):

        parser = argparse.ArgumentParser(description='PyTorch FastPitch Inference', allow_abbrev=False)
        self.symbols_alphabet = symbols_alphabet

        model_parser = models.parse_model_args("FastPitch", symbols_alphabet, parser, add_help=False)
        model_args, model_unk_args = model_parser.parse_known_args()
        model_config = models.get_model_config("FastPitch", model_args)

        self.model = models.get_model("FastPitch", model_config, self.device, self.logger, forward_is_infer=True, jitable=False)
        self.model.eval()
        self.model.device = self.device

    def load_state_dict (self, ckpt_path, ckpt, n_speakers=1, base_lang=None):

        self.ckpt_path = ckpt_path

        with open(ckpt_path.replace(".pt", ".json"), "r") as f:
            data = json.load(f)
            if "symbols_alphabet" in data.keys() and data["symbols_alphabet"]!=self.symbols_alphabet:
                self.logger.info(f'Changing symbols_alphabet from {self.symbols_alphabet} to {data["symbols_alphabet"]}')
                self.init_model(data["symbols_alphabet"])

        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']

        self.model.load_state_dict(ckpt, strict=False)
        self.model = self.model.float()
        self.model.eval()


    def init_arpabet_dicts (self):
        if len(list(self.arpabet_dict.keys()))==0:
            self.refresh_arpabet_dicts()

    def refresh_arpabet_dicts (self):
        self.arpabet_dict = {}
        json_files = sorted(os.listdir(f'{"./resources/app" if self.PROD else "."}/arpabet'))
        json_files = [fname for fname in json_files if fname.endswith(".json")]

        for fname in json_files:
            with codecs.open(f'{"./resources/app" if self.PROD else "."}/arpabet/{fname}', encoding="utf-8") as f:
                json_data = json.load(f)

                for word in list(json_data["data"].keys()):
                    if json_data["data"][word]["enabled"]==True:
                        self.arpabet_dict[word] = json_data["data"][word]["arpabet"]


    def infer_arpabet_dict (self, sentence, plugin_manager=None):
        dict_words = list(self.arpabet_dict.keys())

        data_context = {}
        data_context["sentence"] = sentence
        data_context["dict_words"] = dict_words
        data_context["language"] = "en"
        plugin_manager.run_plugins(plist=plugin_manager.plugins["arpabet-replace"]["pre"], event="pre arpabet-replace", data=data_context)
        sentence = data_context["sentence"]
        dict_words = data_context["dict_words"]

        # Don't run the ARPAbet replacement for every single word, as it would be too slow. Instead, do it only for words that are actually present in the prompt
        words_in_prompt = (sentence+" ").replace("}","").replace("{","").replace(",","").replace("?","").replace("!","").replace("...",".").replace(". "," ").lower().split(" ")
        words_in_prompt = [word.strip() for word in words_in_prompt if len(word.strip()) and word in dict_words]

        if len(words_in_prompt):

            # Pad out punctuation, to make sure they don't get used in the word look-ups
            sentence = " "+sentence.replace(",", " ,").replace(".", " .").replace("!", " !").replace("?", " ?")+" "

            for dict_word in words_in_prompt:


                arpabet_string = " "+self.arpabet_dict[dict_word]+" "
                if "CX" in arpabet_string:
                    # German:
                    # hhhhh sound After "a", "o", "u" and "au"
                    # The usual K HH otherwise
                    # Need to account for multiple ch per word
                    pass
                for key in ARPAbet_replacements_dict.keys():
                    arpabet_string = arpabet_string.replace(f' {key} ', f' {ARPAbet_replacements_dict[key]} ')


                arpabet_string = arpabet_string.strip()

                sentence = re.sub("(?<!\{)\s"+dict_word.strip().replace(".", "\.").replace("(", "\(").replace(")", "\)")+"\s(?![\w\s\(\)]*[\}])", " {"+arpabet_string+"} ", sentence, flags=re.IGNORECASE)
                # Do it twice, because re will not re-use spaces, so if you have two neighbouring words to be replaced,
                # and they share a space character, one of them won't get changed
                sentence = re.sub("(?<!\{)\s"+dict_word.strip().replace(".", "\.").replace("(", "\(").replace(")", "\)")+"\s(?![\w\s\(\)]*[\}])", " {"+arpabet_string+"} ", sentence, flags=re.IGNORECASE)

            # Undo the punctuation padding, to retain the original sentence structure
            sentence = sentence.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
            sentence = re.sub("^\s+", " ", sentence) if sentence.startswith("  ") else re.sub("^\s*", "", sentence)
            sentence = re.sub("\s+$", " ", sentence) if sentence.endswith("  ") else re.sub("\s*$", "", sentence)

        data_context = {}
        data_context["sentence"] = sentence
        plugin_manager.run_plugins(plist=plugin_manager.plugins["arpabet-replace"]["post"], event="post arpabet-replace", data=data_context)
        sentence = data_context["sentence"]
        return sentence



    def infer_batch(self, plugin_manager, linesBatch, outputJSON, vocoder, speaker_i, old_sequence=None, useSR=False, useCleanup=False):
        print(f'Inferring batch of {len(linesBatch)} lines')

        sigma_infer = 0.9
        stft_hop_length = 256
        sampling_rate = 22050
        denoising_strength = 0.01

        text_sequences = []
        cleaned_text_sequences = []
        for record in linesBatch:
            text = record[0]
            text = re.sub(r'[^a-zA-ZäöüÄÖÜß_\s\(\)\[\]0-9\?\.\,\!\'\{\}]+', '', text)
            text = self.infer_arpabet_dict(text, plugin_manager)
            text = text.replace("(", "").replace(")", "")
            text = re.sub(r'[^a-zA-ZäöüÄÖÜß\s\(\)\[\]0-9\?\.\,\!\'\{\}]+', '', text)
            sequence = text_to_sequence(text, "english_basic", ['english_cleaners'])
            cleaned_text_sequences.append(sequence_to_text("english_basic", sequence))
            text = torch.LongTensor(sequence)
            text_sequences.append(text)

        text_sequences = pad_sequence(text_sequences, batch_first=True).to(self.device)

        with torch.no_grad():
            pace = torch.tensor([record[3] for record in linesBatch]).unsqueeze(1).to(self.device)
            pitch_amp = torch.tensor([record[7] for record in linesBatch]).unsqueeze(1).to(self.device)
            pitch_data = None # Maybe in the future
            mel, mel_lens, dur_pred, pitch_pred, energy_pred, start_index, end_index = self.model.infer_advanced(self.logger, plugin_manager, cleaned_text_sequences, text_sequences, speaker_i=speaker_i, pace=pace, pitch_data=pitch_data, old_sequence=None, pitch_amp=pitch_amp)

            if "waveglow" in vocoder:

                self.models_manager.init_model(vocoder)
                audios = self.models_manager.models(vocoder).model.infer(mel, sigma=sigma_infer)
                audios = self.models_manager.models(vocoder).denoiser(audios.float(), strength=denoising_strength).squeeze(1)

                for i, audio in enumerate(audios):
                    audio = audio[:mel_lens[i].item() * stft_hop_length]
                    audio = audio/torch.max(torch.abs(audio))
                    output = linesBatch[i][4]
                    audio = audio.cpu().numpy()
                    if useCleanup:
                        ffmpeg_path = f'{"./resources/app" if self.PROD else "."}/python/ffmpeg.exe'

                        if useSR:
                            write(output.replace(".wav", "_preSR.wav"), sampling_rate, audio)
                        else:
                            write(output.replace(".wav", "_preCleanupPreFFmpeg.wav"), sampling_rate, audio)
                            stream = ffmpeg.input(output.replace(".wav", "_preCleanupPreFFmpeg.wav"))
                            ffmpeg_options = {"ar": 48000}
                            output_path = output.replace(".wav", "_preCleanup.wav")
                            stream = ffmpeg.output(stream, output_path, **ffmpeg_options)
                            out, err = (ffmpeg.run(stream, cmd=ffmpeg_path, capture_stdout=True, capture_stderr=True, overwrite_output=True))
                            os.remove(output.replace(".wav", "_preCleanupPreFFmpeg.wav"))
                    else:
                        write(output.replace(".wav", "_preSR.wav") if useSR else output, sampling_rate, audio)

                    if useSR:
                        self.models_manager.init_model("nuwave2")
                        self.models_manager.models("nuwave2").sr_audio(output.replace(".wav", "_preSR.wav"), output.replace(".wav", "_preCleanup.wav") if useCleanup else output)
                        os.remove(output.replace(".wav", "_preSR.wav"))

                    if useCleanup:
                        self.models_manager.init_model("deepfilternet2")
                        self.models_manager.models("deepfilternet2").cleanup_audio(output.replace(".wav", "_preCleanup.wav"), output)
                        os.remove(output.replace(".wav", "_preCleanup.wav"))
                del audios
            else:
                self.models_manager.load_model("hifigan", f'{"./resources/app" if self.PROD else "."}/python/hifigan/hifi.pt' if vocoder=="qnd" else self.ckpt_path.replace(".pt", ".hg.pt"))

                y_g_hat = self.models_manager.models("hifigan").model(mel)
                audios = y_g_hat.view((y_g_hat.shape[0], y_g_hat.shape[2]))
                # audio = audio * 2.3026  # This brings it to the same volume, but makes it clip in places
                for i, audio in enumerate(audios):
                    audio = audio[:mel_lens[i].item() * stft_hop_length]
                    audio = audio.cpu().numpy()
                    audio = audio * 32768.0
                    audio = audio.astype('int16')
                    output = linesBatch[i][4]

                    if useCleanup:
                        ffmpeg_path = f'{"./resources/app" if self.PROD else "."}/python/ffmpeg.exe'

                        if useSR:
                            write(output.replace(".wav", "_preSR.wav"), sampling_rate, audio)
                        else:
                            write(output.replace(".wav", "_preCleanupPreFFmpeg.wav"), sampling_rate, audio)
                            stream = ffmpeg.input(output.replace(".wav", "_preCleanupPreFFmpeg.wav"))
                            ffmpeg_options = {"ar": 48000}
                            output_path = output.replace(".wav", "_preCleanup.wav")
                            stream = ffmpeg.output(stream, output_path, **ffmpeg_options)
                            out, err = (ffmpeg.run(stream, cmd=ffmpeg_path, capture_stdout=True, capture_stderr=True, overwrite_output=True))
                            os.remove(output.replace(".wav", "_preCleanupPreFFmpeg.wav"))
                    else:
                        write(output.replace(".wav", "_preSR.wav") if useSR else output, sampling_rate, audio)

                    if useSR:
                        self.models_manager.init_model("nuwave2")
                        self.models_manager.models("nuwave2").sr_audio(output.replace(".wav", "_preSR.wav"), output.replace(".wav", "_preCleanup.wav") if useCleanup else output)
                        os.remove(output.replace(".wav", "_preSR.wav"))

                    if useCleanup:
                        self.models_manager.init_model("deepfilternet2")
                        self.models_manager.models("deepfilternet2").cleanup_audio(output.replace(".wav", "_preCleanup.wav"), output)
                        os.remove(output.replace(".wav", "_preCleanup.wav"))


            if outputJSON:
                for ri, record in enumerate(linesBatch):
                    # linesBatch: sequence, pitch, duration, pace, tempFileLocation, outPath, outFolder
                    output_fname = linesBatch[ri][5].replace(".wav", ".json")

                    containing_folder = "/".join(output_fname.split("/")[:-1])
                    os.makedirs(containing_folder, exist_ok=True)

                    with open(output_fname, "w+") as f:
                        data = {}
                        data["inputSequence"] = str(linesBatch[ri][0])
                        data["pacing"] = float(linesBatch[ri][3])
                        data["letters"] = [char.replace("{", "").replace("}", "") for char in list(cleaned_text_sequences[ri].split("|"))]
                        data["currentVoice"] = self.ckpt_path.split("/")[-1].replace(".pt", "")
                        data["resetEnergy"] = [float(val) for val in list(energy_pred[ri].cpu().detach().numpy())]
                        data["resetPitch"] = [float(val) for val in list(pitch_pred[ri][0].cpu().detach().numpy())]
                        data["resetDurs"] = [float(val) for val in list(dur_pred[ri].cpu().detach().numpy())]
                        data["ampFlatCounter"] = 0
                        data["pitchNew"] = data["resetPitch"]
                        data["energyNew"] = data["resetEnergy"]
                        data["dursNew"] = data["resetDurs"]

                        f.write(json.dumps(data, indent=4))


            del mel, mel_lens

        return ""

    def infer(self, plugin_manager, text, output, vocoder, speaker_i, pace=1.0, editor_data=None, old_sequence=None, globalAmplitudeModifier=None, base_lang=None, base_emb=None, useSR=False, useCleanup=False):

        sigma_infer = 0.9
        stft_hop_length = 256
        sampling_rate = 22050
        denoising_strength = 0.01

        text = re.sub(r'[^a-zA-ZäöüÄÖÜß_\s\(\)\[\]0-9\?\.\,\!\'\{\}\-]+', '', text)
        text = self.infer_arpabet_dict(text, plugin_manager)
        text = text.replace("(", "").replace(")", "")
        text = re.sub(r'[^a-zA-ZäöüÄÖÜß\s\(\)\[\]0-9\?\.\,\!\'\{\}]+', '', text)
        sequence = text_to_sequence(text, "english_basic", ['english_cleaners'])
        cleaned_text = sequence_to_text("english_basic", sequence)
        text = torch.LongTensor(sequence)
        text = pad_sequence([text], batch_first=True).to(self.models_manager.device)

        with torch.no_grad():

            if old_sequence is not None:
                old_sequence = re.sub(r'[^a-zA-Z\s\(\)\[\]0-9\?\.\,\!\'\{\}]+', '', old_sequence)
                old_sequence = text_to_sequence(old_sequence, "english_basic", ['english_cleaners'])
                old_sequence = torch.LongTensor(old_sequence)
                old_sequence = pad_sequence([old_sequence], batch_first=True).to(self.models_manager.device)

            mel, mel_lens, dur_pred, pitch_pred, energy_pred, start_index, end_index = self.model.infer_advanced(self.logger, plugin_manager, [cleaned_text], text, speaker_i=speaker_i, pace=pace, pitch_data=editor_data, old_sequence=old_sequence)

            if "waveglow" in vocoder:

                self.models_manager.init_model(vocoder)
                audios = self.models_manager.models(vocoder).model.infer(mel, sigma=sigma_infer)
                audios = self.models_manager.models(vocoder).denoiser(audios.float(), strength=denoising_strength).squeeze(1)

                for i, audio in enumerate(audios):
                    audio = audio[:mel_lens[i].item() * stft_hop_length]
                    audio = audio/torch.max(torch.abs(audio))
                    write(output, sampling_rate, audio.cpu().numpy())
                del audios
            else:
                self.models_manager.load_model("hifigan", f'{"./resources/app" if self.PROD else "."}/python/hifigan/hifi.pt' if vocoder=="qnd" else self.ckpt_path.replace(".pt", ".hg.pt"))

                y_g_hat = self.models_manager.models("hifigan").model(mel)
                audio = y_g_hat.squeeze()
                audio = audio * 32768.0
                # audio = audio * 2.3026  # This brings it to the same volume, but makes it clip in places
                audio = audio.cpu().numpy().astype('int16')

                if useCleanup:
                    ffmpeg_path = f'{"./resources/app" if self.PROD else "."}/python/ffmpeg.exe'

                    if useSR:
                        write(output.replace(".wav", "_preSR.wav"), sampling_rate, audio)
                    else:
                        write(output.replace(".wav", "_preCleanupPreFFmpeg.wav"), sampling_rate, audio)
                        stream = ffmpeg.input(output.replace(".wav", "_preCleanupPreFFmpeg.wav"))
                        ffmpeg_options = {"ar": 48000}
                        output_path = output.replace(".wav", "_preCleanup.wav")
                        stream = ffmpeg.output(stream, output_path, **ffmpeg_options)
                        out, err = (ffmpeg.run(stream, cmd=ffmpeg_path, capture_stdout=True, capture_stderr=True, overwrite_output=True))
                        os.remove(output.replace(".wav", "_preCleanupPreFFmpeg.wav"))

                else:
                    write(output.replace(".wav", "_preSR.wav") if useSR else output, sampling_rate, audio)

                if useSR:
                    self.models_manager.init_model("nuwave2")
                    self.models_manager.models("nuwave2").sr_audio(output.replace(".wav", "_preSR.wav"), output.replace(".wav", "_preCleanup.wav") if useCleanup else output)

                if useCleanup:
                    self.models_manager.init_model("deepfilternet2")
                    self.models_manager.models("deepfilternet2").cleanup_audio(output.replace(".wav", "_preCleanup.wav"), output)

                del audio

            del mel, mel_lens

        [pitch, durations, energy] = [pitch_pred.squeeze().cpu().detach().numpy(), dur_pred.cpu().detach().numpy()[0], energy_pred.cpu().detach().numpy()[0] if energy_pred is not None else []]

        [em_angry, em_happy, em_sad, em_surprise] = [[],[],[],[]]
        pitch_durations_energy_text = ",".join([str(v) for v in pitch]) + "\n" + \
                                      ",".join([str(v) for v in durations]) + "\n" + \
                                      ",".join([str(v) for v in energy]) + "\n" + \
                                      ",".join([str(v) for v in em_angry]) + "\n" + \
                                      ",".join([str(v) for v in em_happy]) + "\n" + \
                                      ",".join([str(v) for v in em_sad]) + "\n" + \
                                      ",".join([str(v) for v in em_surprise]) + "\n" + "{"+"}"

        del pitch_pred, dur_pred, energy_pred, text, sequence
        return pitch_durations_energy_text +"\n"+cleaned_text +"\n"+ f'{start_index}\n{end_index}'

    def set_device (self, device):
        self.device = device
        self.model = self.model.to(device)
        self.model.device = device


    def run_speech_to_speech (self, audiopath, models_manager, plugin_manager, modelType, s2s_components, text):
        return self.model.run_speech_to_speech(self.device, self.logger, models_manager, plugin_manager, modelType, s2s_components, audiopath, text, text_to_sequence, sequence_to_text, self)


