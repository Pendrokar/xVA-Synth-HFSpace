import os
import sys
import traceback
import multiprocessing
import json

torch_dml_device = None

multiprocessing.freeze_support()

# PROD = 'xVASynth.exe' in os.listdir(".")
PROD = True
sys.path.append("./resources/app")
sys.path.append("./resources/app/python")
sys.path.append("./resources/app/deepmoji_plugin")

# Saves me having to do backend re-compilations for every little UI hotfix
with open(f'{"./resources/app" if PROD else "."}/javascript/script.js', encoding="utf8") as f:
    lines = f.read().split("\n")
    APP_VERSION = lines[1].split('"v')[1].split('"')[0]

# Imports and logger setup
# ========================
try:
    # import python.pyinstaller_imports
    import numpy

    import logging
    from logging.handlers import RotatingFileHandler
    import json
    from socketserver     import ThreadingMixIn
    from python.audio_post import run_audio_post, prepare_input_audio, mp_ffmpeg_output, normalize_audio, start_microphone_recording, move_recorded_file
    import ffmpeg
except:
    print(traceback.format_exc())
    with open("./DEBUG_err_imports.txt", "w+") as f:
        f.write(traceback.format_exc())

# Pyinstaller hack
# ================
try:
    def script_method(fn, _rcb=None):
        return fn
    def script(obj, optimize=True, _frames_up=0, _rcb=None):
        return obj
    import torch.jit
    torch.jit.script_method = script_method
    torch.jit.script = script
    import torch
    import tqdm
    import regex
except:
    print(traceback.format_exc())
    with open("./DEBUG_err_import_torch.txt", "w+") as f:
        f.write(traceback.format_exc())
# ================
# CPU_ONLY = not torch.cuda.is_available()
CPU_ONLY = True

try:
    logger = logging.getLogger('serverLog')
    logger.setLevel(logging.DEBUG)
    server_log_path = f'{os.path.dirname(os.path.realpath(__file__))}/{"../../../" if PROD else ""}/server.log'
    fh = RotatingFileHandler(server_log_path, maxBytes=2*1024*1024, backupCount=5)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f'New session. Version: {APP_VERSION}. Installation: {"CPU" if CPU_ONLY else "CPU+GPU"} | Prod: {PROD} | Log path: {server_log_path}')

    logger.orig_info = logger.info

    def prefixed_log (msg):
        logger.info(f'{logger.logging_prefix}{msg}')


    def set_logger_prefix (prefix=""):
        if len(prefix):
            logger.logging_prefix = f'[{prefix}]: '
            logger.log = prefixed_log
        else:
            logger.log = logger.orig_info

    logger.set_logger_prefix = set_logger_prefix
    logger.set_logger_prefix("")

except:
    with open("./DEBUG_err_logger.txt", "w+") as f:
        f.write(traceback.format_exc())
    try:
        logger.info(traceback.format_exc())
    except:
        pass

if CPU_ONLY:
    torch_dml_device = torch.device("cpu")


try:
    from python.plugins_manager import PluginManager
    plugin_manager = PluginManager(APP_VERSION, PROD, CPU_ONLY, logger)
    active_plugins = plugin_manager.get_active_plugins_count()
    logger.info(f'Plugin manager loaded. {active_plugins} active plugins.')
except:
    logger.info("Plugin manager FAILED.")
    logger.info(traceback.format_exc())

plugin_manager.run_plugins(plist=plugin_manager.plugins["start"]["pre"], event="pre start", data=None)


# ======================== Models manager
modelsPaths = {}
try:
    from python.models_manager import ModelsManager
    models_manager = ModelsManager(logger, PROD, device="cpu")
except:
    logger.info("Models manager failed to initialize")
    logger.info(traceback.format_exc())
# ========================



print("Models ready")
logger.info("Models ready")


post_data = ""
def loadModel(post_data):
    logger.info("Direct: loadModel")
    logger.info(post_data)
    ckpt = post_data["model"]
    modelType = post_data["modelType"]
    instance_index = post_data["instance_index"] if "instance_index" in post_data else 0
    modelType = modelType.lower().replace(".", "_").replace(" ", "")
    post_data["pluginsContext"] = json.loads(post_data["pluginsContext"])
    n_speakers = post_data["model_speakers"] if "model_speakers" in post_data else None
    base_lang = post_data["base_lang"] if "base_lang" in post_data else None


    plugin_manager.run_plugins(plist=plugin_manager.plugins["load-model"]["pre"], event="pre load-model", data=post_data)
    models_manager.load_model(modelType, ckpt+".pt", instance_index=instance_index, n_speakers=n_speakers, base_lang=base_lang)
    plugin_manager.run_plugins(plist=plugin_manager.plugins["load-model"]["post"], event="post load-model", data=post_data)

    if modelType=="fastpitch1_1":
        models_manager.models_bank["fastpitch1_1"][instance_index].init_arpabet_dicts()

    return req_response

def synthesize(post_data):
    logger.info("Direct: synthesize")
    post_data["pluginsContext"] = json.loads(post_data["pluginsContext"])
    instance_index = post_data["instance_index"] if "instance_index" in post_data else 0


    # Handle the case where the vocoder remains selected on app start-up, with auto-HiFi turned off, but no setVocoder call is made before synth
    continue_synth = True
    if "waveglow" in post_data["vocoder"]:
        waveglowPath = post_data["waveglowPath"]
        req_response = models_manager.load_model(post_data["vocoder"], waveglowPath, instance_index=instance_index)
        if req_response=="ENOENT":
            continue_synth = False

    device = post_data["device"] if "device" in post_data else models_manager.device_label
    device = torch.device("cpu") if device=="cpu" else (torch_dml_device if CPU_ONLY else torch.device("cuda:0"))
    models_manager.set_device(device, instance_index=instance_index)

    if continue_synth:
        plugin_manager.set_context(post_data["pluginsContext"])
        plugin_manager.run_plugins(plist=plugin_manager.plugins["synth-line"]["pre"], event="pre synth-line", data=post_data)

        modelType = post_data["modelType"]
        text = post_data["sequence"]
        pace = float(post_data["pace"])
        out_path = post_data["outfile"]
        base_lang = post_data["base_lang"] if "base_lang" in post_data else None
        base_emb = post_data["base_emb"] if "base_emb" in post_data else None
        pitch = post_data["pitch"] if "pitch" in post_data else None
        energy = post_data["energy"] if "energy" in post_data else None
        emAngry = post_data["emAngry"] if "emAngry" in post_data else None
        emHappy = post_data["emHappy"] if "emHappy" in post_data else None
        emSad = post_data["emSad"] if "emSad" in post_data else None
        emSurprise = post_data["emSurprise"] if "emSurprise" in post_data else None
        editorStyles = post_data["editorStyles"] if "editorStyles" in post_data else None
        duration = post_data["duration"] if "duration" in post_data else None
        speaker_i = post_data["speaker_i"] if "speaker_i" in post_data else None
        useSR = post_data["useSR"] if "useSR" in post_data else None
        useCleanup = post_data["useCleanup"] if "useCleanup" in post_data else None
        vocoder = post_data["vocoder"]
        globalAmplitudeModifier = float(post_data["globalAmplitudeModifier"]) if "globalAmplitudeModifier" in post_data else None
        editor_data = [pitch, duration, energy, emAngry, emHappy, emSad, emSurprise, editorStyles]
        old_sequence = post_data["old_sequence"] if "old_sequence" in post_data else None

        model = models_manager.models(modelType.lower().replace(".", "_").replace(" ", ""), instance_index=instance_index)
        req_response = model.infer(plugin_manager, text, out_path, vocoder=vocoder, \
            speaker_i=speaker_i, editor_data=editor_data, pace=pace, old_sequence=old_sequence, \
            globalAmplitudeModifier=globalAmplitudeModifier, base_lang=base_lang, base_emb=base_emb, useSR=useSR, useCleanup=useCleanup)

        plugin_manager.run_plugins(plist=plugin_manager.plugins["synth-line"]["post"], event="post synth-line", data=post_data)

    return req_response