import os
import sys
import time
import requests
from subprocess import Popen, PIPE
import threading
# from huggingface_hub import hf_hub_download
import gradio as gr

def run_xvaserver():
	# try:
	# start the process without waiting for a response
	print('Running xVAServer subprocess...\n')
	xvaserver = Popen(['python', f'{os.path.dirname(os.path.abspath(__file__))}/resources/app/server.py'], stdout=PIPE, stderr=PIPE, cwd=f'{os.path.dirname(os.path.abspath(__file__))}/resources/app/')
	# except:
	# 	print('Could not run xVASynth.')
	# 	sys.exit(0)

	# Wait for a moment to ensure the server starts up
	time.sleep(10)

	# Check if the server is running
	if xvaserver.poll() is not None:
		print("Web server failed to start.")
		sys.exit(0)

	# contact local xVASynth server; ~2 second timeout
	print('Attempting to connect to xVASynth...')
	try:
		response = requests.get('http://0.0.0.0:8008')
		response.raise_for_status()  # If the response contains an HTTP error status code, raise an exception
	except requests.exceptions.RequestException as err:
		print('Failed to connect!')
		return

	print('xVAServer running on port 8008')

	# load default voice model
	load_model()

	# Wait for the process to exit
	xvaserver.wait()

def load_model():
	model_name = "Pendrokar/TorchMoji"
	# model_path = hf_hub_download(repo_id=model_name, filename="ccby_nvidia_hifi_6670_M.pt")
	# model_json_path = hf_hub_download(repo_id=model_name, filename="ccby_nvidia_hifi_6670_M.json")
	model_path = '/tmp/hfcache/models--Pendrokar--xvapitch_nvidia_6670/snapshots/2e138a7c459fb1cb1182dd7bc66813f5325d30fd/ccby_nvidia_hifi_6670_M.pt'
	model_json_path = '/tmp/hfcache/models--Pendrokar--xvapitch_nvidia_6670/snapshots/2e138a7c459fb1cb1182dd7bc66813f5325d30fd/ccby_nvidia_hifi_6670_M.json'
	# try:
	# 	os.symlink(model_path, os.path.join('./models/ccby/', os.path.basename(model_path)))
	# 	os.symlink(model_json_path, os.path.join('./models/ccby/', os.path.basename(model_json_path)))
	# except:
	# 	print('Failed creating symlinks, they probably already exist')

	model_type = 'xVAPitch'
	language = 'en'

	data = {
		'outputs': None,
		'version': '3.0',
		'model': model_path.replace('.pt', ''),
		'modelType': model_type,
		'base_lang': language,
		'pluginsContext': '{}',
	}

	try:
		response = requests.post('http://0.0.0.0:8008/loadModel', json=data)
		response.raise_for_status()  # If the response contains an HTTP error status code, raise an exception
	except requests.exceptions.RequestException as err:
		print('Failed to load voice model!')
	return

def predict(input, pacing):

	# reload model just in case
	load_model()

	model_type = 'xVAPitch'
	line = 'Test'
	pace = pacing if pacing else 1.0
	save_path = '/tmp/xvapitch_audio_sample.wav'
	language = 'en'
	base_speaker_emb = "0.02059412143946074,0.008201584451707952,-0.05277030320936666,-0.039501296572360664,0.03819673617993138,0.040965055419247845,0.023416176201646512,0.0738048970470502,-0.022488680839208613,0.004742570652413852,0.08258309825838091,-0.029896974180596347,0.007623321898076605,-0.014385980884378709,-0.04895588345028929,-0.03173952258108633,0.06192960956524358,0.0024332545144194343,0.03269641523992302,0.025787167165119314,0.029786330252299206,0.06342059792605999,0.025040897510403375,-0.033921410832297355,-0.03109066728659562,0.005312140046451575,0.005451900488461911,0.04885825827158006,0.04651092324831847,0.04154541535128829,0.015740028257454398,0.07411105939487676,-0.04705822052194547,0.08234144021402753,-0.007609069202950745,-0.04202066055572556,-0.015052768978905475,0.018694231043643515,0.029143727266584075,0.10079553395035755,-0.013802278590248252,0.007599227800692346,-0.02139346933674926,-0.045930129824978,-0.002763870069305383,-0.02819758760639864,0.033784053053219305,-0.008986875977842606,-0.040692288661311804,-0.01590705578861926,0.048330252381842985,0.05027618679116613,-0.002831847005020155,-0.024023675035413982,-0.02443027882665902,0.004039959747256932,0.07551178626291195,0.029589468027455566,-0.0015828541818794495,0.04676929206444622,-0.020613106888042514,0.042098865959990336,0.09546128130911928,0.020703315059659565,-0.011249538079616667,-0.07540832834600718,0.03465820367046974,0.00393294581254368,0.053209329078733186,-0.03021388709092682,-0.01951826801479834,-0.05977936920461946,0.008797915463121605,-0.05712115565482911,0.04131273251034018,-0.09273814899968984,-0.10207227659178059,0.058790477853312324,0.01897804391189814,-0.02178261259176716,-0.021484194506276158,-0.08229631100545559,-0.05138027446698438,-0.00027628880637762317,0.015647992154012472,0.005354931579772256,0.08213169780215862,-0.047810609907234625,-0.05414263731160324,-0.038296023264480346,-0.007579790276616037,-0.03646887578494079,-0.04008997937172555,0.03678530506896434,-0.09368463678472914,-0.05199597782090462,-0.036921057160887616,0.029692581331897193,0.016386972778861173,0.06202756289126103,0.05066110052307285,0.027814329759178042,-0.007281413355143866,-0.03759511055890598,0.08385584525884529,-0.017281195804149618,0.06913154615865104,0.01315304627673114,-0.017020608851213877,0.06319093348804176,-0.03986924834602306,0.09147952597619971,-0.024315902884245753,-0.06426214898980127,0.03070522487696974,0.007921454523906421,-0.002303376761462218,0.04705365155222134,0.024035771629399676,0.005804419645614805,0.05944420597029526,0.041190475921322904,0.05472544337333494,0.0681096299295807,0.003283463596832385,0.012816178237446085,0.026948904272056224,0.006394830216742351,-0.05120212424670069,-0.039777774021692414,0.05101289164503741,-0.018439937818314864,0.011897065237903383,-0.09698980388851842,0.0047267140457017045,0.022849590908084076,-0.015288020578465004,0.03749845410723472,0.02220631416812867,-0.02319304866090999,0.04008733320760122,0.030397735504018062,-0.008541387401820474,-0.03605930499783825,0.06196725351288037,-0.11121911417017709,-0.04725924740469605,-0.07899495961733442,-0.04244266485396749,-0.027248625321533895,0.05066213618646282,-0.03506564923310191,-0.005840301218127498,0.004162213410093819,0.04909547090209646,0.06071786370157997,-0.05474646558435032,0.07962362481521716,0.07091592467528665,0.05937644418646688,-0.02911951892934846,-0.03254295913596573,-0.022591876345155616,0.0008537834699139989,-0.014769368427959144,0.009630182049977329,0.009811974880308653,-0.0542164803580909,-0.017230848620820638,0.024878364927052834,-0.02575894071886462,-0.043100961132885696,0.04117734547005268,0.03140134792124991,0.08679875127270445,-0.02462917350431534,-0.003768949563922518,0.03494913383390341,0.02756011486463656,0.018993482183639197,0.12100332289166019,0.05367585477269294,-0.07146326847903536,-0.020866162142183346,-0.09688478454024067,0.05966068745682935,-0.06886393149376473,-0.011604896745240395,0.02478997662226238,-0.01309686984815536,0.07455265846421104,-0.014204533134263597,0.012980711683124951,-0.01195490683579427,-0.011568376921449849,0.07520772829704946,0.06424759474690483,-0.05513639092416767,0.04627532131785045,0.05297611672982825,0.005834491664910714,-0.030645055896297464,0.028803855721750933,0.026741277795573754,-0.017403122514379046,0.0623772399556704,-0.045581046657594285,-0.03186385889481072,-0.07502207317506247,-0.02076766956408014,0.054792314306389735,-0.006492469279425618,0.07371799293001495,-0.053286731833541254,0.03164880561816261,-0.03494070488630127,-0.03544574287747602,0.045445782147385726,0.026391253902760294,0.000211523792895327,-0.05624092817469954,0.019716588188751345,-0.06794026260110232,0.03469104941185243,0.06745962027694244,-0.03902418382432907,0.047196376212513304,0.01882557501922099,0.014702989353749826,0.08243897184831722,-0.05324926561095518,-0.03501195345812069,-1.3509955095127642e-05,-0.0015714049399960489,-0.04396534584077557,-0.01538779513563676,0.01766822271506532,0.040906586077407575,-0.03636189620324979,0.0666709730459348,-0.04657088150519267,0.001146672358887494,-0.004549922328779031,-0.0029069946638099293,0.08609365201078932,0.007310233866964863,-0.034266973294036544,0.04385545068939631,-0.06263538073592065,0.045894727005779304,-0.03535625820431461,0.054109224776883194,-0.02543766851001126,-0.03218286823689956,-0.012191509595846955,-0.026217901767453682,-0.003597006255589585,0.02894194997147505,0.01859091536427267,-0.023020929704909758,0.0038375679353477394,0.012759599045732924,0.02959672438355217,-0.04321566263022482,-0.07320415645159528,0.00819928819205296,0.0015323787074705887,-0.0010988803019577774,0.00690822757937127,-0.043951312960657914,-0.013250335394692688,-0.0024827032439735607,-0.024593628635469066,-0.07850441735821104,0.055323840016542794,0.04537621203964216,0.060634791490976565,0.004189244884199575,-0.06038241476854265,0.08238880406663715,-0.06445529728296406,0.032054746393533805,-0.057435940374490804,0.009853677915727594,-0.02029606229240358,-0.00019822253276164476,0.01529772139350525,-0.029564093325759,0.00810868760322473,0.028966266215952125,-0.07600746410335776,-0.006572367772158361,0.03754367405100014,0.03652616232684152,-0.03246471418340199,-0.004224826539332405,0.019481275445988145,0.006692394446589274,-0.08669556196843381,0.013616501076132829,0.0573685407671237,0.05089444776934148,0.02874064510977234,-0.03025669870904372,0.030267210507894167,-0.008183882297229418,0.043102053927437906,-0.03849197151167474,-0.026005145798754858,-0.006368485325804052,0.049011632296519024,0.037056268237165776,-0.01083617499305662,0.04335865035300966,-0.028769596020093165,-0.020898109163372286,0.06369254005853772,0.04098281680003892,-0.020376124538663866,0.04460217810394495,0.014145542171373753,0.00898061909724459,-0.01882876080430672,0.0699170900461264,0.00964391625568027,-0.02712623176060162,0.08191445457087929,0.0268364830395993,0.014825367637737286,-0.025214233127199278,0.026323141640434686,-0.018202527489970047,-0.019837385731968606,0.0479153274012764,0.011188967192150998,0.036603557947021574,-0.05920480367451016,-0.014246720893377059,0.03899231501551069,0.0093478324842559,0.0034190027629483095,-0.0655321143067043,-0.030179479636663353,0.04572833065977283,0.04754136161317401,-0.01639615125308229,-0.052265391394716566,0.026052826282500826,-0.06788857197141301,-0.0868158467238591,-0.09705289755826585,-0.03434318643462554,-0.03669981382775931,-0.027212280322523666,-0.0061442705779012905,-0.08261614430145199,0.028539358839567926,0.01666725842837058,-0.04597714248138618,0.07453649286471851,0.0310573898008201,-0.01680015605486269,-0.01059527520976918,-0.03854385718865193,-0.00432396184739404,-0.015657636695496673,0.04046198712689444,-0.03080469386353861,0.04391195061204182,-0.10213319722523191,0.004702051133095527,0.005826663656902152,0.01754408309175418,-0.00024831212068023034,0.006101960708786389,0.006534040025226202,0.033356185016412083,0.024607735794179922,-0.027833515235262212,0.06332068529755308,0.06072193482512277,0.016997643385796186,-0.017216675994987198,-0.01516786242662637,-0.0497693492123646,0.05368942615234157,0.02393329374415363,0.06444157393557816,0.08351245357532168,-0.01607344106907917,0.04092942101051993,-0.020996099156013536,0.01321767909795303,0.041978731197264404,0.0014093538466343983,0.060341434805568,0.006787355684922708,0.016447466090818014,0.03921590758894128,0.05337420085504805,0.059527523387126455,0.039020486363086006,-0.028823242409137972,-0.055728267576450594,0.00027308253411236763,0.007491355826807867,0.08325049031269463,-0.02770126101575972,0.023306320116218422,-0.01794592313548733,0.0402008194167628,0.023652274357688925,0.05683651955403472,0.042481214178309475,0.026810950836080788,-0.017524332176538086,0.02498395545218121,-0.08779296280792183,0.0004703617308028609,-0.04403565196404175,0.01911810095896643,0.023664926168405282,0.01235814993967046,0.045064818723423127,0.022357160109669854,-0.015132149144046512,0.01410923882672814,0.0011764588005077014,0.0641110797887931,-0.05035627528900969,-0.014802792891870656,-0.025766586370680213,-0.020373629369100272,-0.018469269514097858,-0.03596074791150914,0.03992088692162406,-0.009879509547227565,0.020604571845663673,0.07631463733684833,-0.055150283365207706,-0.0026970588135657907,-0.09605985120437321,0.03111473269531879,0.0498327896477001,0.0388820725625065,-0.020225335602984486,0.03416608369547628,-0.009092833227865778,0.025203313729289144,-0.009401636411159387,-0.0303617648662941,-0.04304890119472444,0.014895982312722093,-0.010428269058355744,-0.0022724695267264425,0.00016320273714103516,-0.03283437462884927,0.023500834664117076,0.05568338209655401,0.06913878390790747,-0.025305667153704487,-0.022165618605172372,-0.01842856720068264,-0.07657789123023183,-0.02496213158330192,-0.021755954820698063,0.013038457308461327,-0.0650250016422321,-0.056893277586973365,0.012522384423864571,-0.06993807889500693,0.06636844613965502,-0.04720404904416439,0.02230666737229115,0.04861068292662666,-0.013092930707195003,0.002404400529968132,-0.001599317394474342,-0.030793678016555918,-0.03940839481045314,-0.03680548451786339,-0.06696752536799822,-0.00999235664311244,-0.029859937769778445,-0.00955514599169136,-0.008834419135113897,-0.023845380071315515,-0.03846403385952657,0.03698221084763036,0.032752190009473235,-0.029644669317778243,0.015529139070716201,0.04679545193071351,0.016159774596640634,0.06658986936626468,-0.0035848799247548274,-0.00024403402835975217,0.070746072376219,-0.058946204418482334,0.0500813801057227,0.012648225573846454,-0.07451525753839452,-0.07115737086890787,0.059743745153663194,0.0035787197210951213,0.03098234532086616,0.064537242977324,0.01715142230089393,-0.046681553456081486,0.02747607486052472,-0.020048862913918768,-0.03125120322749656"
	use_sr = 0
	use_cleanup = 0

	data = {
		'pluginsContext': '{}',
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

	try:
		response = requests.post('http://0.0.0.0:8008/synthesize', json=data)
		response.raise_for_status()  # If the response contains an HTTP error status code, raise an exception
	except requests.exceptions.RequestException as err:
		print('Failed to synthesize!')
		print('server.log contents:')
		with open('resources/app/server.log', 'r') as f:
			print(f.read())

	byte_data = None
	with open(save_path, "rb") as file:
		byte_data = file.read()
	return (22050, byte_data)

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
	outputs="audio",
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
