"use strict"

const path = require('path')
const smi = require('node-nvidia-smi')

window.batch_state = {
    lines: [],
    fastModeActuallyFinishedTasks: 0,
    fastModeOutputPromises: [],
    lastModel: undefined,
    lastVocoder: undefined,
    lineIndex: 0,
    state: false,
    outPathsChecked: [],
    skippedExisting: 0,
    paginationIndex: 0,
    taskBarPercent: 0,
    startTime: undefined,
    linesDoneSinceStart: 0
}


// https://stackoverflow.com/questions/1293147/example-javascript-code-to-parse-csv-data
function CSVToArray( strData, strDelimiter ){
    // Check to see if the delimiter is defined. If not,
    // then default to comma.
    strDelimiter = (strDelimiter || ",");

    // Create a regular expression to parse the CSV values.
    var objPattern = new RegExp(
        (
            // Delimiters.
            "(\\" + strDelimiter + "|\\r?\\n|\\r|^)" +

            // Quoted fields.
            "(?:\"([^\"]*(?:\"\"[^\"]*)*)\"|" +

            // Standard fields.
            "([^\"\\" + strDelimiter + "\\r\\n]*))"
        ),
        "gi"
        );

    // Create an array to hold our data. Give the array
    // a default empty first row.
    var arrData = [[]];

    // Create an array to hold our individual pattern
    // matching groups.
    var arrMatches = null;

    // Keep looping over the regular expression matches
    // until we can no longer find a match.
    while (arrMatches = objPattern.exec( strData )){

        // Get the delimiter that was found.
        var strMatchedDelimiter = arrMatches[ 1 ];

        // Check to see if the given delimiter has a length
        // (is not the start of string) and if it matches
        // field delimiter. If id does not, then we know
        // that this delimiter is a row delimiter.
        if (
            strMatchedDelimiter.length &&
            strMatchedDelimiter !== strDelimiter
            ){
            // Since we have reached a new row of data,
            // add an empty row to our data array.
            arrData.push( [] );
        }

        var strMatchedValue;

        // Now that we have our delimiter out of the way,
        // let's check to see which kind of value we
        // captured (quoted or unquoted).
        if (arrMatches[ 2 ]){
            // We found a quoted value. When we capture
            // this value, unescape any double quotes.
            strMatchedValue = arrMatches[ 2 ].replace(
                new RegExp( "\"\"", "g" ),
                "\""
                );
        } else {
            // We found a non-quoted value.
            strMatchedValue = arrMatches[ 3 ];

        }

        // Now that we have our value string, let's add
        // it to the data array.
        arrData[ arrData.length - 1 ].push( strMatchedValue );
    }

    // Return the parsed data.
    return( arrData );
}
window.CSVToArray = CSVToArray

let smiInterval = setInterval(() => {
    try {
        if (window.userSettings.useGPU) {
            smi((err, data) => {
                if (err) {
                    console.log("smi error: ", err)
                }
                if (data && data.nvidia_smi_log.cuda_version) {
                    let total
                    let used

                    if (data.nvidia_smi_log.gpu.length) {
                        total = parseInt(data.nvidia_smi_log.gpu[0].fb_memory_usage.total.split(" ")[0])
                        used = parseInt(data.nvidia_smi_log.gpu[0].fb_memory_usage.used.split(" ")[0])
                    } else {
                        total = parseInt(data.nvidia_smi_log.gpu.fb_memory_usage.total.split(" ")[0])
                        used = parseInt(data.nvidia_smi_log.gpu.fb_memory_usage.used.split(" ")[0])
                    }
                    const percent = used/total*100

                    vramUsage.innerHTML = `${(used/1000).toFixed(1)}/${(total/1000).toFixed(1)} GB (${percent.toFixed(2)}%)`
                }
            })
        } else {
            vramUsage.innerHTML = window.i18n.NOT_USING_GPU
        }
    } catch (e) {
        console.log(e)
        window.appLogger.log(e.stack)
        clearInterval(smiInterval)
    }
}, 1000)

batch_instructions_btn.addEventListener("click", () => {
    window.createModal("error", `${window.i18n.BATCH_INSTR1} <a href="https://www.youtube.com/watch?v=PK-m54f84q4" target="_blank">${window.i18n.BATCH_INSTR2}</a> <span>${window.i18n.BATCH_INSTR3}</span>`)
})

batch_generateSample.addEventListener("click", () => {
    // const lines = []
    const csv = ["game_id,voice_id,text,vocoder,out_path,pacing"] // TODO: ffmpeg options
    const games = Object.keys(window.games)

    if (games.length==0) {
        window.errorModal(window.i18n.BATCH_ERR_NO_VOICES)
        return
    }

    const sampleText = [
        "Include as many lines of text you wish with one line of data per line of voice to be read out.",
        "Make sure that the required columns (game_id, voice_id, and text) are filled out",
        "The others can be left blank, and the app will figure out some sensible defaults",
        "The valid options for vocoder are one of: quickanddirty, waveglow, waveglowBIG, hifi",
        "If your specified model does not have a bespoke hifi model, it will use the waveglow model, also the default if you leave this blank.",
        "For all the other options, you can leave them blank.",
        "If no output path is specified for a specific voice, the default batch output directory will be used",
    ]

    sampleText.forEach(line => {
        const game = games[parseInt(Math.random()*games.length)]
        const gameModels = window.games[game].models
        const model = gameModels[parseInt(Math.random()*gameModels.length)].variants[0]

        console.log("game", game, "model", model)

        const record = {
            game_id: game,
            voice_id: model.voiceId,
            text: line,
            vocoder: ["quickanddirty","waveglow","waveglowBIG","hifi"][parseInt(Math.random()*4)],
            out_path: "",
            pacing: 1
        }
        // lines.push(record)
        csv.push(Object.values(record).map(v => typeof v =="string" ? `"${v}"` : v).join(","))
    })

    const out_directory = `${__dirname.replace("\\javascript", "").replace(/\\/g,"/")}/batch`.replace(/\/\//g, "/").replace("resources/app/resources/app", "resources/app")
    if (!fs.existsSync(out_directory)){
        fs.mkdirSync(out_directory)
    }
    fs.writeFileSync(`${out_directory}/sample.csv`, csv.join("\n"))
    // shell.showItemInFolder(`${out_directory}/sample.csv`)
    er.shell.showItemInFolder(`${out_directory}/sample.csv`)
    spawn(`explorer`, [out_directory], {stdio: "ignore"})
})

window.readFileTxt = (file) => {
    return new Promise((resolve, reject) => {
        const dataLines = []
        const reader = new FileReader()
        reader.readAsText(file)
        reader.onloadend = () => {
            const lines = reader.result.replace(/\r\n/g, "\n").split("\n")
            lines.forEach(line => {
                if (line.trim().length) {
                    const record = {}
                    record.game_id = window.currentModel.games[0].gameId
                    record.voice_id = window.currentModel.games[0].voiceId
                    record.text = line
                    if (window.currentModel.hifi) {
                        record.vocoder = "hifi"
                    }
                    dataLines.push(record)
                }
            })
            resolve(dataLines)
        }
    })
}

window.readFile = (file) => {
    return new Promise((resolve, reject) => {
        const dataLines = []
        const reader = new FileReader()
        reader.readAsText(file)
        reader.onloadend = () => {
            const lines = reader.result.replace(/\r\n/g, "\n").split("\n")
            const headerLine = lines.shift()

            const doRest = () => {

                const header = headerLine.split(window.userSettings.batch_delimiter).map(head => head.replace(/\r/, ""))

                lines.forEach(line => {
                    const record = {}
                    if (line.trim().length) {
                        const parts = CSVToArray(line, window.userSettings.batch_delimiter)[0]
                        parts.forEach((val, vi) => {
                            try {
                                let header_val = header[vi].replace(/^"/, "").replace(/"$/, "")
                                record[header_val.replace(/\s/g,"")] = (val||"").replace(/\\/g, "/")
                            } catch (e) {
                                window.errorModal(`Error parsing line: ${val}`)
                                console.log(e)
                                window.appLogger.log(e)
                            }
                        })
                        dataLines.push(record)
                    }
                })
                resolve(dataLines)
            }

            const headerLine_clean = headerLine.replaceAll("\"", "")
            if (headerLine_clean.includes(window.userSettings.batch_delimiter)) {
                doRest()
            } else {
                const potentialDelimiter = headerLine_clean.split("voice_id")[0].split("game_id")[1]

                window.confirmModal(window.i18n.BATCH_CHANGE_DELIMITER.replace("_1", window.userSettings.batch_delimiter).replace("_2", potentialDelimiter)).then(response => {
                    if (response) {
                        window.userSettings.batch_delimiter = potentialDelimiter
                        setting_batch_delimiter.value = potentialDelimiter
                        window.saveUserSettings()
                        doRest()
                    }
                })
            }
        }
    })
}

let handleMetadataCSVdrop_response = undefined
const sleep = ms => {
    return new Promise(resolve => {
        setTimeout(() => {
            resolve()
        }, ms)
    })
}
const handleMetadataCSVdrop = (file) => {
    return new Promise(async resolve => {
        i18n_batch_metadata_open_btn.click()

        handleMetadataCSVdrop_response = undefined
        batch_metadata_input_gameID.value = window.currentGame.gameId
        if (window.currentModel) {
            batch_metadata_input_voiceID.value = window.currentModel.voiceId

        }

        await sleep(1000)
        while (handleMetadataCSVdrop_response === undefined) {
            if (batchMetadataCSVContainer.style.opacity.length==0 || parseFloat(batchMetadataCSVContainer.style.opacity)==1) {
                await sleep(1000)
            } else {
                handleMetadataCSVdrop_response = false
            }
        }

        if (handleMetadataCSVdrop_response) {

            const dataLines = []
            const reader = new FileReader()
            reader.readAsText(file)
            reader.onloadend = () => {
                const lines = reader.result.replace(/\r\n/g, "\n").split("\n")
                lines.forEach(line => {
                    if (line.trim().length) {

                        const text = line.split("|")[1]
                        const record = {}
                        record.game_id = batch_metadata_input_gameID.value
                        record.voice_id = batch_metadata_input_voiceID.value
                        record.text = text
                        if (!window.currentModel || window.currentModel.hifi) {
                            record.vocoder = "hifi"
                        }
                        dataLines.push(record)
                    }
                })
                resolve(dataLines)
            }

        } else {
            resolve(false)
        }


    })
}
i18n_batch_metadata_confirm_btn.addEventListener("click", () => {
    handleMetadataCSVdrop_response = true
    batchMetadataCSVContainer.click()
    batchIcon.click()
})

window.uploadBatchCSVs = async (eType, event) => {

    if (["dragenter", "dragover"].includes(eType)) {
        batch_main.style.background = "#5b5b5b"
        batch_main.style.color = "white"
    }
    if (["dragleave", "drop"].includes(eType)) {
        batch_main.style.background = "#4b4b4b"
        batch_main.style.color = "gray"
    }

    event.preventDefault()
    event.stopPropagation()

    const dataLines = []

    if (eType=="drop") {

        batchDropZoneNote.innerHTML = window.i18n.PROCESSING_DATA
        window.batch_state.skippedExisting = 0

        const dataTransfer = event.dataTransfer
        const files = Array.from(dataTransfer.files)
        for (let fi=0; fi<files.length; fi++) {
            const file = files[fi]

            if (!file.name.toLowerCase().endsWith(".csv") || file.name.toLowerCase()=="metadata.csv") {
                if ( file.name.toLowerCase().endsWith(".txt") || file.name.toLowerCase()=="metadata.csv" ) {
                    if (window.currentModel || file.name.toLowerCase()=="metadata.csv") {
                        window.appLogger.log(`Reading file: ${file.name}`)

                        let records

                        if (file.name.toLowerCase()=="metadata.csv") {

                            records = await handleMetadataCSVdrop(file)
                            if (records===false) {
                                continue
                            }

                        } else {
                            if (window.currentModel) {
                                records = await window.readFileTxt(file)
                            }
                        }

                        if (window.currentModel || file.name.toLowerCase()=="metadata.csv") {
                            if (window.userSettings.batch_skipExisting) {
                                window.appLogger.log("Checking existing files before adding to queue")
                            } else {
                                window.appLogger.log("Adding files to queue")
                            }
                            records.forEach(item => {
                                let outPath

                                if (item.out_path && item.out_path.split("/").reverse()[0].includes(".")) {
                                    outPath = item.out_path
                                } else {
                                    if (item.out_path) {
                                        outPath = item.out_path
                                    } else {
                                        outPath = window.userSettings.batchOutFolder
                                    }
                                    if (item.vc_content) {
                                        let vc_content_fname = item.vc_content.split("/").reverse()[0]
                                        outPath = `${outPath}/${vc_content_fname.slice(0,vc_content_fname.length-4).slice(0, window.userSettings.max_filename_chars-10).replace(/\.$/, "")}.${window.userSettings.audio.format}`
                                    } else {
                                        outPath = `${outPath}/${item.voice_id}_${item.vocoder}_${item.text.replace(/[\/\\:\*?<>"|]*/g, "").slice(0, window.userSettings.max_filename_chars-10).replace(/\.$/, "")}.${window.userSettings.audio.format}`
                                    }
                                }

                                outPath = outPath.startsWith("./") ? window.userSettings.batchOutFolder + outPath.slice(1,100000) : outPath
                                item.out_path = outPath

                                if (window.userSettings.batch_skipExisting && fs.existsSync(outPath)) {
                                    window.batch_state.skippedExisting++
                                } else {
                                    dataLines.push(item)
                                }
                            })
                        }

                    }
                    continue
                } else {
                    continue
                }
            }

            window.appLogger.log(`Reading file: ${file.name}`)
            const records = await window.readFile(file)
            if (window.userSettings.batch_skipExisting) {
                window.appLogger.log("Checking existing files before adding to queue")
            } else {
                window.appLogger.log("Adding files to queue")
            }
            records.forEach((item, ii) => {
                let outPath

                if (item.out_path && item.out_path.split("/").reverse()[0].includes(".")) {
                    outPath = item.out_path
                } else {
                    if (item.out_path) {
                        outPath = item.out_path
                    } else {
                        outPath = window.userSettings.batchOutFolder
                    }
                    if (item.vc_content) {
                        let vc_content_fname = item.vc_content.split("/").reverse()[0]
                        outPath = `${outPath}/${vc_content_fname.slice(0,vc_content_fname.length-4).slice(0,item.vc_content.length-4).slice(0, window.userSettings.max_filename_chars-10).replace(/\.$/, "")}.${window.userSettings.audio.format}`
                    } else {
                        outPath = `${outPath}/${item.voice_id}_${item.vocoder}_${item.text.replace(/[\/\\:\*?<>"|]*/g, "").slice(0, window.userSettings.max_filename_chars-10).replace(/\.$/, "")}.${window.userSettings.audio.format}`
                    }
                }

                outPath = outPath.startsWith("./") ? window.userSettings.batchOutFolder + outPath.slice(1,100000) : outPath
                item.out_path = outPath

                if (window.userSettings.batch_skipExisting && fs.existsSync(outPath)) {
                    window.batch_state.skippedExisting++
                } else {
                    dataLines.push(item)
                }
            })
        }

        if (dataLines.length==0 && window.batch_state.skippedExisting) {
            batchDropZoneNote.innerHTML = window.i18n.BATCH_DROPZONE
            return window.errorModal(window.i18n.BATCH_ERR_SKIPPEDALL.replace("_1", window.batch_state.skippedExisting))
        }

        window.batch_state.paginationIndex = 0
        batch_pageNum.value = 1


        window.appLogger.log("Preprocessing data...")
        const cleanedData = window.preProcessCSVData(dataLines)
        if (cleanedData.length) {
            window.populateBatchRecordsList(cleanedData)
            window.appLogger.log("Grouping up lines...")
            const finalOrder = window.groupBatchLines()
            window.refreshBatchRecordsList(finalOrder)
            window.batch_state.lines = finalOrder
        } else {
            // batch_clearBtn.click()
        }
        window.appLogger.log("batch import done")

        const numPages = Math.ceil(window.batch_state.lines.length/window.userSettings.batch_paginationSize)
        batch_total_pages.innerHTML = `of ${numPages}`
        batchDropZoneNote.innerHTML = window.i18n.BATCH_DROPZONE
    }
}

window.preProcessCSVData = data => {

    batch_main.style.display = "block"
    batchDropZoneNote.style.display = "none"
    batchRecordsHeader.style.display = "flex"
    batch_clearBtn.style.display = "inline-block"
    Array.from(batchRecordsHeader.children).forEach(item => item.style.backgroundColor = `#${window.currentGame.themeColourPrimary}`)

    const availableGames = Object.keys(window.games)
    for (let di=0; di<data.length; di++) {
        try {
            const record = data[di]

            // Validate the records first
            // ==================
            if (!record.game_id) {
                window.errorModal(`[${window.i18n.LINE}: ${di+2}] ${window.i18n.ERROR}: ${window.i18n.MISSING} game_id`)
                return []
            }
            if (!record.voice_id) {
                window.errorModal(`[${window.i18n.LINE}: ${di+2}] ${window.i18n.ERROR}: ${window.i18n.MISSING} voice_id`)
                return []
            }
            if ((!record.text || record.text.length==0) && (!record.vc_content)) {
                window.errorModal(`[${window.i18n.LINE}: ${di+2}] ${window.i18n.ERROR}: ${window.i18n.MISSING} text/vc_content`)
                return []
            }

            // Check that the game_id exists
            if (!availableGames.includes(record.game_id)) {
                window.errorModal(`[${window.i18n.LINE}: ${di+2}] ${window.i18n.ERROR}: game_id "${record.game_id}" ${window.i18n.BATCH_ERR_GAMEID} <br><br>(${availableGames.join(', ')})`)
                return []
            }
            // Check that the voice_id exists
            const gameVoices = []
            window.games[record.game_id].models.forEach(model => {
                model.variants.forEach(variant => gameVoices.push(variant.voiceId))
            })

            if (!gameVoices.includes(record.voice_id)) {
                window.errorModal(`[${window.i18n.LINE}: ${di+2}] ${window.i18n.ERROR}: voice_id "${record.voice_id}" ${window.i18n.BATCH_ERR_VOICEID}: ${record.game_id}`)
                return []
            }
            // Check that the vocoder exists
            if (!["quickanddirty", "waveglow", "waveglowBIG", "hifi", "", "-", undefined].includes(record.vocoder)) {
                window.errorModal(`[${window.i18n.LINE}: ${di+2}] ${window.i18n.ERROR}: ${window.i18n.BATCHH_VOCODER} "${record.vocoder}" ${window.i18n.BATCH_ERR_VOCODER1}: quickanddirty, waveglow, waveglowBIG, hifi ${window.i18n.BATCH_ERR_VOCODER2}`)
                return []
            }

            data[di].modelType = undefined
            let hasHifi = false
            window.games[data[di].game_id].models.forEach(model => {
                model.variants.forEach(variant => {
                    if (variant.voiceId==data[di].voice_id) {
                        data[di].modelType = variant.modelType || model.modelType
                        record.voiceName = model.voiceName // For easy access later on
                        if (variant.hifi) {
                            hasHifi = variant.hifi
                        }
                        if (variant.lang && !record.lang) {
                            record.lang = "en"
                        }
                        // TODO allow batch mode voice conversion/speech to speech by computing the embs of specified audio files instead of using the base voice embedding
                        // Also TODO, might need to allow for custom voice embeddings
                        if (variant.modelType=="xVAPitch") {
                            record.base_emb = variant.base_speaker_emb
                            record.vocoder = "-"
                        }
                    }
                })
            })

            // Fill with defaults
            // ==================
            if (!record.out_path) {
                record.out_path = window.userSettings.batchOutFolder
            }
            if (!record.pacing) {
                record.pacing = 1
            }
            record.pacing = parseFloat(record.pacing)


            if (!record.pitch_amp) {
                record.pitch_amp = 1
            }
            record.pitch_amp = parseFloat(record.pitch_amp)

            if (!record.out_path.includes(":/") && !record.out_path.startsWith("./")) {
                record.out_path = `./${record.out_path}`
            }

            if (!record.vocoder || (record.vocoder=="hifi" && !hasHifi)) {
                record.vocoder = "quickanddirty"
            }
        } catch (e) {
            console.log(e)
            window.appLogger.log(e)
            console.log(data[di])
            console.log(window.games[data[di].game_id])
        }
    }

    return data
}

window.populateBatchRecordsList = records => {
    batch_synthesizeBtn.style.display = "inline-block"
    batchDropZoneNote.style.display = "none"

    records.forEach((record, ri) => {
        const row = createElem("div")

        const rNumElem = createElem("div", batchRecordsContainer.children.length.toString())
        const rStatusElem = createElem("div", "Ready")
        const rActionsElem = createElem("div")
        const rVoiceElem = createElem("div", record.voice_id)
        const rTextElem = createElem("div", (record.vc_content?record.vc_content:record.text).toString())
        rTextElem.title = rTextElem.innerText
        if (record.vc_content) {
            rTextElem.style.fontStyle = "italic"
        }
        const rGameElem = createElem("div", record.game_id)
        const rOutPathElem = createElem("div", "&lrm;"+record.out_path+"&lrm;")
        rOutPathElem.title = record.out_path
        const rBaseLangElem = createElem("div", record.vc_content?"-":(record.lang||" ").toString())
        const rVCStyleElem = createElem("div", (record.vc_style||" ").toString())
        rVCStyleElem.title = rVCStyleElem.innerText
        const rVocoderElem = createElem("div", record.vocoder)
        const rPacingElem = createElem("div", record.vc_content?"-":(record.pacing||" ").toString())
        const rPitchAmpElem = createElem("div", record.vc_content?"-":(record.pitch_amp||" ").toString())


        row.appendChild(rNumElem)
        row.appendChild(rStatusElem)
        row.appendChild(rActionsElem)
        row.appendChild(rVoiceElem)
        row.appendChild(rTextElem)
        row.appendChild(rGameElem)
        row.appendChild(rOutPathElem)
        row.appendChild(rBaseLangElem)
        row.appendChild(rVCStyleElem)
        row.appendChild(rVocoderElem)
        row.appendChild(rPacingElem)
        row.appendChild(rPitchAmpElem)

        window.batch_state.lines.push([record, row, ri])
    })
}

window.refreshBatchRecordsList = (finalOrder) => {
    batchRecordsContainer.innerHTML = ""
    finalOrder = finalOrder ? finalOrder : window.batch_state.lines

    const startIndex = (window.batch_state.paginationIndex*window.userSettings.batch_paginationSize)
    const endIndex = Math.min(startIndex+window.userSettings.batch_paginationSize, finalOrder.length)

    for (let ri=startIndex; ri<endIndex; ri++) {
        const recordAndElem = finalOrder[ri]
        recordAndElem[1].children[0].innerHTML = (ri+1)//batchRecordsContainer.children.length.toString()
        batchRecordsContainer.appendChild(recordAndElem[1])
    }
    window.toggleNumericalRecordsDisplay()
}

// Sort the lines by voice_id, and then by vocoder used
window.groupBatchLines = () => {
    if (window.userSettings.batch_doGrouping) {
        const voices_order = []

        const lines = window.batch_state.lines.sort((a,b) => {
            return a.voice_id - b.voice_id
        })

        const voices_groups = {}

        // Get the order of the voice_id, and group them up
        window.batch_state.lines.forEach(record => {
            if (!voices_order.includes(record[0].voice_id)) {
                voices_order.push(record[0].voice_id)
                voices_groups[record[0].voice_id] = []
            }
            voices_groups[record[0].voice_id].push(record)
        })

        // Go through the voice groups and sort them by vocoder
        if (window.userSettings.batch_doVocoderGrouping) {
            voices_order.forEach(voice_id => {
                voices_groups[voice_id] = voices_groups[voice_id].sort((a,b) => a[0].vocoder<b[0].vocoder?1:-1)
            })
        }

        // Collate everything back into the final order
        const finalOrder = []
        voices_order.forEach(voice_id => {
            voices_groups[voice_id].forEach(record => finalOrder.push(record))
        })

        return finalOrder

    } else {
        return window.batch_state.lines
    }
}


batch_clearBtn.addEventListener("click", () => {

    window.batch_state.lines = []
    batch_main.style.display = "flex"
    batchDropZoneNote.style.display = "block"
    batchRecordsHeader.style.display = "none"
    batch_clearBtn.style.display = "none"
    batch_outputFolderInput.style.display = "inline-block"
    batch_clearDirOpts.style.display = "flex"
    batch_skipExistingOpts.style.display = "flex"
    batch_useSR.style.display = "flex"
    batch_useCleanup.style.display = "flex"
    batch_outputNumericallyOpts.style.display = "flex"
    batch_progressItems.style.display = "none"
    batch_progressBar.style.display = "none"

    batch_pauseBtn.style.display = "none"
    batch_stopBtn.style.display = "none"
    batch_synthesizeBtn.style.display = "none"

    batchRecordsContainer.innerHTML = ""
})


window.startBatch = () => {

    // Output directory
    if (!fs.existsSync(window.userSettings.batchOutFolder)) {
        window.userSettings.batchOutFolder.split("/")
        .reduce((prevPath, folder) => {
            const currentPath = path.join(prevPath, folder, path.sep);
            if (!fs.existsSync(currentPath)){
                try {
                    fs.mkdirSync(currentPath);
                } catch (e) {
                    window.errorModal(`${window.i18n.SOMETHING_WENT_WRONG}:<br><br>`+e.message)
                    throw ""
                }
            }
            return currentPath;
         }, '');
    }
    if (batch_clearDirFirstCkbx.checked) {
        const filesAndFolders = fs.readdirSync(window.userSettings.batchOutFolder)
        filesAndFolders.forEach(faf => {
            // Ignore .csv and .txt files
            if (faf.toLowerCase().endsWith(".csv") || faf.toLowerCase().endsWith(".txt")) {
                return
            }
            if (fs.lstatSync(`${window.userSettings.batchOutFolder}/${faf}`).isDirectory()) {
                window.deleteFolderRecursive(`${window.userSettings.batchOutFolder}/${faf}`, false)
            } else {
                fs.unlinkSync(`${window.userSettings.batchOutFolder}/${faf}`)
            }
            console.log(`${window.userSettings.batchOutFolder}/${faf}`,  )
        })
    }

    batch_synthesizeBtn.style.display = "none"
    batch_clearBtn.style.display = "none"
    batch_outputFolderInput.style.display = "none"
    batch_clearDirOpts.style.display = "none"
    batch_skipExistingOpts.style.display = "none"
    batch_useSR.style.display = "none"
    batch_useCleanup.style.display = "none"
    batch_outputNumericallyOpts.style.display = "none"
    batch_progressItems.style.display = "flex"
    batch_progressBar.style.display = "flex"
    batch_pauseBtn.style.display = "inline-block"
    batch_stopBtn.style.display = "inline-block"
    batch_openDirBtn.style.display = "none"

    window.batch_state.lines.forEach(record => {
        record[1].children[1].innerHTML = window.i18n.READY
        record[1].children[1].style.background = "none"
    })

    window.batch_state.fastModeOutputPromises = []
    window.batch_state.fastModeActuallyFinishedTasks = 0
    window.batch_state.lineIndex = 0
    window.batch_state.state = true
    window.batch_state.outPathsChecked = []
    window.batch_state.startTime = new Date()
    window.batch_state.linesDoneSinceStart = 0
    window.performSynthesis()
}

window.batchChangeVoice = (game, voice, modelType) => {
    return new Promise((resolve) => {
        if (!window.batch_state.state) {
            return resolve()
        }
        // Update the main app with any changes, if a voice has already been selected
        if (window.currentModel) {
            generateVoiceButton.innerHTML = window.i18n.LOAD_MODEL
            keepSampleButton.style.display = "none"
            wavesurferContainer.innerHTML = ""

            const modelGameFolder = window.currentModel.audioPreviewPath.split("/")[0]
            const modelFileName = window.currentModel.audioPreviewPath.split("/")[1].split(".wav")[0]
            generateVoiceButton.dataset.modelQuery = JSON.stringify({
                outputs: parseInt(window.currentModel.outputs),
                model: `${window.path}/models/${modelGameFolder}/${modelFileName}`,
                model_speakers: window.currentModel.emb_size,
                cmudict: window.currentModel.cmudict
            })
        }


        if (window.batch_state.state) {
            batch_progressNotes.innerHTML = `${window.i18n.BATCH_CHANGING_MODEL_TO}: ${voice}`
        }

        let model
        window.games[game].models.forEach(gameModel => {
            gameModel.variants.forEach(variant => {
                if (variant.voiceId==voice) {
                    model = variant
                }
            })
        })

        doFetch(`http://localhost:8008/loadModel`, {
            method: "Post",
            body: JSON.stringify({
                "modelType": modelType,
                "outputs": null,
                "model": `${window.userSettings[`modelspath_${game}`]}/${voice}`,
                "model_speakers": model.num_speakers,
                "base_lang": model.lang,
                "pluginsContext": JSON.stringify(window.pluginsContext)
            })
        }).then(r=>r.text()).then(res => {
            resolve()
        }).catch(async e => {
            if (e.code=="ECONNREFUSED" || e.code=="ECONNRESET") {
                await window.batchChangeVoice(game, voice, modelType)
                resolve()
            } else {
                console.log(e)
                window.appLogger.log(e)
                batch_pauseBtn.click()

                if (document.getElementById("activeModal")) {
                    activeModal.remove()
                }
                if (e.code=="ENOENT") {
                    window.errorModal(window.i18n.ERR_SERVER)
                } else {
                    window.errorModal(e.message)
                }
                resolve()
            }
        })
    })
}
window.batchChangeVocoder = (vocoder, game, voice) => {
    return new Promise((resolve) => {
        if (!window.batch_state.state) {
            return resolve()
        }
        console.log("Changing vocoder: ", vocoder)
        if (window.batch_state.state) {
            batch_progressNotes.innerHTML = `${window.i18n.BATCH_CHANGING_VOCODER_TO}: ${vocoder}`
        }

        const vocoderMappings = [["waveglow", "256_waveglow"], ["waveglowBIG", "big_waveglow"], ["quickanddirty", "qnd"], ["hifi", `${game}/${voice}.hg.pt`]]
        const vocoderId = vocoderMappings.find(record => record[0]==vocoder)[1]

        doFetch(`http://localhost:8008/setVocoder`, {
            method: "Post",
            body: JSON.stringify({
                vocoder: vocoderId,
                modelPath: vocoderId=="256_waveglow" ? window.userSettings.waveglow_path : window.userSettings.bigwaveglow_path,
            })
        }).then(r=>r.text()).then((res) => {
            if (res=="ENOENT") {
                closeModal(undefined, batchGenerationContainer).then(() => {
                    setTimeout(() => {
                        vocoder_select.value = window.userSettings.vocoder
                        window.errorModal(`${window.i18n.BATCH_MODEL_NOT_FOUND}.${vocoderId.includes("waveglow")?" "+window.i18n.BATCH_DOWNLOAD_WAVEGLOW:""}`)
                        batch_pauseBtn.click()
                        resolve()
                    }, 300)
                })
            } else {
                window.batch_state.lastVocoder = vocoder
                resolve()
            }
        }).catch(async e => {
            if (e.code=="ECONNREFUSED" || e.code=="ECONNRESET") {
                await window.batchChangeVocoder(vocoder, game, voice)
                resolve()
            } else {
                console.log(e)
                window.appLogger.log(e)
                batch_pauseBtn.click()

                if (document.getElementById("activeModal")) {
                    activeModal.remove()
                }
                if (e.code=="ENOENT") {
                    window.errorModal(window.i18n.ERR_SERVER)
                } else {
                    window.errorModal(e.message)
                }
                resolve()
            }
        })
    })
}


window.prepareLinesBatchForSynth = () => {

    const linesBatch = []
    const records = []
    let firstItemVoiceId = undefined
    let firstItemVocoder = undefined
    let speaker_i = 0

    for (let i=0; i<Math.min(window.userSettings.batch_batchSize, window.batch_state.lines.length-window.batch_state.lineIndex); i++) {

        const record = window.batch_state.lines[window.batch_state.lineIndex+i]

        const vocoderMappings = [["waveglow", "256_waveglow"], ["waveglowBIG", "big_waveglow"], ["quickanddirty", "qnd"], ["hifi", `${record[0].game_id}/${record[0].voice_id}.hg.pt`]]
        const vocoder = record[0].vocoder=="-"?"-":vocoderMappings.find(voc => voc[0]==record[0].vocoder)[1]

        if (firstItemVoiceId==undefined) firstItemVoiceId = record[0].voice_id
        if (firstItemVocoder==undefined) firstItemVocoder = vocoder

        if (record[0].voice_id!=firstItemVoiceId || vocoder!=firstItemVocoder) {
            break
        }

        let model
        window.games[record[0].game_id].models.forEach(gamesModel => {
            gamesModel.variants.forEach(variant => {
                if (variant.voiceId==record[0].voice_id) {
                    model = variant
                }
            })
        })

        const sequence = record[0].text
        const pitch = undefined // maybe later
        const duration = undefined // maybe later
        speaker_i = model.emb_i || 0
        let pace = record[0].pacing
        pace = Number.isNaN(pace) ? 1.0 : pace
        let pitch_amp = record[0].pitch_amp
        pitch_amp = Number.isNaN(pitch_amp) ? 1.0 : pitch_amp

        const tempFileNum = `${Math.random().toString().split(".")[1]}`
        const tempFileLocation = `${window.path}/output/temp-${tempFileNum}.wav`

        let outPath
        let outFolder

        outPath = record[0].out_path
        outFolder = String(record[0].out_path).split("/").reverse().slice(1,10000).reverse().join("/")
        outFolder = outFolder.length ? outFolder : window.userSettings.batchOutFolder

        if (batch_outputNumerically.checked) {
            outPath = `${window.userSettings.batchOutFolder}/${String(record[2]).padStart(10, '0')}.${outPath.split(".").reverse()[0]}`
        } else {
            outPath = outPath.startsWith("./") ? window.userSettings.batchOutFolder + outPath.slice(1,100000) : outPath
        }

        linesBatch.push([sequence, pitch, duration, pace, tempFileLocation, outPath, outFolder, pitch_amp, record[0].lang, record[0].base_emb, record[0].vc_content, record[0].vc_style])
        records.push(record)
    }

    return [speaker_i, firstItemVoiceId, firstItemVocoder, linesBatch, records]
}


window.addActionButtons = (records, ri) => {

    let audioPreview
    const playButton = createElem("button.smallButton", window.i18n.PLAY)
    playButton.style.background = `#${window.currentGame.themeColourPrimary}`
    playButton.addEventListener("click", () => {

        let audioPreviewPath = records[ri][0].fileOutputPath
        if (audioPreviewPath.startsWith("./")) {
            audioPreviewPath = window.userSettings.batchOutFolder + audioPreviewPath.replace("./", "/")
        }

        if (audioPreview==undefined) {
            const audioPreview = createElem("audio", {autoplay: false}, createElem("source", {
                src: audioPreviewPath
            }))
            audioPreview.addEventListener("play", () => {
                if (window.ctrlKeyIsPressed) {
                    audioPreview.setSinkId(window.userSettings.alt_speaker)
                } else {
                    audioPreview.setSinkId(window.userSettings.base_speaker)
                }
            })
            audioPreview.setSinkId(window.userSettings.base_speaker)
        }
    })
    records[ri][1].children[2].appendChild(playButton)

    // If not a Voice Conversion line
    if (!records[ri][0].vc_content) {
        const editButton = createElem("button.smallButton", window.i18n.EDIT)
        editButton.style.background = `#${window.currentGame.themeColourPrimary}`
        editButton.addEventListener("click", () => {
            audioPreview = undefined


            if (window.batch_state.state) {
                window.errorModal(window.i18n.BATCH_ERR_EDIT)
                return
            }

            // Change app theme to the voice's game
            if (window.currentGame.gameId!=records[ri][0].game_id) {
                window.changeGame(window.gameAssets[records[ri][0].game_id])
            }

            dialogueInput.value = records[ri][0].text

            // Simulate voice loading through the UI
            if (!window.currentModel || window.currentModel.voiceId != records[ri][0].voice_id) {
                const voiceName = records[ri][0].voiceName
                const voiceButton = Array.from(voiceTypeContainer.children).find(button => button.innerHTML==voiceName)
                voiceButton.click()
                vocoder_select.value = records[ri][0].vocoder=="hifi" ? `${records[ri][0].game_id}/${records[ri][0].voice_id}.hg.pt` : records[ri][0].vocoder
                generateVoiceButton.click()
            }
            window.closeModal(batchGenerationContainer)

            setTimeout(() => {
                let audioPreviewPath = records[ri][0].fileOutputPath
                if (audioPreviewPath.startsWith("./")) {
                    audioPreviewPath = window.userSettings.batchOutFolder + audioPreviewPath.replace("./", "/")
                }
                keepSampleButton.dataset.newFileLocation = "BATCH_EDIT"+audioPreviewPath
                generateVoiceButton.click()
            }, 500)
        })
        records[ri][1].children[2].appendChild(editButton)
    }
}


window.batchKickOffMPffmpegOutput = (records, tempPaths, outPaths, options, extraInfo) => {
    let hasShownError = false
    return new Promise((resolve, reject) => {
        doFetch(`http://localhost:8008/batchOutputAudio`, {
            method: "Post",
            body: JSON.stringify({
                input_paths: tempPaths,
                output_paths: outPaths,
                isBatchMode: true,
                pluginsContext: JSON.stringify(window.pluginsContext),
                processes: window.userSettings.batch_MPCount,
                extraInfo: extraInfo,
                options: JSON.stringify(options)
            })
        }).then(r=>r.text()).then(res => {
            res = res.split("\n")
            res.forEach((resItem, ri) => {
                if (resItem.length && resItem!="-") {
                    window.appLogger.log(`Batch error, item ${ri} - ${resItem}`)
                    if (window.batch_state.state) {
                        batch_pauseBtn.click()
                    }
                    window.errorModal(`${window.i18n.SOMETHING_WENT_WRONG}:<br><br>`+resItem)
                    hasShownError = true

                    records[ri][1].children[1].innerHTML = window.i18n.FAILED
                    records[ri][1].children[1].style.background = "red"

                } else {

                    records[ri][1].children[1].innerHTML = window.i18n.DONE
                    records[ri][1].children[1].style.background = "green"
                    fs.unlinkSync(tempPaths[ri])
                    window.addActionButtons(records, ri)
                }

                // if (!window.userSettings.batch_fastMode) { // No more fast modde. TODO, remove completely
                window.batch_state.lineIndex += 1
                // }

                window.batch_state.fastModeActuallyFinishedTasks += 1

            })
            const percentDone = (window.batch_state.fastModeActuallyFinishedTasks) / window.batch_state.lines.length * 100
            batch_progressBar.style.background = `linear-gradient(90deg, green ${parseInt(percentDone)}%, rgba(255,255,255,0) ${parseInt(percentDone)}%)`
            batch_progressBar.innerHTML = `${parseInt(percentDone* 100)/100}%`
            window.batch_state.taskBarPercent = percentDone/100
            window.electronBrowserWindow.setProgressBar(window.batch_state.taskBarPercent)
            window.adjustETA()
            resolve()

        }).catch(async e => {
            if (e.code=="ECONNREFUSED" || e.code=="ECONNRESET") {
                await window.batchKickOffMPffmpegOutput(records, tempPaths, outPaths, options, extraInfo)
                resolve()
            } else {
                console.log(e)
                window.appLogger.log(e.stack)
                if (document.getElementById("activeModal")) {
                    activeModal.remove()
                }
                if (!hasShownError) {
                    window.errorModal(e.message)
                }
                resolve()
            }
        })
    })
}


window.batchKickOffFfmpegOutput = (ri, linesBatch, records, tempFileLocation, body) => {
    return new Promise((resolve, reject) => {
        doFetch(`http://localhost:8008/outputAudio`, {
            method: "Post",
            body
        }).then(r=>r.text()).then(res => {
            if (res.length && res!="-") {
                window.appLogger.log("res", res)
                if (window.batch_state.state) {
                    batch_pauseBtn.click()
                }

                for (let ri2=ri; ri2<linesBatch.length; ri2++) {
                    records[ri][1].children[1].innerHTML = window.i18n.FAILED
                    records[ri][1].children[1].style.background = "red"
                }

                reject(res)
            } else {
                records[ri][1].children[1].innerHTML = window.i18n.DONE
                records[ri][1].children[1].style.background = "green"
                fs.unlinkSync(tempFileLocation)
                window.addActionButtons(records, ri)
                window.batch_state.fastModeActuallyFinishedTasks += 1

                const percentDone = (window.batch_state.fastModeActuallyFinishedTasks) / window.batch_state.lines.length * 100
                batch_progressBar.style.background = `linear-gradient(90deg, green ${parseInt(percentDone)}%, rgba(255,255,255,0) ${parseInt(percentDone)}%)`
                batch_progressBar.innerHTML = `${parseInt(percentDone* 100)/100}%`
                window.batch_state.taskBarPercent = percentDone/100
                window.electronBrowserWindow.setProgressBar(window.batch_state.taskBarPercent)
                window.adjustETA()
                resolve()
            }
        }).catch(async e => {
            if (e.code=="ECONNREFUSED" || e.code=="ECONNRESET") {
                await window.batchKickOffFfmpegOutput(ri, linesBatch, records, tempFileLocation, body)
                resolve()
            } else {
                console.log(e)
                window.appLogger.log(e)
                batch_pauseBtn.click()
                if (document.getElementById("activeModal")) {
                    activeModal.remove()
                }
                window.errorModal(e.message)
                resolve()
            }
        })
    })
}

window.batchKickOffGeneration = () => {
    return new Promise((resolve) => {
        if (!window.batch_state.state) {
            return resolve()
        }
        const [speaker_i, voice_id, vocoder, linesBatch, records] = window.prepareLinesBatchForSynth()

        records.forEach((record, ri) => {
            record[1].children[1].innerHTML = window.i18n.RUNNING
            record[1].children[1].style.background = "goldenrod"
            record[0].fileOutputPath = linesBatch[ri][5]
        })

        const record = window.batch_state.lines[window.batch_state.lineIndex]

        if (window.batch_state.state) {
            if (linesBatch.length==1) {
                batch_progressNotes.innerHTML = `${window.i18n.SYNTHESIZING}: <i>${record[0].text}</i>`
            } else {
                batch_progressNotes.innerHTML = `${window.i18n.SYNTHESIZING} ${linesBatch.length} ${window.i18n.LINES}`
            }
        }
        const batchPostData = {
            modelType: records[0][0].modelType,
            batchSize: window.userSettings.batch_batchSize,
            defaultOutFolder: window.userSettings.batchOutFolder,
            pluginsContext: JSON.stringify(window.pluginsContext),
            outputJSON: window.userSettings.batch_json,
            useSR: batch_useSRCkbx.checked,
            useCleanup: batch_useCleanupCkbx.checked,
            speaker_i, vocoder, linesBatch
        }
        doFetch(`http://localhost:8008/synthesize_batch`, {
            method: "Post",
            body: JSON.stringify(batchPostData)
        }).then(r=>r.text()).then(async (res) => {

            if (res && res!="-") {
                if (res=="CUDA OOM") {
                    window.errorModal(window.i18n.BATCH_ERR_CUDA_OOM)
                } else {
                    window.errorModal(res.replace(/\n/g, "<br>"))
                }
                if (window.batch_state.state) {
                    batch_pauseBtn.click()
                }
                return
            }


            // Create the output directory if it does not exist
            linesBatch.forEach(record => {
                let outFolder = record[6].startsWith("./") ? window.userSettings.batchOutFolder + record[6].slice(1,100000) : record[6]

                if (!window.batch_state.outPathsChecked.includes(outFolder)) {
                    window.batch_state.outPathsChecked.push(outFolder)
                    if (!fs.existsSync(outFolder)) {
                        window.createFolderRecursive(outFolder)
                    }
                }
            })

            if (window.userSettings.audio.ffmpeg) {
                const options = {
                    hz: window.userSettings.audio.hz,
                    padStart: window.userSettings.audio.padStart,
                    padEnd: window.userSettings.audio.padEnd,
                    bit_depth: window.userSettings.audio.bitdepth,
                    amplitude: window.userSettings.audio.amplitude,
                    pitchMult: window.userSettings.audio.pitchMult,
                    tempo: window.userSettings.audio.tempo,
                    deessing: window.userSettings.audio.deessing,
                    nr: window.userSettings.audio.nr,
                    nf: window.userSettings.audio.nf,
                    useNR: window.userSettings.audio.useNR,
                    useSR: batch_useSRCkbx.checked,
                    useCleanup: batch_useCleanupCkbx.checked,
                }

                if (window.batch_state.state) {
                    batch_progressNotes.innerHTML = window.i18n.BATCH_OUTPUTTING_FFMPEG
                }

                const tempPaths = linesBatch.map(line => line[4])
                const outPaths = linesBatch.map((line, li) => {
                    let outPath = linesBatch[li][5].includes(":") || linesBatch[li][5].includes("./") ? linesBatch[li][5] : `${linesBatch[li][6]}/${linesBatch[li][5]}`
                    if (outPath.startsWith("./")) {
                        outPath = window.userSettings.batchOutFolder + outPath.slice(1,100000)
                    }
                    return outPath
                })

                if (window.userSettings.batch_useMP) {
                    const extraInfo = {
                        game: records.map(rec => rec[0].game_id),
                        voiceId: records.map(rec => rec[0].voice_id),
                        voiceName: records.map(rec => rec[0].voiceName),
                        inputSequence: records.map(rec => rec[0].text)
                    }
                    if (window.userSettings.batch_fastMode && false) { // No more fast mode. TODO, remove completely
                        window.batch_state.fastModeOutputPromises.push(window.batchKickOffMPffmpegOutput(records, tempPaths, outPaths, options, JSON.stringify(extraInfo)))
                        window.batch_state.lineIndex += records.length
                    } else {
                        await window.batchKickOffMPffmpegOutput(records, tempPaths, outPaths, options, JSON.stringify(extraInfo))
                    }
                } else {
                    for (let ri=0; ri<linesBatch.length; ri++) {
                        let tempFileLocation = tempPaths[ri]
                        let outPath = outPaths[ri]
                        try {
                            if (window.batch_state.state) {
                                records[ri][1].children[1].innerHTML = window.i18n.OUTPUTTING
                                const extraInfo = {
                                    game: records[ri][0].game_id,
                                    voiceId: records[ri][0].voiceId,
                                    voiceName: records[ri][0].voiceName,
                                    letters: records[ri][0].text
                                }

                                if (window.userSettings.batch_fastMode && false) { // No more fast modde. TODO, remove completely
                                    window.batch_state.fastModeOutputPromises.push(window.batchKickOffFfmpegOutput(ri, linesBatch, records, tempFileLocation, JSON.stringify({
                                        input_path: tempFileLocation,
                                        output_path: outPath,
                                        isBatchMode: true,
                                        pluginsContext: JSON.stringify(window.pluginsContext),
                                        extraInfo: JSON.stringify(extraInfo),
                                        options: JSON.stringify(options)
                                    })))
                                } else {
                                    await window.batchKickOffFfmpegOutput(ri, linesBatch, records, tempFileLocation, JSON.stringify({
                                        input_path: tempFileLocation,
                                        output_path: outPath,
                                        isBatchMode: true,
                                        pluginsContext: JSON.stringify(window.pluginsContext),
                                        extraInfo: JSON.stringify(extraInfo),
                                        options: JSON.stringify(options)
                                    }))
                                }
                                window.batch_state.lineIndex += 1
                            }
                        } catch (e) {
                            console.log(e)
                            window.errorModal(`${window.i18n.SOMETHING_WENT_WRONG}:<br><br>`+e)
                            resolve()
                        }
                    }
                }
                window.batch_state.linesDoneSinceStart += linesBatch.length
                resolve()
            } else {
                linesBatch.forEach((lineRecord, li) => {
                    let tempFileLocation = lineRecord[4]
                    let outPath = lineRecord[5]
                    try {
                        fs.copyFileSync(tempFileLocation, outPath)
                        records[li][1].children[1].innerHTML = window.i18n.DONE
                        records[li][1].children[1].style.background = "green"

                        window.batch_state.lineIndex += 1

                        window.addActionButtons(records, li)

                    } catch (err) {
                        console.log(err)
                        window.appLogger.log(err)
                        window.errorModal(err.message)
                        batch_pauseBtn.click()
                    }
                    window.batch_state.linesDoneSinceStart += linesBatch.length
                    resolve()
                })
            }
        }).catch(async e => {
            if (e.code=="ECONNREFUSED" || e.code=="ECONNRESET") {
                await window.batchKickOffGeneration()
                resolve()
            } else {
                console.log(e)
                window.appLogger.log(e)
                batch_pauseBtn.click()
                if (document.getElementById("activeModal")) {
                    activeModal.remove()
                }
                console.log(e.message)
                window.errorModal(e.message).then(() => resolve())
            }
        })
    })
}

window.performSynthesis = async () => {

    if (batch_state.lineIndex-batch_state.fastModeActuallyFinishedTasks > window.userSettings.batch_fastModeMaxParallelizations) {
        console.log(`Ahead by ${batch_state.lineIndex-batch_state.fastModeActuallyFinishedTasks} tasks. Waiting...`)
        setTimeout(() => {window.performSynthesis()}, 1000)
        return
    }

    if (!window.batch_state.state) {
        return
    }

    if (window.batch_state.lineIndex==0) {
        const percentDone = (window.batch_state.lineIndex) / window.batch_state.lines.length * 100
        batch_progressBar.style.background = `linear-gradient(90deg, green ${parseInt(percentDone)}%, rgba(255,255,255,0) ${parseInt(percentDone)}%)`
        batch_progressBar.innerHTML = `${parseInt(percentDone* 100)/100}%`
        window.batch_state.taskBarPercent = percentDone/100
        window.electronBrowserWindow.setProgressBar(window.batch_state.taskBarPercent)
    }


    const record = window.batch_state.lines[window.batch_state.lineIndex]

    // Change the voice model if the next line uses a different one
    if (window.batch_state.lastModel!=record[0].voice_id) {
        await window.batchChangeVoice(record[0].game_id, record[0].voice_id, record[0].modelType)
        window.batch_state.lastModel = record[0].voice_id
    }

    // Change the vocoder if the next line uses a different one
    if (window.batch_state.lastVocoder!=record[0].vocoder && record[0].vocoder!="-") {
        await window.batchChangeVocoder(record[0].vocoder, record[0].game_id, record[0].voice_id)
    }

    await window.batchKickOffGeneration()

    if (window.batch_state.lineIndex==window.batch_state.lines.length) {
        // The end
        if (window.userSettings.batch_fastMode && false) { // No more fast modde. TODO, remove completely
            Promise.all(window.batch_state.fastModeOutputPromises).then(() => {
                window.stopBatch()
                batch_openDirBtn.style.display = "inline-block"
            })
        } else {
            window.stopBatch()
            batch_openDirBtn.style.display = "inline-block"
        }

    } else {
        window.performSynthesis()
    }
}

window.pauseResumeBatch = () => {

    batch_progressNotes.innerHTML = window.i18n.PAUSED

    const isRunning = window.batch_state.state
    batch_pauseBtn.innerHTML = isRunning ? window.i18n.RESUME : window.i18n.PAUSE
    window.batch_state.state = !isRunning

    window.electronBrowserWindow.setProgressBar(window.batch_state.taskBarPercent?window.batch_state.taskBarPercent:1, {mode: isRunning ? "paused" : "normal"})

    if (window.batch_state.state) {
        window.batch_state.startTime = new Date()
        window.batch_state.linesDoneSinceStart = 0
    }


    if (!isRunning) {
        window.performSynthesis()
    }
}

window.stopBatch = (stoppedByUser) => {
    window.electronBrowserWindow.setProgressBar(0)
    window.batch_state.state = false
    window.batch_state.lineIndex = 0

    batch_ETA_container.style.opacity = 0
    batch_synthesizeBtn.style.display = "inline-block"
    batch_clearBtn.style.display = "inline-block"
    batch_outputFolderInput.style.display = "inline-block"
    batch_clearDirOpts.style.display = "flex"
    batch_skipExistingOpts.style.display = "flex"
    batch_useSR.style.display = "flex"
    batch_useCleanup.style.display = "flex"
    batch_outputNumericallyOpts.style.display = "flex"
    batch_progressItems.style.display = "none"
    batch_progressBar.style.display = "none"
    batch_pauseBtn.style.display = "none"
    batch_stopBtn.style.display = "none"

    window.batch_state.lines.forEach(record => {
        if (record[1].children[1].innerHTML==window.i18n.READY || record[1].children[1].innerHTML==window.i18n.RUNNING) {
            record[1].children[1].innerHTML = window.i18n.STOPPED
            record[1].children[1].style.background = "none"
        }
    })

    const pluginData = {
        stoppedByUser: stoppedByUser
    }
    window.pluginsManager.runPlugins(window.pluginsManager.pluginsModules["batch-stop"]["post"], event="post batch-stop", pluginData)
}

window.adjustETA = () => {
    if (window.batch_state.state && window.batch_state.fastModeActuallyFinishedTasks>=2) {
        batch_ETA_container.style.opacity = 1

        // Lines per second
        const timeNow = new Date()
        const timeSinceStart = timeNow - window.batch_state.startTime
        const avgMSTimePerLine = timeSinceStart / window.batch_state.fastModeActuallyFinishedTasks
        batch_eta_lps.innerHTML = parseInt((1000/avgMSTimePerLine)*100)/100


        const remainingLines = window.batch_state.lines.length - window.batch_state.fastModeActuallyFinishedTasks
        let estTimeRemaining = avgMSTimePerLine*remainingLines

        // Estimated finish time
        const finishTime = new Date(timeNow.getTime() + estTimeRemaining)
        let etaFinishTime = `${finishTime.getHours()}:${String(finishTime.getMinutes()).padStart(2, "0")}:${String(finishTime.getSeconds()).padStart(2, "0")}`
        const days = [window.i18n.SUNDAY, window.i18n.MONDAY, window.i18n.TUESDAY, window.i18n.WEDNESDAY, window.i18n.THURSDAY, window.i18n.FRIDAY, window.i18n.SATURDAY]
        etaFinishTime = `${days[finishTime.getDay()]} ${etaFinishTime}`

        batch_eta_eta.innerHTML = etaFinishTime

        // Time remaining
        let etaTimeDisplay = []
        if (estTimeRemaining > (1000*60*60)) { // hours
            const hours = parseInt(estTimeRemaining/(1000*60*60))
            etaTimeDisplay.push(hours+"h")
            estTimeRemaining -= hours*(1000*60*60)
        }
        if (estTimeRemaining > (1000*60)) { // minutes
            const minutes = parseInt(estTimeRemaining/(1000*60))
            etaTimeDisplay.push(String(minutes).padStart(2, "0")+"m")
            estTimeRemaining -= minutes*(1000*60)
        }
        if (estTimeRemaining > (1000)) { // seconds
            const seconds = parseInt(estTimeRemaining/(1000))
            etaTimeDisplay.push(String(seconds).padStart(2, "0")+"s")
            estTimeRemaining -= seconds*(1000)
        }
        batch_eta_time.innerHTML = etaTimeDisplay.join(" ")

    } else {
        batch_ETA_container.style.opacity = 0
    }
}


const openOutput = () => {
    er.shell.showItemInFolder(window.userSettings.batchOutFolder+"/dummy.txt")
    spawn(`explorer`, [window.userSettings.batchOutFolder.replace(/\//g, "\\")], {stdio: "ignore"})
}


batch_paginationPrev.addEventListener("click", () => {
    batch_pageNum.value = Math.max(1, parseInt(batch_pageNum.value)-1)
    window.batch_state.paginationIndex = batch_pageNum.value-1
    window.refreshBatchRecordsList()
})
batch_paginationNext.addEventListener("click", () => {
    const numPages = Math.ceil(window.batch_state.lines.length/window.userSettings.batch_paginationSize)
    batch_pageNum.value = Math.min(parseInt(batch_pageNum.value)+1, numPages)
    window.batch_state.paginationIndex = batch_pageNum.value-1
    window.refreshBatchRecordsList()
})
batch_pageNum.addEventListener("change", () => {
    const numPages = Math.ceil(window.batch_state.lines.length/window.userSettings.batch_paginationSize)
    batch_pageNum.value = Math.max(1, Math.min(parseInt(batch_pageNum.value), numPages))
    window.batch_state.paginationIndex = batch_pageNum.value-1
    window.refreshBatchRecordsList()
})
setting_batch_paginationSize.addEventListener("change", () => {
    const numPages = Math.ceil(window.batch_state.lines.length/window.userSettings.batch_paginationSize)
    batch_pageNum.value = Math.max(1, Math.min(parseInt(batch_pageNum.value), numPages))
    window.batch_state.paginationIndex = batch_pageNum.value-1
    batch_total_pages.innerHTML = window.i18n.PAGINATION_TOTAL_OF.replace("_1", numPages)

    window.refreshBatchRecordsList()
})

window.toggleNumericalRecordsDisplay = () => {
    window.batch_state.lines.forEach(record => {
        record[1].children[6].innerHTML = batch_outputNumerically.checked ? `${window.userSettings.batchOutFolder}/${String(record[2]).padStart(10, '0')}` : record[0].out_path
    })
}
batch_outputNumerically.addEventListener("click", () => {
    window.toggleNumericalRecordsDisplay()
})
batch_saveToCSV.addEventListener("click", () => {

    try {
        const csv_file = [`game_id|voice_id|text`]
        window.batch_state.lines.forEach(line => {
            csv_file.push(`${line[0].game_id}|${line[0].voice_id}|${line[0].text}`)
        })

        const outFileName = JSON.stringify(new Date()).replace("\"","").replaceAll(":","_").split(".")[0]+"_batch.csv"

        fs.writeFileSync(`${window.userSettings.batchOutFolder}/${outFileName}`, csv_file.join("\n"), "utf8")

        window.createModal("error", `${window.i18n.BATCH_TOCSV_DONE}<br><br>${window.userSettings.batchOutFolder}/${outFileName}`)

    } catch(e) {
        console.log(e)
        window.appLogger.log(e.stack)
        window.errorModal(e.stack)
    }

})


batch_main.addEventListener("dragenter", event => window.uploadBatchCSVs("dragenter", event), false)
batch_main.addEventListener("dragleave", event => window.uploadBatchCSVs("dragleave", event), false)
batch_main.addEventListener("dragover", event => window.uploadBatchCSVs("dragover", event), false)
batch_main.addEventListener("drop", event => window.uploadBatchCSVs("drop", event), false)


batch_synthesizeBtn.addEventListener("click", window.startBatch)
batch_pauseBtn.addEventListener("click", window.pauseResumeBatch)
batch_stopBtn.addEventListener("click", () => window.stopBatch(true))
batch_openDirBtn.addEventListener("click", openOutput)