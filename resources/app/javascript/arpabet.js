"use strict"

window.arpabetMenuState = {
    currentDict: undefined,
    dictionaries: {},
    paginationIndex: 0,
    totalPages: 0,
    clickedRecord: undefined,
    skipRefresh: false,
    hasInitialised: false,
    isRefreshing: false,
    hasChangedARPAbet: false
}

window.ARPAbetSymbols = ['AA0', 'AA1', 'AA2', 'AA', 'AE0', 'AE1', 'AE2', 'AE', 'AH0', 'AH1', 'AH2', 'AH',
  'AO0', 'AO1', 'AO2', 'AO', 'AW0', 'AW1', 'AW2', 'AW', 'AY0', 'AY1', 'AY2', 'AY',
  'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'EH', 'ER0', 'ER1', 'ER2', 'ER',
  'EY0', 'EY1', 'EY2', 'EY', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IH', 'IY0', 'IY1',
  'IY2', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OW', 'OY0',
  'OY1', 'OY2', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UH',
  'UW0', 'UW1', 'UW2', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', ,"}","{", "_",
  "AX", "AXR", "IX", "UX", "DX", "EL", "EM", "EN0", "EN1", "EN2", "EN", "NX", "Q", "WH",
  "RRR", "HR", "OE", "RH", "TS", "RR", "UU", "OO", "KH", "SJ", "HJ", "BR",

  "A1", "A2", "A3", "A4", "A5", "AI1", "AI2", "AI3", "AI4", "AI5", "AIR2", "AIR3", "AIR4", "AN1", "AN2", "AN3", "AN4", "AN5", "ANG1", "ANG2", "ANG3", "ANG4", "ANG5", "ANGR2", "ANGR3", "ANGR4", "ANR1", "ANR3", "ANR4", "AO1", "AO2", "AO3", "AO4", "AO5", "AOR1", "AOR2", "AOR3", "AOR4", "AOR5", "AR2", "AR3", "AR4", "AR5", "E1", "E2", "E3", "E4", "E5", "EI1", "EI2", "EI3", "EI4", "EI5", "EIR4", "EN1", "EN2", "EN3", "EN4", "EN5", "ENG1", "ENG2", "ENG3", "ENG4", "ENG5", "ENGR1", "ENGR4", "ENR1", "ENR2", "ENR3", "ENR4", "ENR5", "ER1", "ER2", "ER3", "ER4", "ER5", "I1", "I2", "I3", "I4", "I5", "IA1", "IA2", "IA3", "IA4", "IA5", "IAN1", "IAN2", "IAN3", "IAN4", "IAN5", "IANG1", "IANG2", "IANG3", "IANG4", "IANG5", "IANGR2", "IANR1", "IANR2", "IANR3", "IANR4", "IANR5", "IAO1", "IAO2", "IAO3", "IAO4", "IAO5", "IAOR1", "IAOR2", "IAOR3", "IAOR4", "IAR1", "IAR4", "IE1", "IE2", "IE3", "IE4", "IE5", "IN1", "IN2", "IN3", "IN4", "IN5", "ING1", "ING2", "ING3", "ING4", "ING5", "INGR2", "INGR4", "INR1", "INR4", "IONG1", "IONG2", "IONG3", "IONG4", "IONG5", "IR1", "IR3", "IR4", "IU1", "IU2", "IU3", "IU4", "IU5", "IUR1", "IUR2", "O1", "O2", "O3", "O4", "O5", "ONG1", "ONG2", "ONG3", "ONG4", "ONG5", "OR1", "OR2", "OU1", "OU2", "OU3", "OU4", "OU5", "OUR2", "OUR3", "OUR4", "OUR5", "U1", "U2", "U3", "U4", "U5", "UA1", "UA2", "UA3", "UA4", "UA5", "UAI1", "UAI2", "UAI3", "UAI4", "UAIR4", "UAIR5", "UAN1", "UAN2", "UAN3", "UAN4", "UAN5", "UANG1", "UANG2", "UANG3", "UANG4", "UANG5", "UANR1", "UANR2", "UANR3", "UANR4", "UAR1", "UAR2", "UAR4", "UE1", "UE2", "UE3", "UE4", "UE5", "UER2", "UER3", "UI1", "UI2", "UI3", "UI4", "UI5", "UIR1", "UIR2", "UIR3", "UIR4", "UN1", "UN2", "UN3", "UN4", "UN5", "UNR1", "UNR2", "UNR3", "UNR4", "UO1", "UO2", "UO3", "UO4", "UO5", "UOR1", "UOR2", "UOR3", "UOR5", "UR1", "UR2", "UR4", "UR5", "V2", "V3", "V4", "V5", "VE4", "VR3", "WA1", "WA2", "WA3", "WA4", "WA5", "WAI1", "WAI2", "WAI3", "WAI4", "WAN1", "WAN2", "WAN3", "WAN4", "WAN5", "WANG1", "WANG2", "WANG3", "WANG4", "WANG5", "WANGR2", "WANGR4", "WANR2", "WANR4", "WANR5", "WEI1", "WEI2", "WEI3", "WEI4", "WEI5", "WEIR1", "WEIR2", "WEIR3", "WEIR4", "WEIR5", "WEN1", "WEN2", "WEN3", "WEN4", "WEN5", "WENG1", "WENG2", "WENG3", "WENG4", "WENR2", "WO1", "WO2", "WO3", "WO4", "WO5", "WU1", "WU2", "WU3", "WU4", "WU5", "WUR3", "YA1", "YA2", "YA3", "YA4", "YA5", "YAN1", "YAN2", "YAN3", "YAN4", "YANG1", "YANG2", "YANG3", "YANG4", "YANG5", "YANGR4", "YANR3", "YAO1", "YAO2", "YAO3", "YAO4", "YAO5", "YE1", "YE2", "YE3", "YE4", "YE5", "YER4", "YI1", "YI2", "YI3", "YI4", "YI5", "YIN1", "YIN2", "YIN3", "YIN4", "YIN5", "YING1", "YING2", "YING3", "YING4", "YING5", "YINGR1", "YINGR2", "YINGR3", "YIR4", "YO1", "YO3", "YONG1", "YONG2", "YONG3", "YONG4", "YONG5", "YONGR3", "YOU1", "YOU2", "YOU3", "YOU4", "YOU5", "YOUR2", "YOUR3", "YOUR4", "YU1", "YU2", "YU3", "YU4", "YU5", "YUAN1", "YUAN2", "YUAN3", "YUAN4", "YUAN5", "YUANR2", "YUANR4", "YUE1", "YUE2", "YUE4", "YUE5", "YUER4", "YUN1", "YUN2", "YUN3", "YUN4",

  "@BREATHE_IN", "@BREATHE_OUT", "@LAUGH", "@GIGGLE", "@SIGH", "@COUGH", "@AHEM", "@SNEEZE", "@WHISTLE", "@UGH", "@HMM", "@GASP", "@AAH", "@GRUNT", "@YAWN", "@SNIFF"

  ]
// window.ARPAbetSymbols = [
//   'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
//   'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
//   'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
//   'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
//   'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
//   'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
//   'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
//   ,"}","{", "_"
// ]

window.refreshDictionariesList = () => {

    return new Promise(resolve => {

        // Don't spam with changes when the menu isn't open
        // if (arpabetModal.parentElement.style.display!="flex" && window.arpabetMenuState.hasInitialised) {
        if (arpabetModal.parentElement.style.display!="flex") {
            return
        }

        window.arpabetMenuState.hasInitialised = true
        if (window.arpabetMenuState.isRefreshing) {
            resolve()
            return
        }
        window.arpabetMenuState.isRefreshing = true

        if (window.arpabetMenuState.skipRefresh) {
            resolve()
            return
        }

        spinnerModal(window.i18n.LOADING_DICTIONARIES)
        window.arpabetMenuState.dictionaries = {}
        arpabet_dicts_list.innerHTML = ""

        const jsonFiles = fs.readdirSync(`${window.path}/arpabet`).filter(fname => fname.includes(".json"))

        const readFile = (fileCounter) => {

            const fname = jsonFiles[fileCounter]
            if (!fname.includes(".json")) {
                if ((fileCounter+1)<jsonFiles.length) {
                    readFile(fileCounter+1)
                } else {
                    window.arpabetRunSearch()
                    window.arpabetMenuState.isRefreshing = false
                    closeModal(undefined, arpabetContainer)
                    resolve()
                }
                return
            }
            const dictId = fname.replace(".json", "")

            fs.readFile(`${window.path}/arpabet/${fname}`, "utf8", (err, data) => {
                const jsonData = JSON.parse(data)

                const dictButton = createElem("button", jsonData.title)
                dictButton.title = jsonData.description
                dictButton.style.background = window.currentGame ? `#${window.currentGame.themeColourPrimary}` : "#aaa"
                arpabet_dicts_list.appendChild(dictButton)

                window.arpabetMenuState.dictionaries[dictId] = jsonData

                dictButton.addEventListener("click", ()=>handleDictClick(dictId))

                if ((fileCounter+1)<jsonFiles.length) {
                    readFile(fileCounter+1)
                } else {
                    window.arpabetRunSearch()
                    window.arpabetMenuState.isRefreshing = false
                    closeModal(undefined, arpabetContainer)
                    resolve()
                }
            })
        }
        if (jsonFiles.length) {
            readFile(0)
        } else {
            window.arpabetMenuState.isRefreshing = false
            closeModal(undefined, arpabetContainer)
            resolve()
        }
    })
}

window.handleDictClick = (dictId) => {

    if (window.arpabetMenuState.currentDict==dictId) {
        return
    }
    arpabet_enableall_button.disabled = false
    arpabet_disableall_button.disabled = false
    window.arpabetMenuState.currentDict = dictId
    window.arpabetMenuState.paginationIndex = 0
    window.arpabetMenuState.totalPages = 0

    arpabet_word_search_input.value = ""
    window.arpabetRunSearch()
    window.refreshDictWordList()
}

window.refreshDictWordList = () => {

    const dictId = window.arpabetMenuState.currentDict
    arpabetWordsListContainer.innerHTML = ""

    const wordKeys = Object.keys(window.arpabetMenuState.dictionaries[dictId].filteredData)
    let startIndex = window.arpabetMenuState.paginationIndex*window.userSettings.arpabet_paginationSize
    const endIndex = Math.min(startIndex+window.userSettings.arpabet_paginationSize, wordKeys.length)

    window.arpabetMenuState.totalPages = Math.ceil(wordKeys.length/window.userSettings.arpabet_paginationSize)
    arpabet_pagination_numbers.innerHTML = window.i18n.PAGINATION_X_OF_Y.replace("_1", window.arpabetMenuState.paginationIndex+1).replace("_2", window.arpabetMenuState.totalPages)

    for (let i=startIndex; i<endIndex; i++) {
        const data = window.arpabetMenuState.dictionaries[dictId].filteredData[wordKeys[i]]
        const word = wordKeys[i]

        const rowElem = createElem("div.arpabetRow")
        const ckbx = createElem("input.arpabetRowItem", {type: "checkbox"})
        ckbx.checked = data.enabled
        ckbx.style.marginTop = 0
        ckbx.addEventListener("click", () => {
            window.arpabetMenuState.dictionaries[dictId].data[wordKeys[i]].enabled = ckbx.checked
            window.arpabetMenuState.skipRefresh = true
            window.saveARPAbetDict(dictId)
            window.arpabetMenuState.hasChangedARPAbet = true
            setTimeout(() => window.arpabetMenuState.skipRefresh = false, 1000)
        })

        const deleteButton = createElem("button.smallButton.arpabetRowItem", window.i18n.DELETE)
        deleteButton.style.background = window.currentGame ? `#${window.currentGame.themeColourPrimary}` : "#aaa"
        deleteButton.addEventListener("click", () => {
            window.confirmModal(window.i18n.ARPABET_CONFIRM_DELETE_WORD.replace("_1", word)).then(response => {
                if (response) {
                    setTimeout(() => {
                        delete window.arpabetMenuState.dictionaries[dictId].data[word]
                        delete window.arpabetMenuState.dictionaries[dictId].filteredData[word]
                        window.saveARPAbetDict(dictId)
                        window.refreshDictWordList()
                    }, 210)
                }
            })
        })

        const wordElem = createElem("div.arpabetRowItem", word)
        wordElem.title = word

        const arpabetElem = createElem("div.arpabetRowItem", data.arpabet)
        arpabetElem.title = data.arpabet


        rowElem.appendChild(createElem("div.arpabetRowItem", ckbx))
        rowElem.appendChild(createElem("div.arpabetRowItem", deleteButton))
        rowElem.appendChild(wordElem)
        rowElem.appendChild(arpabetElem)

        rowElem.addEventListener("click", () => {
            window.arpabetMenuState.clickedRecord = {elem: rowElem, word}
            arpabet_word_input.value = word
            arpabet_arpabet_input.value = data.arpabet
        })

        arpabetWordsListContainer.appendChild(rowElem)
    }
}

window.saveARPAbetDict = (dictId) => {

    const dataOut = {
        title: window.arpabetMenuState.dictionaries[dictId].title,
        description: window.arpabetMenuState.dictionaries[dictId].description,
        version: window.arpabetMenuState.dictionaries[dictId].version,
        author: window.arpabetMenuState.dictionaries[dictId].author,
        nexusLink: window.arpabetMenuState.dictionaries[dictId].nexusLink,
        data: window.arpabetMenuState.dictionaries[dictId].data
    }

    doFetch(`http://localhost:8008/updateARPABet`, {
        method: "Post",
        body: JSON.stringify({})
    })//.then(r => r.text()).then(r => {console.log(r)})


    fs.writeFileSync(`${window.path}/arpabet/${dictId}.json`, JSON.stringify(dataOut, null, 4))
}




window.refreshARPABETReferenceDisplay = () => {

    const V2 = [2]
    const V3 = [3]
    const V2_3 = [2,3]

    const data = [
        // Min model version, symbols, examples
        [V2, "AA0, AA1, AA2", "b<b>al</b>m, b<b>o</b>t, c<b>o</b>t"],
        [V3, "AA, AA0, AA1, AA2", "b<b>al</b>m, b<b>o</b>t, c<b>o</b>t"],

        [V2, "AE0, AE1, AE2", "b<b>a</b>t, f<b>a</b>st"],
        [V3, "AE, AE0, AE1, AE2", "b<b>a</b>t, f<b>a</b>st"],

        [V2, "AH0, AH1, AH2", "b<b>u</b>tt"],
        [V3, "AH, AH0, AH1, AH2", "b<b>u</b>tt"],

        [V2, "AO0, AO1, AO2", "st<b>o</b>ry"],
        [V3, "AO, AO0, AO1, AO2", "st<b>o</b>ry"],

        [V2, "AW0, AW1, AW2", "b<b>ou</b>t"],
        [V3, "AW, AW0, AW1, AW2", "b<b>ou</b>t"],

        [V3, "AX", "comm<b>a</b>"],
        [V3, "AXR", "lett<b>er</b>"],

        [V2, "AY0, AY1, AY2", "b<b>i</b>te"],
        [V3, "AY, AY0, AY1, AY2", "b<b>i</b>te"],

        [V2_3, "B", "<b>b</b>uy"],

        [V3, "BR", "B and RRR sounds, together"],

        [V2_3, "CH", "<b>ch</b>ina"],
        [V2_3, "D", "<b>d</b>ie"],
        [V3, "DX", "bu<b>tt</b>er"],
        [V2_3, "DH", "<b>th</b>y"],
        [V2, "EH0, EH1, EH2", "b<b>e</b>t"],
        [V3, "EH,EH0, EH1, EH2", "b<b>e</b>t"],

        [V3, "EL", "bott<b>le</b>"],
        [V3, "EM", "rhyth<b>m</b>"],
        [V3, "EN, EN0, EN1, EN2", "butt<b>on</b>"],

        [V2, "ER0, ER1, ER2", "b<b>i</b>rd"],
        [V3, "ER, ER0, ER1, ER2", "b<b>i</b>rd"],

        [V2, "EY0, EY1, EY2", "b<b>ai</b>t"],
        [V3, "EY, EY0, EY1, EY2", "b<b>ai</b>t"],

        [V2_3, "F", "<b>f</b>ight"],
        [V2_3, "G", "<b>g</b>uy"],
        [V2_3, "HH", "<b>h</b>igh"],
        [V3, "HJ", "J sound if mouth was open"],
        [V3, "HR", "hrr sound typical in Arabic"],
        [V2, "IH0, IH1, IH2", "b<b>i</b>t"],
        [V3, "IH, IH0, IH1, IH2", "b<b>i</b>t"],
        [V3, "IX", "ros<b>e</b>s, rabb<b>i</b>t"],

        [V2, "IY0, IY1, IY2", "b<b>ea</b>t"],
        [V3, "IY, IY0, IY1, IY2", "b<b>ea</b>t"],

        [V2_3, "JH", "<b>j</b>ive"],
        [V2_3, "K", "<b>k</b>ite"],
        [V3, "KH", "K and H sounds, but together"],
        [V2_3, "L", "<b>l</b>ie"],
        [V2_3, "M", "<b>m</b>y"],
        [V2_3, "N", "<b>n</b>igh"],
        [V2_3, "NG", "si<b>ng</b>"],
        [V3, "NX", "wi<b>nn</b>er"],

        [V3, "OE", "german m<b>ö</b>ve, french eu (bl<b>eu</b>)"],
        [V3, "OO", "hard o sound"],
        [V2, "OW0, OW1, OW2", "b<b>oa</b>t"],
        [V3, "OW, OW0, OW1, OW2", "b<b>oa</b>t"],

        [V2, "OY0, OY1, OY2", "b<b>oy</b>"],
        [V3, "OY, OY0, OY1, OY2", "b<b>oy</b>"],

        [V2_3, "P", "<b>p</b>ie"],
        [V3, "Q", "(glottal stop) uh<b>-</b>oh)"],
        [V2_3, "R", "<b>r</b>ye"],
        [V3, "RH, RR", "<b>r</b>un"],
        [V3, "RRR", "<strong r>"],
        [V2_3, "S", "<b>s</b>igh"],
        [V3, "SJ", "swedish sj"],
        [V2_3, "SH", "<b>sh</b>y"],
        [V2_3, "T", "<b>t</b>ie"],
        [V2_3, "TH", "<b>th</b>igh"],
        [V3, "TS", "T and S sounds together (eg romanian ț)"],
        [V2, "UH0, UH1, UH2", "b<b>oo</b>k"],
        [V3, "UH, UH0, UH1, UH2", "b<b>oo</b>k"],
        [V3, "UU", "german <b>ü</b>ber"],

        [V2, "UW0, UW1, UW2", "b<b>oo</b>t"],
        [V3, "UW, UW0, UW1, UW2", "b<b>oo</b>t"],
        [V3, "UX", "d<b>u</b>de"],
        [V3, "WH", "<b>wh</b>at, <b>wh</b>y (w with 'h' sound)"],

        [V2_3, "V", "<b>v</b>ie"],
        [V2_3, "W", "<b>w</b>ise"],
        [V2_3, "Y", "<b>y</b>acht"],
        [V2_3, "Z", "<b>z</b>oo"],
        [V2_3, "ZH", "plea<b>s</b>ure"],
    ]
    arpabetReferenceList.innerHTML = ""
    data.forEach(item => {
        if (item[0].includes(parseInt(arpabetMenuModelDropdown.value))) {
            const div = createElem("div")
            div.appendChild(createElem("div", item[1]))

            const exampleDiv = createElem("div")
            exampleDiv.appendChild(createElem("div",item[2]))
            div.appendChild(exampleDiv)

            arpabetReferenceList.appendChild(div)
        }
    })
}
arpabetMenuModelDropdown.value = "3"
window.refreshARPABETReferenceDisplay()
arpabetMenuModelDropdown.addEventListener("click", window.refreshARPABETReferenceDisplay)


arpabet_save.addEventListener("click", () => {
    const word = arpabet_word_input.value.trim().toLowerCase()
    const arpabet = arpabet_arpabet_input.value.trim().toUpperCase().replace(/\s{2,}/g, " ")

    if (!word.length || !arpabet.length) {
        return window.errorModal(window.i18n.ARPABET_ERROR_EMPTY_INPUT)
    }

    const badSymbols = arpabet.split(" ").filter(symb => !window.ARPAbetSymbols.includes(symb))
    if (badSymbols.length) {
        return window.errorModal(window.i18n.ARPABET_ERROR_BAD_SYMBOLS.replace("_1", badSymbols.join(", ")))
    }

    const wordKeys = Object.keys(window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data)

    const doTheRest_updateDict = () => {
        window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data[word] = {enabled: true, arpabet: arpabet}
        window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].filteredData = window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data

        window.refreshDictWordList()
        window.saveARPAbetDict(window.arpabetMenuState.currentDict)
    }


    // Delete the old record
    if (window.arpabetMenuState.clickedRecord && window.arpabetMenuState.clickedRecord.word != word) {
        delete window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data[window.arpabetMenuState.clickedRecord.word]
    }

    let wordExists = []
    Object.keys(window.arpabetMenuState.dictionaries).forEach(dictName => {
        if (dictName==window.arpabetMenuState.currentDict) {
            return
        }

        if (Object.keys(window.arpabetMenuState.dictionaries[dictName].data).includes(word)) {
            wordExists.push(dictName)
        }
    })

    if (wordExists.length) {
        window.confirmModal(window.i18n.ARPABET_CONFIRM_SAME_WORD.replace("_1", word).replace("_2", wordExists.join("<br>"))).then(response => {
            if (response) {
                doTheRest_updateDict()
            }
        })
    } else {
        doTheRest_updateDict()
    }
})

arpabetModal.addEventListener("click", (event) => {
    if (window.arpabetMenuState.clickedRecord && event.target.className!="arpabetRow"&& event.target.className!="arpabetRowItem" && ![arpabet_word_input, arpabet_arpabet_input, arpabet_save, arpabet_prev_btn, arpabet_next_btn].includes(event.target)) {
        window.arpabetMenuState.clickedRecord = undefined
        arpabet_word_input.value = ""
        arpabet_arpabet_input.value = ""
    }
})
arpabet_prev_btn.addEventListener("click", () => {
    window.arpabetMenuState.paginationIndex = Math.max(0, window.arpabetMenuState.paginationIndex-1)
    window.refreshDictWordList()
})
arpabet_next_btn.addEventListener("click", () => {
    window.arpabetMenuState.paginationIndex = Math.min(window.arpabetMenuState.totalPages-1, window.arpabetMenuState.paginationIndex+1)
    window.refreshDictWordList()
})

window.arpabetRunSearch = () => {
    if (!window.arpabetMenuState.currentDict) {
        return
    }
    window.arpabetMenuState.paginationIndex = 0
    window.arpabetMenuState.totalPages = 0

    let query = arpabet_word_search_input.value.trim().toLowerCase()

    if (!query.length) {
        if (arpabet_search_only_enabled.checked) {
            const filteredKeys = Object.keys(window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data).filter(key => window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data[key].enabled)
            window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].filteredData = {}
            filteredKeys.forEach(key => {
                window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].filteredData[key] = window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data[key]
            })
        } else {
            window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].filteredData = window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data
        }
    } else {
        const strictQuery = query.startsWith("\"")
        const filteredKeys = Object.keys(window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data)
        .filter(key => {
            if (strictQuery) {
                query = query.replaceAll("\"", "")
                return key==query && (arpabet_search_only_enabled.checked ? (window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data[key].enabled) : true)
            } else {
                return key.includes(query) && (arpabet_search_only_enabled.checked ? (window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data[key].enabled) : true)
            }
        })

        window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].filteredData = {}
        filteredKeys.forEach(key => {
            window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].filteredData[key] = window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data[key]
        })
    }

    window.refreshDictWordList()
}
let arpabetSearchInterval
arpabet_word_search_input.addEventListener("keyup", () => {
    if (arpabetSearchInterval!=null) {
        clearTimeout(arpabetSearchInterval)
    }
    arpabetSearchInterval = setTimeout(window.arpabetRunSearch, 500)
})

arpabet_enableall_button.addEventListener("click", () => {

    const dictName = window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].title
    window.confirmModal(window.i18n.ARPABET_CONFIRM_ENABLE_ALL.replace("_1", dictName)).then(response => {
        if (response) {
            setTimeout(() => {
                const wordKeys = Object.keys(window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data)
                wordKeys.forEach(word => {
                    window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data[word].enabled = true
                })

                window.saveARPAbetDict(window.arpabetMenuState.currentDict)
                window.arpabetRunSearch()
            }, 210)
        }
    })
})

arpabet_disableall_button.addEventListener("click", () => {

    const dictName = window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].title
    window.confirmModal(window.i18n.ARPABET_CONFIRM_DISABLE_ALL.replace("_1", dictName)).then(response => {
        if (response) {
            setTimeout(() => {
                const wordKeys = Object.keys(window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data)
                wordKeys.forEach(word => {
                    window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data[word].enabled = false
                })

                window.saveARPAbetDict(window.arpabetMenuState.currentDict)
                window.arpabetRunSearch()
            }, 210)
        }
    })
})
arpabet_search_only_enabled.addEventListener("click", () => window.arpabetRunSearch())



fs.watch(`${window.path}/arpabet`, {recursive: false, persistent: true}, (eventType, filename) => {console.log(eventType, filename);refreshDictionariesList()})