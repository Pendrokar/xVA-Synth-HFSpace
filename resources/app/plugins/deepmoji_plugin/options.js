"use strict"

const fs = require("fs")

let hasShownModal = false

const setupSettings = () => {
    const pathRoot = `${__dirname.replace(/\\/g,"/").replace("/javascript", "")}/`.replace("/resources/app/resources/app", "/resources/app")
    window.pluginsManager.registerINIFile("deepmoji_plugin", "deepmoji_plugin_settings", `${pathRoot}/deepmoji.ini`)

    if (hasShownModal) {
        return
    }

    try {
        fs.statSync(`${path}/plugins/deepmoji_plugin/DeepMoji/model/pytorch_model.bin`)
    } catch (e) {
        setTimeout(() => {
            window.errorModal(`The dependency file at the following location is not found, for the DeepMoji plugin:<br><br>${path}/plugins/deepmoji_plugin/DeepMoji/model/pytorch_model.bin<br><br>Be sure to download this file from the Nexus page linked on the plugin page.`)
        }, 1000)
    }

    hasShownModal = true
}

const setup = () => {
    setupSettings()
}
const teardown = () => {
    document.querySelectorAll(".deepmoji_plugin_plugin_setting").forEach(elem => elem.remove())
    hasShownModal = false
}
exports.setup = setup
exports.teardown = teardown
exports.setupSettings = setupSettings