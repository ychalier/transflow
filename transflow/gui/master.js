const VERSION = "2025w34a";
const VIDEO_FILETYPES = "*.mp4 *.avi *.mkv *.mov *.mpg";
const IMAGE_FILETYPES = "*.jpg *.jpeg *.png";

var config = {
    version: VERSION,
    seed: Math.floor(Math.random() * (Math.pow(2, 31) - 1)),
    previewUrl: null,
    outputUrl: null,
    isRunning: false,
    flowSource: {
        file: null,
        direction: "backward",
        maskPath: null,
        kernelPath: null,
        cvConfig: null,
        flowFilters: null,
        useMvs: false,
        roundFlow: false,
        exportFlow: false,
        seekTime: null,
        durationTime: null,
        repeat: 1,
        lockMode: "stay",
        lockExpr: null,
    },
    compositor: {
        layerCount: 1,
        layers: [{
            classname: "moveref",
            maskAlpha: null,
            maskSource: null,
            maskDestination: null,
            flagMoveTransparent: false,
            flagMoveToEmpty: true,
            flagMoveToFilled: true,
            flagLeaveEmpty: false,
            maskIntroduction: null,
            introduceEmpty: true,
            introduceFilled: true,
            introduceMoving: true,
            introduceUnmoving: true,
            introduceOnce: false,
            introduceAllEmpty: false,
            introduceAllFilled: false,
            resetMode: "off",
            maskReset: null,
            resetRandomFactor: 0.1,
            resetConstantStep: 1,
            resetLinearFactor: 0.1,
            sourceCount: 1,
            sources: [{
                file: null,
                type: "bwnoise",
                color: "#cff010",
                alterationPath: null,
                seekTime: null,
                repeat: 1, 
            }]
        }],
        backgroundColor: "#ffffff"
    },
    output: {
        file: null,
        outputIntensity: false,
        outputHeatmap: false,
        outputAccumulator: false,
        renderScale: 0.1,
        renderColors: null,
        renderBinary: false,
        checkpointEvery: null,
        checkpointEnd: false,
        vcodec: "h264",
    }
};

const MAX_LAYERS_AND_SOURCES = 5;

for (let i = 0; i <= MAX_LAYERS_AND_SOURCES; i++) {
    config.compositor.layers[0].sources.push(JSON.parse(JSON.stringify(config.compositor.layers[0].sources[0])));
}

for (let i = 0; i <= MAX_LAYERS_AND_SOURCES; i++) {
    config.compositor.layers.push(JSON.parse(JSON.stringify(config.compositor.layers[0])));
}

const STORAGE_KEY = "transflow-gui";

function loadConfigFromStorage() {
    if (localStorage.getItem(STORAGE_KEY) != null) {
        function mergeConfig(configObj, storageObj) {
            for (const key in storageObj) {
                if (typeof(storageObj[key]) == "object") {
                    if (!(key in configObj)) configObj[key] = {};
                    mergeConfig(configObj[key], storageObj[key]);
                } else {
                    configObj[key] = storageObj[key];
                }
            }
        }
        const storageConfig = JSON.parse(localStorage.getItem(STORAGE_KEY));
        if (storageConfig.version != VERSION) return;
        mergeConfig(config, storageConfig);
    }
}
loadConfigFromStorage();
config.isRunning = false;

function saveConfigToStorage() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
}

var leftPanelActiveTab = "Flow Source";
var websocket;
var websocketRetryCount = 0;
var leftPanel;
var rightPanel;
var wssConnectionIndicator;
var overlayInterval;

function create(parent=null, tag="div", className=null) {
    const element = document.createElement(tag);
    if (parent != null) parent.appendChild(element);
    if (className != null) element.className = className;
    return element;
}

function remove(element) {
    element.parentElement.removeChild(element);
}

function getSelectedValue(select) {
    for (const option of select.querySelectorAll("option")) {
        if (option.selected) {
            return option.value;
        }
    }
}

function getPathSuffix(path) {
    const split = path.split(".");
    return "." + split[split.length - 1];
}

function getPathName(path) {
    const separator = path.includes("/") ? "/" : "\\";
    const split = path.split(separator);
    return split[split.length - 1];
}

function createInputContainer(container, label) {
    const inputContainer = create(container, "div", "input-container");
    create(inputContainer, "label").textContent = label;
    return inputContainer;
}

function configGet(key) { //TODO: Support arrays
    const split = key.split(".");
    let o = config;
    for (let i = 0; i < split.length - 1; i++) {
        o = o[split[i]];
    }
    return o[split[split.length - 1]];
}

function configSet(key, value) { //TODO: Support arrays
    const split = key.split(".");
    let o = config;
    for (let i = 0; i < split.length - 1; i++) {
        o = o[split[i]];
    }
    const oldValue = o[split[split.length - 1]];
    if (oldValue != value) {
        o[split[split.length - 1]] = value;
        onConfigChange(key);
    }
}

function onConfigChange(key) {
    console.log("Config changed", key, configGet(key));
    saveConfigToStorage();
}

function createFileOpenInput(container, label, key, filetypes) {
    const inputContainer = createInputContainer(container, label);
    const inputButtons = create(inputContainer, "div", "row");
    const input = create(inputButtons, "button");
    input.textContent = "Select file";
    input.addEventListener("click", () => {
        const data = { key: key, filetypes: filetypes };
        websocket.send(`FILE_OPEN ${JSON.stringify(data)}`);
    });
    const fileUrl = configGet(key);
    if (fileUrl != null) {
        const ext = getPathSuffix(fileUrl);
        if (VIDEO_FILETYPES.includes(ext)) {
            const video = create(inputContainer, "video");
            video.src = `/media?url=${fileUrl}`;
            video.setAttribute("controls", 1);
        } else if (IMAGE_FILETYPES.includes(ext)) {
            const img = create(inputContainer, "img");
            img.src = `/media?url=${fileUrl}`;
        }
        input.textContent = getPathName(fileUrl);
        const buttonClear = create(inputButtons, "button");
        buttonClear.textContent = "Clear";
        buttonClear.addEventListener("click", () => {
            configSet(key, null);
            inflateLeftPanel(leftPanel);
        });
    }
}

function createFileSaveInput(container, label, key, defaultextension, filetypes) {
    const inputContainer = createInputContainer(container, label);
    const inputButtons = create(inputContainer, "div", "row");
    const input = create(inputButtons, "button");
    input.textContent = "Select file";
    input.addEventListener("click", () => {
        const data = { key: key, defaultextension: defaultextension , filetypes: filetypes};
        websocket.send(`FILE_SAVE ${JSON.stringify(data)}`);
    });
    const fileUrl = configGet(key);
    if (fileUrl != null) {
        input.textContent = getPathName(fileUrl);
        const buttonClear = create(inputButtons, "button");
        buttonClear.textContent = "Clear";
        buttonClear.addEventListener("click", () => {
            configSet(key, null);
            inflateLeftPanel(leftPanel);
        });
    }
}

function createBoolInput(container, label, key) {
    const inputContainer = createInputContainer(container, label);
    inputContainer.classList.add("input-container-bool");
    const input = create(inputContainer, "input");
    input.type = "checkbox";
    input.id = `input-${key}`;
    inputContainer.querySelector("label").setAttribute("for", `input-${key}`);
    if (configGet(key)) input.checked = true;
    input.addEventListener("change", () => {
        configSet(key, input.checked);
    });
}

function inflateSelect(select, values, initialValue) {
    select.innerHTML = "";
    for (const value of values) {
        const option = create(select, "option");
        option.value = value;
        option.textContent = value;
        if (value == initialValue) option.selected = true;
    }
}

function createSelect(container, label, key, values) {
    const selectContainer = createInputContainer(container, label);
    const select = create(selectContainer, "select");
    inflateSelect(select, values, configGet(key));
    select.addEventListener("change", () => {
        configSet(key, getSelectedValue(select));
    });
    return select;
}

function createTextInput(container, label, key, placeholder=undefined) {
    const inputContainer = createInputContainer(container, label);
    const input = create(inputContainer, "input");
    input.value = configGet(key);
    if (placeholder != undefined) {
        input.placeholder = placeholder;
    }
    input.addEventListener("change", () => {
        let value = input.value.trim();
        if (value == "") value = null;
        configSet(key, value);
    });
}

const createTimestampInput = createTextInput; //TODO: custom input

function createColorInput(container, label, key) {
    const inputContainer = createInputContainer(container, label);
    const input = create(inputContainer, "input");
    input.type = "color";
    input.value = configGet(key);
    input.addEventListener("change", () => {
        if (input.value != undefined) configSet(key, input.value);
    });
}

function createRangeInput(container, label, key, min=0, max=1, decimals=3) {
    const inputContainer = createInputContainer(container, label);
    const input = create(inputContainer, "input");
    input.type = "range";
    input.min = min;
    input.max = max;
    input.step = 1 / Math.pow(10, decimals);
    input.value = configGet(key);
    const valueLabel = create(inputContainer, "span");
    valueLabel.textContent = configGet(key).toFixed(decimals);
    input.addEventListener("input", () => {
        valueLabel.textContent = parseFloat(input.value).toFixed(decimals);
    });
    input.addEventListener("change", () => {
        configSet(key, parseFloat(input.value));
    });
}

function createNumberInput(container, label, key, min=null, max=null, step=1) {
    const inputContainer = createInputContainer(container, label);
    const input = create(inputContainer, "input");
    input.type = "number";
    if (min != null) input.min = min;
    if (max != null) input.max = max;
    input.step = step;
    input.value = configGet(key);
    input.addEventListener("change", () => {
        let value = input.value;
        value = value == "" ? null : parseFloat(value);
        configSet(key, value);
    });
    return input;
}

function createAddButton(container, label, key) {
    const button = create(create(container, "div"), "button", "button-input");
    button.textContent = label;
    button.addEventListener("click", () => {
        let value = Math.min(configGet(key) + 1, MAX_LAYERS_AND_SOURCES);
        configSet(key, value);
        inflateLeftPanel(leftPanel);
    });
}

function createDeleteButton(container, label, countKey, arrayKey, arrayIndex) {
    const button = create(create(container, "div"), "button");
    button.textContent = label;
    button.addEventListener("click", () => {
        let value = configGet(countKey);
        for (let i = arrayIndex; i < MAX_LAYERS_AND_SOURCES; i++) {
            configSet(`${arrayKey}.${i}`, JSON.parse(JSON.stringify(configGet(`${arrayKey}.${i+1}`))));
        }
        value = Math.max(0, value - 1);
        configSet(countKey, value);
        inflateLeftPanel(leftPanel);
    });
}

function createMoveButton(container, label, countKey, arrayKey, arrayIndex) {
    const button = create(create(container, "div"), "button");
    button.textContent = label;
    button.addEventListener("click", () => {
        const count = configGet(countKey);
        const answer = prompt(`Move to?\nSpecify index between 0 and ${count-1}`, arrayIndex);
        if (answer == null) return;
        const newIndex = parseInt(answer);
        if (newIndex < 0 || newIndex >= count) {
            alert(`Index out of bounds.\nIndex must be between 0 and ${count-1}`);
            return;
        }
        const aux = JSON.parse(JSON.stringify(configGet(`${arrayKey}.${newIndex}`)));
        configSet(`${arrayKey}.${newIndex}`, configGet(`${arrayKey}.${arrayIndex}`));
        configSet(`${arrayKey}.${arrayIndex}`, aux);
        inflateLeftPanel(leftPanel);
    });
}

function setWssConnectionIndicator(state) {
    if (wssConnectionIndicator == undefined) return;
    wssConnectionIndicator.textContent = state;
}

function formatDuration(totalSeconds) {
    let s = totalSeconds;
    const hours = Math.floor(s / 3600);
    s -= hours * 3600;
    const minutes = Math.floor(s / 60);
    s -= minutes * 60;
    const seconds = Math.floor(s);
    if (hours == 0) {
        return `${minutes}:${seconds.toString().padStart(2, "0")}`;
    }
    return `${hours}:${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
}

function connectWebsocket(wssUrl) {
    setWssConnectionIndicator(`connecting… [${websocketRetryCount}]`);
    websocket = new WebSocket(wssUrl);
    websocket.onopen = () => {
        websocketRetryCount = 0;
        console.log("Websocket is connected");
        setWssConnectionIndicator("connected");
        const overlay = document.querySelector(".overlay");
        if (overlay != null) {
            clearInterval(overlayInterval);
            remove(overlay);
        }
        websocket.send("RELOAD");
    }
    websocket.onmessage = (message) => {
        console.log("WS>", message.data);
        if (message.data.startsWith("FILE ")) {
            const query = message.data.slice(5);
            let [key, fileUrl] = query.split(" ", 2);
            if (fileUrl == "") fileUrl = null;
            configSet(key, fileUrl);
            inflateLeftPanel(leftPanel);
        } else if (message.data.startsWith("PREVIEW")) {
            config.isRunning = true;
            config.previewUrl = message.data.slice(8);
            onConfigChange("previewUrl");
            inflateRightPanel(rightPanel);
        } else if (message.data.startsWith("DONE")) {
            config.isRunning = false;
            document.getElementById("button-generate").removeAttribute("disabled");
            document.getElementById("button-interrupt").setAttribute("disabled", true);
            const query = message.data.slice(5);
            if (query) {
                config.outputUrl = query;
                onConfigChange("outputUrl");
                inflateRightPanel(rightPanel);
            }
        } else if (message.data.startsWith("CANCEL")) {
            config.isRunning = false;
            document.getElementById("button-generate").removeAttribute("disabled");
            document.getElementById("button-interrupt").setAttribute("disabled", true);
        } else if (message.data.startsWith("STATUS")) {
            const progressBarContainer = document.getElementById("progress-bar");
            if (progressBarContainer == null) return;
            const status = JSON.parse(message.data.slice(7));
            const progress = progressBarContainer.querySelector("progress");
            const progressInfo = progressBarContainer.querySelector(".progress-info");
            progress.min = 0;
            progress.max = status.total;
            progress.value = status.cursor;
            if (status.error != null) {
                progressInfo.textContent = `Error: ${status.error}`;
            } else {
                const rate = status.cursor / status.elapsed;
                const remaining = (status.total - status.cursor) / rate;
                progressInfo.textContent = `${(100*status.cursor/status.total).toFixed(0)}% ${status.cursor}/${status.total} [${formatDuration(status.elapsed)}<${formatDuration(remaining)}, ${rate.toFixed(2)}frame/s]`
            }
        } else if (message.data.startsWith("RELOAD")) {
            const data = JSON.parse(message.data.slice(7));
            config.isRunning = data.ongoing;
            config.outputUrl = data.outputFile;
            config.previewUrl = data.previewUrl;
            inflateRightPanel(rightPanel);
        }
        websocket.send("PONG");
    };
    websocket.onclose = (event) => {
        websocketRetryCount++;
        let delay = 0.01;
        if (websocketRetryCount >= 10) {
            delay = 5;
        } else if (websocketRetryCount >= 3) {
            delay = 1;
        }
        console.log(`Socket is closed. Reconnect will be attempted in ${delay} second.`, event.reason);
        setWssConnectionIndicator("closed");
        setTimeout(() => { connectWebsocket(wssUrl); }, delay * 1000);
    };
    websocket.onerror = (err) => {
        console.error("Socket encountered error: ", err.message, "Closing socket");
        setWssConnectionIndicator("error");
        websocket.close();
    };
}

function openWssConnection() {
    setWssConnectionIndicator("scanning…");
    fetch("/wss").then(res => res.text()).then(text => {
        const wssUrl = text;  // `ws://${res.headers.get("Wss-Host")}:${res.headers.get("Wss-Port")}`;
        connectWebsocket(wssUrl);
    });
}

function inflatePaneFlowSource(container) {
    container.innerHTML = "";
    createFileOpenInput(container, "File", "flowSource.file", VIDEO_FILETYPES);
    createBoolInput(container, "Use Motion Vectors", "flowSource.useMvs");
    createSelect(container, "Direction", "flowSource.direction", ["backward", "forward"]);
    createFileOpenInput(container, "Mask", "flowSource.maskPath", IMAGE_FILETYPES);
    createFileOpenInput(container, "Kernel", "flowSource.kernelPath", "*.npy");
    createFileOpenInput(container, "CV Config", "flowSource.cvConfig", "*.json");
    createTextInput(container, "Filters", "flowSource.flowFilters", "scale=2;clip=5;polar=r:a");
    createBoolInput(container, "Round Flow", "flowSource.roundFlow");
    createBoolInput(container, "Export Flow", "flowSource.exportFlow");
    createTimestampInput(container, "Seek Time", "flowSource.seekTime", "00:00:00");
    createTimestampInput(container, "Duration Time", "flowSource.durationTime", "00:00:00");
    createNumberInput(container, "Repeat", "flowSource.repeat", 0, null, 1);
    createSelect(container, "Lock Mode", "flowSource.lockMode", ["stay", "skip"]);
    createTextInput(container, "Lock Expression", "flowSource.lockExpr", "(1,1),(4,1) (stay) t>=2 and t<=3 (skip)");
}

function inflatePaneCompositor(container) {
    container.innerHTML = "";
    createColorInput(container, "Background Color", "compositor.backgroundColor");
    const layersContainer = create(container, "div", "layers");
    for (let i = 0; i < config.compositor.layerCount; i++) {
        const layerContainer = create(layersContainer, "div", "layer");
        const classnameSelect = createSelect(layerContainer, "Method", `compositor.layers.${i}.classname`, ["moveref", "introduction", "sum", "static"]);
        const layerInputs = create(layerContainer, "div", "input-container");
        function onClassnameChange() {
            const value = getSelectedValue(classnameSelect);
            layerInputs.innerHTML = "";
            createFileOpenInput(layerInputs, "Mask Alpha", `compositor.layers.${i}.maskAlpha`, IMAGE_FILETYPES);
            if (value == "moveref" || value == "introduction") {
                const details = create(layerInputs, "details");
                create(details, "summary").textContent = "Movement Options";
                createFileOpenInput(details, "Source Mask", `compositor.layers.${i}.maskSource`, IMAGE_FILETYPES);
                createFileOpenInput(details, "Destination Mask", `compositor.layers.${i}.maskDestination`, IMAGE_FILETYPES);
                createBoolInput(details, "Transparent Pixels Can Move", `compositor.layers.${i}.flagMoveTransparent`);
                createBoolInput(details, "Pixels Can Move to Empty Spots", `compositor.layers.${i}.flagMoveToEmpty`);
                createBoolInput(details, "Pixels Can Move to Filled Spots", `compositor.layers.${i}.flagMoveToFilled`);
                createBoolInput(details, "Moving Pixels Leave an Empty Spot", `compositor.layers.${i}.flagLeaveEmpty`);
            }
            if (value == "introduction") {
                const details = create(layerInputs, "details");
                create(details, "summary").textContent = "Introduction Options";
                createFileOpenInput(details, "Introduction Mask", `compositor.layers.${i}.maskIntroduction`, IMAGE_FILETYPES);
                createBoolInput(details, "Introduce Pixels on Empty Spots", `compositor.layers.${i}.introduceEmpty`);
                createBoolInput(details, "Introduce Pixels on Filled Spots", `compositor.layers.${i}.introduceFilled`);
                createBoolInput(details, "Introduce Moving Pixels", `compositor.layers.${i}.introduceMoving`);
                createBoolInput(details, "Introduce Unmoving Pixels", `compositor.layers.${i}.introduceUnmoving`);
                createBoolInput(details, "Introduce Only Once", `compositor.layers.${i}.introduceOnce`);
                createBoolInput(details, "Introduce On All Empty Spots", `compositor.layers.${i}.introduceAllEmpty`);
                createBoolInput(details, "Introduce On All Filled Spots", `compositor.layers.${i}.introduceAllFilled`);
            }
            if (value == "moveref" || value == "sum") {
                const details = create(layerInputs, "details");
                create(details, "summary").textContent = "Reset Options";
                createFileOpenInput(details, "Reset Mask", `compositor.layers.${i}.maskReset`, IMAGE_FILETYPES);
                const resetSelect = createSelect(details, "Reset Mode", `compositor.layers.${i}.resetMode`, ["off", "random", "constant", "linear"]);
                const resetInputs = create(details, "div", "input-container");
                function onResetChange() {
                    const value = getSelectedValue(resetSelect);
                    resetInputs.innerHTML = "";
                    if (value == "random") {
                        createRangeInput(resetInputs, "Random Reset Factor", `compositor.layers.${i}.resetRandomFactor`);
                    }
                    if (value == "constant") {
                        createNumberInput(resetInputs, "Constant Reset Step", `compositor.layers.${i}.resetConstantStep`, 0, null, 1);
                    }
                    if (value == "linear") {
                        createRangeInput(resetInputs, "Linear Reset Factor", `compositor.layers.${i}.resetLinearFactor`);
                    }
                }
                resetSelect.addEventListener("change", onResetChange);
                onResetChange();
            }
        }
        classnameSelect.addEventListener("change", onClassnameChange);
        onClassnameChange();
        const sourcesContainer = create(layerContainer, "div", "sources");
        for (let j = 0; j < config.compositor.layers[i].sourceCount; j++) {
            const sourceContainer = create(sourcesContainer, "div", "source");
            const typeSelect = createSelect(sourceContainer, "Type", `compositor.layers.${i}.sources.${j}.type`, ["file", "color", "noise", "bwnoise", "cnoise", "gradient", "first"]);
            const sourceInputs = create(sourceContainer, "div", "input-container");
            function onBitmapSelectChange() {
                const value = getSelectedValue(typeSelect);
                sourceInputs.innerHTML = "";
                if (value == "file") {
                    createFileOpenInput(sourceInputs, "File", `compositor.layers.${i}.sources.${j}.file`, VIDEO_FILETYPES + " " + IMAGE_FILETYPES);
                } else if (value == "color") {
                    createColorInput(sourceInputs, "Color", `compositor.layers.${i}.sources.${j}.color`);
                }
            }
            typeSelect.addEventListener("change", onBitmapSelectChange);
            onBitmapSelectChange();
            createFileOpenInput(sourceContainer, "Alteration", `compositor.layers.${i}.sources.${j}.alterationPath`, "*.png");
            createTimestampInput(sourceContainer, "Seek Time", `compositor.layers.${i}.sources.${j}.seekTime`, "00:00:00");
            createNumberInput(sourceContainer, "Repeat", `compositor.layers.${i}.sources.${j}.repeat`, 0, null, 1);    
            const buttonContainer = create(sourceContainer, "div", "button-container");
            createMoveButton(buttonContainer, "Move Source", `compositor.layers.${i}.sourceCount`, `compositor.layers.${i}.sources`, j);
            createDeleteButton(buttonContainer, "Delete Source", `compositor.layers.${i}.sourceCount`, `compositor.layers.${i}.sources`, j);
        }
        const buttonContainer = create(layerContainer, "div", "button-container");
        createAddButton(buttonContainer, "Add Source", `compositor.layers.${i}.sourceCount`);
        createMoveButton(buttonContainer, "Move Layer", `compositor.layerCount`, `compositor.layers`, i);
        createDeleteButton(buttonContainer, "Delete Layer", `compositor.layerCount`, `compositor.layers`, i);
    }
    const buttonContainer = create(container, "div", "button-container");
    createAddButton(buttonContainer, "Add Layer", "compositor.layerCount");
}

function inflatePaneOutput(container) {
    container.innerHTML = "";
    createFileSaveInput(container, "File", "output.file", ".mp4", VIDEO_FILETYPES);
    createTextInput(container, "Video Codec", "output.vcodec", "h264");
    createBoolInput(container, "Output Intensity", "output.outputIntensity");
    createNumberInput(container, "Render Scale", "output.renderScale", null, null, 0.001);
    createTextInput(container, "Render Colors", "output.renderColors", "#000000,#ffffff");
    createBoolInput(container, "Render Binary", "output.renderBinary");
    createNumberInput(container, "Checkpoint Every", "output.checkpointEvery", 0, null, 1);
    createBoolInput(container, "Checkpoint End", "output.checkpointEnd");
}

function inflateLeftPanel(container) {
    container.innerHTML = "";
    const tabsBar = create(container, "div", "tabsbar");
    for (const tabName of ["Flow Source", "Compositor", "Output"]) {
        const button = create(tabsBar, "div", "tabsbar-item");
        if (tabName == leftPanelActiveTab) {
            button.classList.add("active");
        }
        button.textContent = tabName;
        button.addEventListener("click", () => {
            leftPanelActiveTab = tabName;
            inflateLeftPanel(container);
        });
    }
    const paneMain = create(container, "div", "pane");
    switch (leftPanelActiveTab) {
        case "Flow Source":
            inflatePaneFlowSource(paneMain);
            break;
        case "Compositor":
            inflatePaneCompositor(paneMain);
            break;
        case "Output":
            inflatePaneOutput(paneMain);
            break;
    }
}

function inflateRightPanel(container) {
    container.innerHTML = "";
    const pane = create(container, "div", "pane");

    if (config.outputUrl != null) {
        const videoContainer = create(pane, "div");
        const video = create(videoContainer, "video");
        video.src = `/media?url=${config.outputUrl}`;
        video.setAttribute("controls", "1");
    } else if (config.previewUrl != null && config.isRunning) {
        var imgLoaded = false;
        const imgContainer = create(pane, "div");
        const img = create(imgContainer, "img");
        img.src = "wait.gif";
        const dummyImg = new Image();
        dummyImg.onload = () => {
            if (imgLoaded) return;
            img.src = dummyImg.src;
            imgLoaded = true;
        };
        dummyImg.onerror = () => {
            if (config.isRunning) {
                dummyImg.src = config.previewUrl + "?t=" + new Date().getTime();
            }
        }
        dummyImg.src = config.previewUrl;
        const progressBarContainer = create(pane, "div", "progress-bar-container");
        progressBarContainer.setAttribute("id", "progress-bar")
        create(progressBarContainer, "progress");
        create(progressBarContainer, "div", "progress-info");
    }

    const buttonRow = create(pane, "div", "row");

    const buttonGenerate = create(buttonRow, "button");
    buttonGenerate.setAttribute("id", "button-generate");
    if (config.isRunning) buttonGenerate.setAttribute("disabled", true);
    buttonGenerate.textContent = "Generate";
    
    const buttonInterrupt = create(buttonRow, "button");
    buttonInterrupt.setAttribute("id", "button-interrupt");
    if (!config.isRunning) buttonInterrupt.setAttribute("disabled", true);
    buttonInterrupt.textContent = "Interrupt";
    
    buttonGenerate.addEventListener("click", () => {
        buttonGenerate.setAttribute("disabled", true);
        buttonInterrupt.removeAttribute("disabled");
        config.isRunning = true;
        config.previewUrl = null;
        config.outputUrl = null;
        websocket.send(`GENERATE ${JSON.stringify(config)}`);
    });
    buttonInterrupt.addEventListener("click", () => {
        websocket.send("INTERRUPT");
    });

    const paneFooter = create(container, "div", "pane pane-shrink");
    const seedInput = createNumberInput(paneFooter, "Seed", "seed", 0, 2147483647, 1);
    const seedInputContainer = seedInput.parentElement;
    const seedRandomButton = create(seedInputContainer, "button");
    seedRandomButton.textContent = "Random";
    seedRandomButton.addEventListener("click", () => {
        const value = Math.floor(Math.random() * Math.pow(2, 31) - 1);
        configSet("seed", value);
        seedInput.value = value;
    });

}

function inflateBody(container) {
    container.innerHTML = "";
    leftPanel = create(container, "div", "panel panel-left");
    inflateLeftPanel(leftPanel);
    rightPanel = create(container, "div", "panel panel-right");
    inflateRightPanel(rightPanel);
}

function inflateFooter(container) {
    container.innerHTML = "";
    const footerLeft = create(container, "div", "footer-left");
    create(footerLeft, "span").innerHTML = `<b>transflow</b>`;
    create(footerLeft, "span").innerHTML = `<a href="https://chalier.fr/transflow/">web</a>`;
    create(footerLeft, "span").innerHTML = `<a href="https://github.com/ychalier/transflow">github</a>`;
    create(footerLeft, "span").innerHTML = `<a href="https://github.com/ychalier/transflow/blob/main/USAGE.md">docs</a>`;
    create(footerLeft, "span").innerHTML = `<a href="https://chalier.fr/">author</a>`;
    const footerRight = create(container, "div", "footer-right");
    wssConnectionIndicator = create(footerRight, "span");
    wssConnectionIndicator.textContent = "disconnected";
}

function inflate() {
    document.body.innerHTML = "";
    const body = create(document.body, "div", "body");
    inflateBody(body);
    const footer = create(document.body, "div", "footer");
    inflateFooter(footer);
    const overlay = create(document.body, "div", "overlay");
    const overlaySpan = create(overlay, "span")
    overlaySpan.setAttribute("state", "-1");
    function writeOverlay() {
        let state = parseInt(overlaySpan.getAttribute("state"));
        state = (state + 1) % 4;
        overlaySpan.setAttribute("state", state);
        overlaySpan.textContent = "Connecting" + ".".repeat(state);
    }
    writeOverlay();
    overlayInterval = setInterval(writeOverlay, 500);
}

function onLoad() {
    openWssConnection();
    inflate();
}

window.addEventListener("load", onLoad);