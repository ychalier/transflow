const VIDEO_FILETYPES = "*.mp4 *.avi *.mkv *.mov *.mpg";
const IMAGE_FILETYPES = "*.jpg *.jpeg *.png";

var config = {
    seed: Math.floor(Math.random() * (Math.pow(2, 31) - 1)),
    previewUrl: null,
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
    bitmapSource: {
        file: null,
        type: "file",
        color: "#cff010",
        alterationPath: null,
        seekTime: null,
        repeat: 1,
    },
    accumulator: {
        method: "map",
        resetMode: "off",
        resetAlpha: 0.1,
        resetMask: null,
        heatmapMode: "discrete",
        heatmapArgs: "0:4:2:1",
        heatmapResetThreshold: null,
        background: "#ffffff",
        stackComposer: "top",
        initialCanvasFile: null,
        initialCanvasColor: "#ffffff",
        bitmapMask: null,
        crumble: false,
        bitmapIntroductionFlags: 1,
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

var leftPanelActiveTab = "Flow Source";
var websocket;
var websocketRetryCount = 0;
var leftPanel;
var rightPanel;
var wssConnectionIndicator;

function create(parent=null, tag="div", className=null) {
    const element = document.createElement(tag);
    if (parent != null) parent.appendChild(element);
    if (className != null) element.className = className;
    return element;
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

function configGet(key) {
    const split = key.split(".");
    let o = config;
    for (let i = 0; i < split.length - 1; i++) {
        o = o[split[i]];
    }
    return o[split[split.length - 1]];
}

function configSet(key, value) {
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
}

function createFileOpenInput(container, label, key, filetypes) {
    const inputContainer = createInputContainer(container, label);
    const inputButtons = create(inputContainer, "div", "input-buttons");
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
    const inputButtons = create(inputContainer, "div", "input-buttons");
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

function createTextInput(container, label, key) {
    const inputContainer = createInputContainer(container, label);
    const input = create(inputContainer, "input");
    input.value = configGet(key);
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

function setWssConnectionIndicator(state) {
    if (wssConnectionIndicator == undefined) return;
    wssConnectionIndicator.textContent = state;
}

function connectWebsocket(wssUrl) {
    setWssConnectionIndicator(`Connecting… [${websocketRetryCount}]`);
    websocket = new WebSocket(wssUrl);
    websocket.onopen = () => {
        websocketRetryCount = 0;
        console.log("Websocket is connected");
        setWssConnectionIndicator("Connected");
    }
    websocket.onmessage = (message) => {
        console.log(message);
        if (message.data.startsWith("FILE ")) {
            const query = message.data.slice(5);
            let [key, fileUrl] = query.split(" ", 2);
            if (fileUrl == "") fileUrl = null;
            configSet(key, fileUrl);
            inflateLeftPanel(leftPanel);
        } else if (message.data.startsWith("OUT")) {
            config.previewUrl = message.data.slice(4);
            onConfigChange("previewUrl");
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
        setWssConnectionIndicator("Closed");
        setTimeout(() => { connectWebsocket(wssUrl); }, delay * 1000);
    };
    websocket.onerror = (err) => {
        console.error("Socket encountered error: ", err.message, "Closing socket");
        setWssConnectionIndicator("Error");
        websocket.close();
    };
}

function openWssConnection() {
    setWssConnectionIndicator("Scanning…");
    fetch("/wss").then(res => res.text()).then(text => {
        const wssUrl = text;  // `ws://${res.headers.get("Wss-Host")}:${res.headers.get("Wss-Port")}`;
        connectWebsocket(wssUrl);
    });
}

function inflateHeader(container) {
    container.innerHTML = "";
    create(container).textContent = "transflow";
}

function inflatePaneFlowSource(container) {
    container.innerHTML = "";
    createFileOpenInput(container, "File", "flowSource.file", VIDEO_FILETYPES);
    createBoolInput(container, "Use Motion Vectors", "flowSource.useMvs");
    createSelect(container, "Direction", "flowSource.direction", ["backward", "forward"]);
    createFileOpenInput(container, "Mask", "flowSource.maskPath", IMAGE_FILETYPES);
    createFileOpenInput(container, "Kernel", "flowSource.kernelPath", "*.npy");
    createFileOpenInput(container, "CV Config", "flowSource.cvConfig", "*.json");
    createTextInput(container, "Filters", "flowSource.flowFilters");
    createBoolInput(container, "Round Flow", "flowSource.roundFlow");
    createBoolInput(container, "Export Flow", "flowSource.exportFlow");
    createTimestampInput(container, "Seek Time", "flowSource.seekTime");
    createTimestampInput(container, "Duration Time", "flowSource.durationTime");
    createNumberInput(container, "Repeat", "flowSource.repeat", 0, null, 1);
    createSelect(container, "Lock Mode", "flowSource.lockMode", ["stay", "skip"]);
    createTextInput(container, "Lock Expression", "flowSource.lockExpr");
}

function inflatePaneBitmapSource(container) {
    container.innerHTML = "";
    const bitmapSelect = createSelect(container, "Type", "bitmapSource.type", ["file", "color", "noise", "bwnoise", "cnoise", "gradient", "first"]);
    const bitmapInputs = create(container, "div", "input-container");
    function onBitmapSelectChange() {
        const value = getSelectedValue(bitmapSelect);
        bitmapInputs.innerHTML = "";
        if (value == "file") {
            createFileOpenInput(bitmapInputs, "File", "bitmapSource.file", VIDEO_FILETYPES + " " + IMAGE_FILETYPES);
        } else if (value == "color") {
            createColorInput(bitmapInputs, "Color", "bitmapSource.color");
        }
    }
    bitmapSelect.addEventListener("change", onBitmapSelectChange);
    onBitmapSelectChange();
    createFileOpenInput(container, "Alteration", "bitmapSource.alterationPath", "*.png");
    createTimestampInput(container, "Seek Time", "bitmapSource.seekTime");
    createNumberInput(container, "Repeat", "bitmapSource.repeat", 0, null, 1);
}

function inflatePaneAccumulator(container) {
    container.innerHTML = "";
    const methodSelect = createSelect(container, "Method", "accumulator.method", ["map", "stack", "sum", "crumble", "canvas"]);
    const accumulatorInputs = create(container, "div", "input-container");
    function onAccumulatorMethodChange() {
        const value = getSelectedValue(methodSelect);
        accumulatorInputs.innerHTML = "";
        if (value == "stack" || value == "crumble") {
            createColorInput(accumulatorInputs, "Background", "accumulator.background");
        } else if (value == "stack") {
            createSelect(accumulatorInputs, "Stack Composer", "accumulator.stackComposer", ["top", "add", "sub", "avg"]);
        } else if (value == "canvas") {
            createFileOpenInput(accumulatorInputs, "Initial Canvas File", "accumulator.initialCanvasFile", IMAGE_FILETYPES);
            createColorInput(accumulatorInputs, "Initial Canvas Color", "accumulator.initialCanvasColor");
            createFileOpenInput(accumulatorInputs, "Bitmap Mask", "accumulator.bitmapMask", IMAGE_FILETYPES);
            createNumberInput(accumulatorInputs, "Bitmap Introduction Flags", "accumulator.bitmapIntroductionFlags", 0, null, 1);
            createBoolInput(accumulatorInputs, "Crumble", "accumulator.crumble");
        }
    }
    methodSelect.addEventListener("change", onAccumulatorMethodChange);
    onAccumulatorMethodChange();
    createSelect(container, "Reset Mode", "accumulator.resetMode", ["off", "random", "linear"]);
    createRangeInput(container, "Reset Alpha", "accumulator.resetAlpha");
    createFileOpenInput(container, "Reset Mask", "accumulator.resetMask", "*.jpg *.jpeg *.png");
    createSelect(container, "Heatmap Mode", "accumulator.heatmapMode", ["discrete", "continuous"]);
    createTextInput(container, "Heatmap Args", "accumulator.heatmapArgs");
    createTextInput(container, "Heatmap Reset Threshold", "accumulator.heatmapResetThreshold"); // TODO: maybe set to null if empty or parseFloat
}

function inflatePaneOutput(container) {
    container.innerHTML = "";
    createFileSaveInput(container, "File", "output.file", ".mp4", VIDEO_FILETYPES);
    createTextInput(container, "Video Codec", "output.vcodec");
    createBoolInput(container, "Output Intensity", "output.outputIntensity");
    createBoolInput(container, "Output Heatmap", "output.outputHeatmap");
    createBoolInput(container, "Output Accumulator", "output.outputAccumulator");
    createNumberInput(container, "Render Scale", "output.renderScale", null, null, 0.001);
    createTextInput(container, "Render Colors", "output.renderColors");
    createBoolInput(container, "Render Binary", "output.renderBinary");
    createNumberInput(container, "Checkpoint Every", "output.checkpointEvery", 0, null, 1);
    createBoolInput(container, "Checkpoint End", "output.checkpointEnd");
}

function inflateLeftPanel(container) {
    container.innerHTML = "";
    const tabsBar = create(container, "div", "tabsbar");
    for (const tabName of ["Flow Source", "Bitmap Source", "Accumulator", "Output"]) {
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
        case "Bitmap Source":
            inflatePaneBitmapSource(paneMain);
            break;
        case "Accumulator":
            inflatePaneAccumulator(paneMain);
            break;
        case "Output":
            inflatePaneOutput(paneMain);
            break;
    }
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

function inflateRightPanel(container) {
    container.innerHTML = "";
    const pane = create(container, "div", "pane");

    if (config.previewUrl != null) {
        const imgContainer = create(pane, "div");
        const img = create(imgContainer, "img");
        img.src = config.previewUrl;
        img.addEventListener("error", () => {
            img.src = config.previewUrl + "?t=" + new Date().getTime();
        });
    }

    const buttonGenerate = create(pane, "button");
    buttonGenerate.textContent = "Generate";
    buttonGenerate.addEventListener("click", () => {
        websocket.send(`GENERATE ${JSON.stringify(config)}`);
    });
    const buttonInterrupt = create(pane, "button");
    buttonInterrupt.textContent = "Interrupt";
    buttonInterrupt.addEventListener("click", () => {
        websocket.send("INTERRUPT");
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
    wssConnectionIndicator = create(container, "span");
    wssConnectionIndicator.textContent = "Disconnected";
    create(container, "span").innerHTML = `<a href="https://github.com/ychalier/transflow">GitHub</a>`;
}

function inflate() {
    document.body.innerHTML = "";
    const header = create(document.body, "div", "header");
    inflateHeader(header);
    const body = create(document.body, "div", "body");
    inflateBody(body);
    const footer = create(document.body, "div", "footer");
    inflateFooter(footer);
}

function onLoad() {
    openWssConnection();
    inflate();
}

window.addEventListener("load", onLoad);