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

var config = {
    flowSource: {
        file: null,
        direction: "backward",
        maskPath: null,
        kernelPath: null,
        cvConfig: null,
    },
    bitmapSource: {
        file: null,
        type: "file",
        color: "#cff010"
    },
    accumulator: {
        method: "map",
        resetMode: "off",
        resetAlpha: 0.1,
        resetMask: null,
        heatmapMode: "discrete",
        heatmapArgs: "0:4:2:1",
    },
    output: {
        previewUrl: null
    }
}

var leftPanelActiveTab = "Flow Source";
var websocket;
var websocketRetryCount = 0;
var leftPanel;
var rightPanel;
var wssConnectionIndicator;

function setWssConnectionIndicator(state) {
    if (wssConnectionIndicator == undefined) return;
    wssConnectionIndicator.textContent = state;
}

function configSet(keys, value) {
    let o = config;
    for (let i = 0; i < keys.length - 1; i++) {
        o = o[keys[i]];
    }
    o[keys[keys.length - 1]] = value;
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
        if (message.data.startsWith("FILEIN ")) {
            const query = message.data.slice(7);
            let [inputName, fileUrl] = query.split(" ", 2);
            if (fileUrl == "") fileUrl = null;
            const configKeys = inputName.split(".");
            configSet(configKeys, fileUrl);
            onConfigChange(`config.${inputName}`);
            inflateLeftPanel(leftPanel);
        } else if (message.data.startsWith("OUT")) {
            config.output.previewUrl = message.data.slice(4);
            onConfigChange(`config.output.previewUrl`);
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

function onConfigChange(paramName) {
    console.log("Config changed", paramName);
}

function inflateHeader(container) {
    container.innerHTML = "";
    create(container).textContent = "transflow";
}

function createFileInput(container, inputName, fileTypes) {
    const input = create(container, "button");
    input.textContent = "Select file";
    input.addEventListener("click", () => {
        const data = {
            name: inputName,
            filetypes: fileTypes,
        }
        websocket.send(`FILEIN ${JSON.stringify(data)}`);
    });
}

const VIDEO_FILETYPES = "*.mp4 *.avi *.mkv *.mov *.mpg";
const IMAGE_FILETYPES = "*.jpg *.jpeg *.png *.webp *.gif";


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
    create(inputContainer, "span").textContent = label;
    return inputContainer;
}

function inflatePaneFlowSource(container) {
    container.innerHTML = "";

    const inputFileContainer = createInputContainer(container, "File");
    createFileInput(inputFileContainer, "flowSource.file", VIDEO_FILETYPES);
    if (config.flowSource.file != null) {
        const video = create(inputFileContainer, "video");
        video.src = `/media?url=${config.flowSource.file}`;
        video.setAttribute("controls", 1);
    }

    const inputDirectionContainer = createInputContainer(container, "Direction");
    const directionSelect = create(inputDirectionContainer, "select");
    inflateSelect(directionSelect, [
        {name: "backward", label: "backward"},
        {name: "forward", label: "forward"},
    ], config.flowSource.direction);
    directionSelect.addEventListener("change", () => {
        config.flowSource.direction = getSelectedValue(directionSelect);
    });

    const inputMaskContainer = createInputContainer(container, "Mask");
    createFileInput(inputMaskContainer, "flowSource.maskPath", IMAGE_FILETYPES);
    if (config.flowSource.maskPath != null) {
        create(inputMaskContainer, "span").textContent = getPathName(config.flowSource.maskPath);
    }

    const inputKernelContainer = createInputContainer(container, "Kernel");
    createFileInput(inputKernelContainer, "flowSource.kernelPath", "*.npy");
    if (config.flowSource.kernelPath != null) {
        create(inputKernelContainer, "span").textContent = getPathName(config.flowSource.kernelPath);
    }

    const inputCvConfigContainer = createInputContainer(container, "CV Config");
    createFileInput(inputCvConfigContainer, "flowSource.cvConfig", "*.json");
    if (config.flowSource.cvConfig != null) {
        create(inputCvConfigContainer, "span").textContent = getPathName(config.flowSource.cvConfig);
    }

}

function inflateSelect(select, options, initialValue) {
    select.innerHTML = "";
    for (const op of options) {
        const option = create(select, "option");
        option.value = op.name;
        option.textContent = op.label;
        if (option.value == initialValue) option.selected = true;
    }
}

function inflatePaneBitmapSource(container) {
    container.innerHTML = "";

    const inputTypeContainer = createInputContainer(container, "Type");
    const bitmapSelect = create(inputTypeContainer, "select");
    inflateSelect(bitmapSelect, [
        {name: "file", label: "file"},
        {name: "color", label: "color"},
        {name: "noise", label: "grey noise"},
        {name: "bwnoise", label: "b&w noise"},
        {name: "cnoise", label: "colored noise"},
        {name: "gradient", label: "gradient"},
        {name: "first", label: "first frame"},
    ], config.bitmapSource.type);

    const bitmapInputs = create(container, "div", "input-container");
    function onBitmapSelectChange() {
        const value = getSelectedValue(bitmapSelect);
        config.bitmapSource.type = value;
        bitmapInputs.innerHTML = "";
        switch(value) {
            case "file":
                const inputFileContainer = createInputContainer(bitmapInputs, "File");
                createFileInput(inputFileContainer, "bitmapSource.file", VIDEO_FILETYPES + " " + IMAGE_FILETYPES);
                if (config.bitmapSource.file != null) {
                    if (VIDEO_FILETYPES.includes(getPathSuffix(config.bitmapSource.file))) {
                        const video = create(inputFileContainer, "video");
                        video.src = `/media?url=${config.bitmapSource.file}`;
                        video.setAttribute("controls", 1);
                    } else if (IMAGE_FILETYPES.includes(getPathSuffix(config.bitmapSource.file))) {
                        const image = create(inputFileContainer, "img");
                        image.src = `/media?url=${config.bitmapSource.file}`;
                    }
                }
                break;
            case "color":
                const inputColorContainer = createInputContainer(bitmapInputs, "Color");
                const colorInput = create(bitmapInputs, "input");
                colorInput.type = "color";
                if (config.bitmapSource.color != undefined) {
                    colorInput.value = config.bitmapSource.color;
                }
                colorInput.addEventListener("change", () => {
                    config.bitmapSource.color = colorInput.value;
                });
                break;
            case "noise":
            case "bwnoise":
            case "cnoise":
            case "gradient":
            case "first":
                break;
            default:
                alert("Unknown bitmap source!");
                console.error("Unkown bitmap source", value);
                break;
        }
    }

    bitmapSelect.addEventListener("change", onBitmapSelectChange);
    onBitmapSelectChange();

}

function inflatePaneAccumulator(container) {
    container.innerHTML = "";

    const inputMethodContainer = createInputContainer(container, "Method");
    const methodSelect = create(inputMethodContainer, "select");
    inflateSelect(methodSelect, [
        {name: "map", label: "map"},
        {name: "stack", label: "stack"},
        {name: "sum", label: "sum"},
        {name: "crumble", label: "crumble"},
        {name: "canvas", label: "canvas"},
    ], config.accumulator.method);

    const accumulatorInputs = create(container, "div", "input-container");
    function onAccumulatorMethodChange() {
        const value = getSelectedValue(methodSelect);
        config.accumulator.method = value;
        onConfigChange(`config.accumulator.method`);
        accumulatorInputs.innerHTML = "";

        const resetModeSelectContainer = createInputContainer(accumulatorInputs, "Reset Mode");
        const resetModeSelect = create(resetModeSelectContainer, "select");
        inflateSelect(resetModeSelect, [
            {name: "off", label: "off"},
            {name: "random", label: "random"},
            {name: "linear", label: "linear"},
        ], config.accumulator.resetMode);
        resetModeSelect.addEventListener("change", () => {
            config.accumulator.resetMode = getSelectedValue(resetModeSelect);
        });

        const resetAlphaContainer = createInputContainer(accumulatorInputs, "Reset Alpha");
        const resetAlpha = create(resetAlphaContainer, "input");
        resetAlpha.type = "range";
        resetAlpha.min = 0;
        resetAlpha.max = 1;
        resetAlpha.step = 0.001;
        resetAlpha.value = config.accumulator.resetAlpha;
        const resetAlphaLabel = create(resetAlphaContainer, "span");
        resetAlphaLabel.textContent = config.accumulator.resetAlpha.toFixed(3);
        resetAlpha.addEventListener("input", () => {
            config.accumulator.resetAlpha = parseFloat(resetAlpha.value);
            resetAlphaLabel.textContent = config.accumulator.resetAlpha.toFixed(3);
        });

        const resetMaskContainer = createInputContainer(accumulatorInputs, "Reset Mask");
        createFileInput(resetMaskContainer, "accumulator.resetMask", "*.jpg *.jpeg *.png");
        if (config.accumulator.resetMask != null) {
            create(resetMaskContainer, "span").textContent = getPathName(config.accumulator.resetMask);
        }

        const heatmapModeContainer = createInputContainer(accumulatorInputs, "Heatmap Mode");
        const heatmapModeSelect = create(heatmapModeContainer, "select");
        inflateSelect(heatmapModeSelect, [
            {name: "discrete", label: "discrete"},
            {name: "continuous", label: "continuous"},
        ], config.accumulator.heatmapMode);
        resetModeSelect.addEventListener("change", () => {
            config.accumulator.heatmapMode = getSelectedValue(heatmapModeSelect);
        });

        const heatmapArgsContainer = createInputContainer(accumulatorInputs, "Heatmap Args");
        const heatmapArgs = create(heatmapArgsContainer, "input");
        heatmapArgs.value = config.accumulator.heatmapArgs;
        heatmapArgs.addEventListener("change", () => {
            config.accumulator.heatmapArgs = heatmapArgs.value;
        });

    }

    methodSelect.addEventListener("change", onAccumulatorMethodChange);
    onAccumulatorMethodChange();
}

function inflateLeftPanel(container) {
    container.innerHTML = "";
    const tabsBar = create(container, "div", "tabsbar");
    for (const tabName of ["Flow Source", "Bitmap Source", "Accumulator"]) {
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
    const pane = create(container, "div", "pane");
    switch (leftPanelActiveTab) {
        case "Flow Source":
            inflatePaneFlowSource(pane);
            break;
        case "Bitmap Source":
            inflatePaneBitmapSource(pane);
            break;
        case "Accumulator":
            inflatePaneAccumulator(pane);
            break;
    }
}

function inflatePanePreview(container) {
    container.innerHTML = "";
    if (config.output.previewUrl != null) {
        const img = create(container, "img");
        img.src = config.output.previewUrl;
        img.addEventListener("error", () => {
            img.src = config.output.previewUrl + "?t=" + new Date().getTime();
        });
    }
}

function inflatePaneGenerate(container) {
    container.innerHTML = "";
    const buttonGenerate = create(container, "button");
    buttonGenerate.textContent = "Generate";
    buttonGenerate.addEventListener("click", () => {
        websocket.send(`GEN ${JSON.stringify(config)}`);
    });
    const buttonInterrupt = create(container, "button");
    buttonInterrupt.textContent = "Interrupt";
    buttonInterrupt.addEventListener("click", () => {
        websocket.send("INTERRUPT");
    });
}

function inflateRightPanel(container) {
    container.innerHTML = "";
    const panePreview = create(container, "div", "pane");
    panePreview.setAttribute("id", "pane-preview");
    inflatePanePreview(panePreview);
    const paneGenerate = create(container, "div", "pane");
    paneGenerate.setAttribute("id", "pane-generate");
    inflatePaneGenerate(paneGenerate);
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
    const footer = create(document.body, "div");
    inflateFooter(footer);
}

function onLoad() {
    openWssConnection();
    inflate();
}

window.addEventListener("load", onLoad);