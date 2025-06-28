const VIDEO_FILETYPES = "*.mp4 *.avi *.mkv *.mov *.mpg";
const IMAGE_FILETYPES = "*.jpg *.jpeg *.png";

var config = {
    flowSource: {
        file: null,
        direction: "backward",
        maskPath: null,
        kernelPath: null,
        cvConfig: null,
        flowFilters: null,
        useMvs: false,
    },
    bitmapSource: {
        file: null,
        type: "file",
        color: "#cff010",
        alterationPath: null,
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
        crumble: false
    },
    output: {
        previewUrl: null
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
    create(inputContainer, "span").textContent = label;
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

function createFileInput(container, label, key, filetypes) {
    const inputContainer = createInputContainer(container, label);
    const input = create(inputContainer, "button");
    input.textContent = "Select file";
    input.addEventListener("click", () => {
        const data = { key: key, filetypes: filetypes };
        websocket.send(`FILEIN ${JSON.stringify(data)}`);
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
    }
}

function createBoolInput(container, label, key) {
    const inputContainer = createInputContainer(container, label);
    const input = create(inputContainer, "input");
    input.type = "checkbox";
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
        configSet(key, input.value);
    });
}

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
        if (message.data.startsWith("FILEIN ")) {
            const query = message.data.slice(7);
            let [key, fileUrl] = query.split(" ", 2);
            if (fileUrl == "") fileUrl = null;
            configSet(key, fileUrl);
            inflateLeftPanel(leftPanel);
        } else if (message.data.startsWith("OUT")) {
            config.output.previewUrl = message.data.slice(4);
            onConfigChange("output.previewUrl");
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
    createFileInput(container, "File", "flowSource.file", VIDEO_FILETYPES);
    createBoolInput(container, "Use Motion Vectors", "flowSource.useMvs");
    createSelect(container, "Direction", "flowSource.direction", ["backward", "forward"]);
    createFileInput(container, "Mask", "flowSource.maskPath", IMAGE_FILETYPES);
    createFileInput(container, "Kernel", "flowSource.kernelPath", "*.npy");
    createFileInput(container, "CV Config", "flowSource.cvConfig", "*.json");
    createTextInput(container, "Filters", "flowSource.flowFilters");
}

function inflatePaneBitmapSource(container) {
    container.innerHTML = "";
    const bitmapSelect = createSelect(container, "Type", "bitmapSource.type", ["file", "color", "noise", "bwnoise", "cnoise", "gradient", "first"]);
    const bitmapInputs = create(container, "div", "input-container");
    function onBitmapSelectChange() {
        const value = getSelectedValue(bitmapSelect);
        bitmapInputs.innerHTML = "";
        switch(value) {
            case "file":
                createFileInput(bitmapInputs, "File", "bitmapSource.file", VIDEO_FILETYPES + " " + IMAGE_FILETYPES);
                break;
            case "color":
                createColorInput(bitmapInputs, "Color", "bitmapSource.color");
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
    createFileInput(container, "Alteration", "bitmapSource.alterationPath", "*.png");
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
            createFileInput(accumulatorInputs, "Initial Canvas File", "accumulator.initialCanvasFile", IMAGE_FILETYPES);
            createColorInput(accumulatorInputs, "Initial Canvas Color", "accumulator.initialCanvasColor");
            createFileInput(accumulatorInputs, "Bitmap Mask", "accumulator.bitmapMask", IMAGE_FILETYPES);
            createBoolInput(accumulatorInputs, "Crumble", "accumulator.crumble");
        }
    }
    methodSelect.addEventListener("change", onAccumulatorMethodChange);
    onAccumulatorMethodChange();
    createSelect(container, "Reset Mode", "accumulator.resetMode", ["off", "random", "linear"]);
    createRangeInput(container, "Reset Alpha", "accumulator.resetAlpha");
    createFileInput(container, "Reset Mask", "accumulator.resetMask", "*.jpg *.jpeg *.png");
    createSelect(container, "Heatmap Mode", "accumulator.heatmapMode", ["discrete", "continuous"]);
    createTextInput(container, "Heatmap Args", "accumulator.heatmapArgs");
    createTextInput(container, "Heatmap Reset Threshold", "accumulator.heatmapResetThreshold"); // TODO: maybe set to null if empty or parseFloat
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