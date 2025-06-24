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
    },
    bitmapSource: {
        file: null,
    },
    accumulator: {
        method: "map"
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

function connectWebsocket(wssUrl) {
    websocket = new WebSocket(wssUrl);
    websocket.onopen = () => {
        websocketRetryCount = 0;
        console.log("Websocket is connected");
    }
    websocket.onmessage = (message) => {
        console.log(message);
        if (message.data.startsWith("FILEIN FS")) {
            const fileUrl = message.data.slice(10);
            if (fileUrl == "") {
                config.flowSource.file = null;
            } else {
                config.flowSource.file = fileUrl;
            }
            onConfigChange(`config.flowSource.file`);
            inflateLeftPanel(leftPanel);
        } else if (message.data.startsWith("FILEIN BM")) {
            const fileUrl = message.data.slice(10);
            if (fileUrl == "") {
                config.bitmapSource.file = null;
            } else {
                config.bitmapSource.file = fileUrl;
            }
            onConfigChange(`config.bitmapSource.file`);
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
        let delay = 1;
        if (websocketRetryCount >= 10) {
            delay = 30;
        } else if (websocketRetryCount >= 3) {
            delay = 5;
        }
        console.log(`Socket is closed. Reconnect will be attempted in ${delay} second.`, event.reason);
        setTimeout(() => { connectWebsocket(wssUrl); }, delay * 1000);
    };
    websocket.onerror = (err) => {
        console.error("Socket encountered error: ", err.message, "Closing socket");
        websocket.close();
    };
}

function openWssConnection() {
    fetch("/").then(res => {
        const wssUrl = `ws://${res.headers.get("Wss-Host")}:${res.headers.get("Wss-Port")}`;
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

function createFileInput(container, inputName) {
    const input = create(container, "button");
    input.textContent = "Select file";
    input.addEventListener("click", () => {
        websocket.send(`FILEIN ${inputName}`);
    });
}

function inflatePaneFlowSource(container) {
    container.innerHTML = "";
    createFileInput(container, "FS");
    if (config.flowSource.file != null) {
        const video = create(container, "video");
        video.src = `/media?url=${config.flowSource.file}`;
        video.setAttribute("controls", 1);
    }
}

function inflatePaneBitmapSource(container) {
    container.innerHTML = "";
    createFileInput(container, "BM");
    if (config.bitmapSource.file != null) {
        const video = create(container, "video");
        video.src = `/media?url=${config.bitmapSource.file}`;
        video.setAttribute("controls", 1);
    }
}

function inflatePaneAccumulator(container) {
    container.innerHTML = "";
    const methodSelect = create(container, "select");
    for (const methodName of ["map", "stack", "sum", "crumble", "canvas"]) {
        const option = create(methodSelect, "option");
        option.textContent = methodName;
        option.value = methodName;
        if (methodName == config.accumulator.method) {
            option.selected = true;
        }
    }
    methodSelect.addEventListener("input", () => {
        config.accumulator.method = getSelectedValue(methodSelect);
        onConfigChange(`config.accumulator.method`);
    });
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