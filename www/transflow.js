/** CONSTANTS *****************************************************************/

const tINT = 0;
const tFLOAT = 1;


/** GLOBAL VARIABLES **********************************************************/

var inputCounter = 0;
var textureSourcesReady = 0;
var initialized = false;
var cursorTimeout = null;
var writeParamsToUrlTimeout = null;
var videoDevices = null;


/** UTILITY FUNCTIONS *********************************************************/

function create(parent = null, tag = "div", className = "") {
    const element = document.createElement(tag);
    element.className = className;
    if (parent != null) {
        parent.appendChild(element);
    }
    return element;
}


function getClosestInteger(x, y) {
    return Math.round(x / y) * y;
}


function createEmptyArrayInt16(width, height) {
    const array = [];
    for (let k = 0; k < width * height; k++) {
        array.push(0);
        array.push(128);
        array.push(0);
        array.push(128);
    }
    return new Uint8Array(array);
}


function createEmptyArray(width, height) {
    const array = [];
    for (let k = 0; k < width * height * 4; k++) {
        array.push(0);
    }
    return new Uint8Array(array);
}


function getSelectedOption(select) {
    for (const option of select.querySelectorAll("option")) {
        if (option.selected) return option.value;
    }
}


function setSelectedOption(select, optionValue) {
    for (const option of select.querySelectorAll("option")) {
        if (option.value == optionValue) {
            option.selected = true;
        } else {
            option.removeAttribute("selected");
        }
    }
    select.dispatchEvent(new Event("change"));
}


function smartFormat(x) {
    if (Math.floor(x) == x) {
        return x.toString();
    } else if (x > 1000) {
        return x.toFixed(0);
    } else if (x > 100) {
        return x.toFixed(1);
    } else if (x > 10) {
        return x.toFixed(2);
    } else if (x >= 0.00099) {
        return x.toFixed(3);
    } else {
        return x.toExponential(1);
    }
}


const getVideoDevices = new Promise((resolve, reject) => {
    if (videoDevices != null && videoDevices.length > 0) {
        resolve(videoDevices);
        return;
    }
    videoDevices = [];
    navigator.mediaDevices.getUserMedia({ video: true }).then(() => {
        navigator.mediaDevices.enumerateDevices().then((devices) => {
            for (const device of devices) {
                if (device.kind == "videoinput") {
                    videoDevices.push(device);
                }
            }
            resolve(devices);
        });
    }).catch((err) => {
        reject(err);
    });
});


const ident = x => x;
const formatSource = x => typeof(source) == "object" ? x.name : x;
const parseOdd = x => parseInt(x) * 2 + 1;
const formatOdd = x => Math.floor((x - 1) / 2);
const parseExp = x => Math.pow(10, parseFloat(x));
const formatExp = x => Math.log10(x);
// Ease between [0, 1] with x -> x^p with p such as y = 0.5^p
const parseEase = y => x => Math.pow(parseFloat(x), Math.log10(y) / Math.log10(0.5));
const formatEase = y => x => Math.pow(x, Math.log10(0.5) / Math.log10(y));


/** PARAMETERS ****************************************************************/

class Parameter {
    constructor(label, defl, urlParser, urlFormatter, inputParser, inputFormatter) {
        this.label = label;
        this.defl = defl;
        this.urlParser = urlParser;
        this.urlFormatter = urlFormatter;
        this.inputParser = inputParser;
        this.inputFormatter = inputFormatter;
        this.value = this.defl;
    }

    parseUrlValue(urlValue) {
        this.value = this.urlParser(urlValue);
    }

    formatUrlValue() {
        return this.urlFormatter(this.value);
    }

    parseInputValue(inputValue) {
        this.value = this.inputParser(inputValue);
        return this.value;
    }

    formatInputValue(value=null) {
        return value == null ? this.inputFormatter(this.value) : this.inputFormatter(value);
    }
}


class RangeParameter extends Parameter {
    constructor(label, defl, urlParser, urlFormatter, inputParser, inputFormatter, min=0, max=1) {
        super(label, defl, urlParser, urlFormatter, inputParser, inputFormatter);
        this.min = min;
        this.max = max;
        this.inputEl = null;
        this.valueEl = null;
        this.onInput = null;
    }

    create(parent, step=1, onInput=undefined) {
        const inputId = inputCounter;
        inputCounter++;
        const containerEl = create(parent, "div", "parameter");
        const labelEl = create(containerEl, "label", "parameter-label");
        labelEl.textContent = this.label;
        this.inputEl = create(containerEl, "input", "parameter-input");
        this.inputEl.setAttribute("id", `rangeInput-${inputId}`);
        this.inputEl.type = "range";
        this.inputEl.step = step;
        this.inputEl.min = this.formatInputValue(this.min);
        this.inputEl.max = this.formatInputValue(this.max);
        this.inputEl.value = this.formatInputValue(this.value);
        labelEl.setAttribute("for", `rangeInput-${inputId}`);
        this.valueEl = create(containerEl, "span", "parameter-value");
        this.valueEl.textContent = smartFormat(this.value);
        var self = this;
        this.onInput = onInput;
        this.inputEl.addEventListener("input", () => {
            self.setValue(self.parseInputValue(self.inputEl.value));
        });
        this.inputEl.addEventListener("dblclick", () => {
            self.reset();
        });
        return containerEl;
    }

    setValue(value) {
        this.value = value;
        this.inputEl.value = this.formatInputValue(this.value);
        this.valueEl.textContent = smartFormat(this.value);
        writeParamsToUrl(true);
        if (this.onInput != undefined) {
            this.onInput(this.value);
        }
    }

    reset() {
        this.setValue(this.defl);
    }
}


const suggestedDimensions = {
    width: window.innerWidth < window.innerHeight ? 720 : 1280,
    height: window.innerWidth < window.innerHeight ? 1280 : 720
};


var params = {
    motionType: new Parameter("Motion Source Type", "webcam", ident, ident, ident, ident),
    motionSource: new Parameter("Motion Source", "0", ident, formatSource, ident, ident),
    bitmapType: new Parameter("Bitmap Source Type", "webcam", ident, ident, ident, ident),
    bitmapSource: new Parameter("Bitmap Source", "1", ident, formatSource, ident, ident),
    width: new Parameter("Width", suggestedDimensions.width, parseInt, ident, parseInt, ident),
    height: new Parameter("Height", suggestedDimensions.height, parseInt, ident, parseInt, ident),
    flowWidth: new Parameter("Flow Width", suggestedDimensions.width, parseInt, ident, parseInt, ident),
    flowHeight: new Parameter("Flow Height", suggestedDimensions.height, parseInt, ident, parseInt, ident),
    blockSize: new Parameter("Block Size", 1, parseInt, ident, parseInt, ident),
    fps: new Parameter("FPS", 60, parseInt, ident, parseInt, ident),
    interpolation: new Parameter("Texture Interpolation", "nearest", ident, ident, ident, ident),
    method: new Parameter("Method", "hs", ident, ident, ident, ident),
    showFlow: new RangeParameter("show flow", 0, parseInt, ident, parseInt, ident),
    lockFlow: new RangeParameter("lock flow", 0, parseInt, ident, parseInt, ident),
    lockBitmap: new RangeParameter("lock bitmap", 0, parseInt, ident, parseInt, ident),
    windowSize: new RangeParameter("window", 7, parseInt, ident, parseOdd, formatOdd, 1, 13),
    gaussianSize: new RangeParameter("neighborhood", 5, parseInt, ident, parseOdd, formatOdd, 1, 13),
    minimumMovement: new RangeParameter("threshold", 0.03, parseFloat, ident, parseEase(0.03), formatEase(0.03)),
    flowDecay: new RangeParameter("decay", 0.95, parseFloat, ident, parseFloat, ident),
    alpha: new RangeParameter("alpha", 1, parseFloat, ident, parseExp, formatExp, 0.01, 100),
    scale: new RangeParameter("scale", 1, parseFloat, ident, parseExp, formatExp, 0.01, 100),
    blurSize: new RangeParameter("blur", 1, parseInt, ident, parseOdd, formatOdd, 1, 13),
    decay: new RangeParameter("recall", 0.001, parseFloat, ident, parseEase(0.001), formatEase(0.001)),
    flowMirrorX: new RangeParameter("flow mirror x", 0, parseInt, ident, parseInt, ident),
    flowMirrorY: new RangeParameter("flow mirror y", 0, parseInt, ident, parseInt, ident),
    bitmapMirrorX: new RangeParameter("bitmap mirror x", 0, parseInt, ident, parseInt, ident),
    bitmapMirrorY: new RangeParameter("bitmap mirror y", 0, parseInt, ident, parseInt, ident),
    bypassMenu: new Parameter("bypass menu", 0, parseInt, ident, parseInt, ident),
}


const rangeParams = ["showFlow", "lockFlow", "lockBitmap", "windowSize",
    "gaussianSize", "minimumMovement", "flowDecay", "alpha", "scale",
    "blurSize", "decay", "flowMirrorX", "flowMirrorY", "bitmapMirrorX",
    "bitmapMirrorY"];


function readParamsFromUrl() {
    const urlParams = new URLSearchParams(window.location.search);
    for (param in params) {
        if (urlParams.has(param)) {
            params[param].parseUrlValue(urlParams.get(param));
        }
    }
}


function writeParamsToUrl(delayed=false, push=false) {
    const url = new URL(location.protocol + "//" + location.host + location.pathname + "?");
    for (param in params) {
        url.searchParams.set(param, params[param].formatUrlValue());
    }
    if (writeParamsToUrlTimeout != null) {
        clearTimeout(writeParamsToUrlTimeout);
        writeParamsToUrlTimeout = null;
    }
	const timeout = delayed ? 1000 : 0;
    writeParamsToUrlTimeout = setTimeout(() => {
		if (push) {
            window.history.pushState("", "", url);
        } else {
			window.history.replaceState("", "", url);
		}
		writeParamsToUrlTimeout = null;
    }, timeout);
}


function readParamsFromForm(form) {
    const formData = new FormData(form);
    for (const param in params) {
        if (formData.has(param)) {
            params[param].parseInputValue(formData.get(param));
        }
    }
}


/** WEBGL UTILITY FUNCTIONS ***************************************************/

async function createShader(gl, shaderType, shaderUrl) {
    const shader = gl.createShader(shaderType);
    gl.shaderSource(shader, await fetch(shaderUrl).then(res => res.text()));
    gl.compileShader(shader);
    return shader;
}


function createProgram(gl, vertexShader, fragmentShader) {
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    return program;
}


function initializeTexture(gl, texture) {
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, params.interpolation == "linear" ? gl.LINEAR : gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, params.interpolation == "linear" ? gl.LINEAR : gl.NEAREST);
}


function initializeBuffer(gl) {
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
}


function bindTexture(gl, program, varName, textureUnit, texture) {
    gl.uniform1i(gl.getUniformLocation(program, varName), textureUnit);
    gl.activeTexture(gl.TEXTURE0 + textureUnit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
}

/** MENU FUNCTIONS ************************************************************/

function inflateMenu() {

    document.getElementById("button-reset-menu").addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        window.location.href = new URL(location.protocol + "//" + location.host + location.pathname);
        return false;
    });
    
    setSelectedOption(document.getElementById("select-motion-source"), params.motionType.value);
    if (params.motionType.value == "video") {
        document.getElementById("select-motion-video").value = params.motionSource.value;
    }

    setSelectedOption(document.getElementById("select-bitmap-source"), params.bitmapType.value);
    if (params.bitmapType.value == "video") {
        document.getElementById("select-bitmap-video").value = params.bitmapSource.value;
    } else if (params.bitmapType.value == "image") {
        document.getElementById("select-bitmap-image").value = params.bitmapSource.value;
    }

    document.querySelectorAll(".optional-input").forEach((formGroup) => {
        const select = document.getElementById(formGroup.getAttribute("for"));
        function onSelectInput() {
            if (getSelectedOption(select) == formGroup.getAttribute("option")) {
                formGroup.classList.remove("hidden");
                formGroup.querySelectorAll("input,select").forEach(el => {
                    el.removeAttribute("disabled");
                });
            } else {
                formGroup.classList.add("hidden");
                formGroup.querySelectorAll("input,select").forEach(el => {
                    el.disabled = true;
                });
            }
        }
        select.addEventListener("change", onSelectInput);
        onSelectInput();
    });

    document.getElementById("select-motion-webcam").disabled = true;
    document.getElementById("select-bitmap-webcam").disabled = true;

    getVideoDevices.then(devices => {
        document.querySelectorAll(".select-webcam").forEach((select, i) => {
            select.innerHTML = "";
            for (let j = 0; j < videoDevices.length; j++) {
                const option = document.createElement("option");
                option.textContent = videoDevices[j].label;
                option.value = j;
                select.appendChild(option);
            }
        });
        if (params.motionType.value == "webcam") {
            setSelectedOption(document.getElementById("select-motion-webcam"), params.motionSource.value);
            document.getElementById("select-motion-webcam").removeAttribute("disabled");
        }
        if (params.bitmapType.value == "webcam") {
            setSelectedOption(document.getElementById("select-bitmap-webcam"), params.bitmapSource.value);
            document.getElementById("select-bitmap-webcam").removeAttribute("disabled");
        }
    }).catch((err) => {
        const options = document.querySelectorAll("option[value='webcam']");
        for (const option of options) {
            option.parentElement.removeChild(option);
        }
        document.querySelectorAll("#select-motion-source,#select-bitmap-source").forEach(selectEl => {
            selectEl.dispatchEvent(new Event("change"));
        });
    });

    document.querySelectorAll(".input-file").forEach(fileInput => {
        fileInput.addEventListener("change", () => {
            const sibling = fileInput.nextElementSibling;
            if (sibling == null) return;
            if (!(sibling.classList.contains("input-file-label"))) return;
            if (fileInput.files.length > 0) {
                sibling.textContent = fileInput.files[0].name;
            } else {
                sibling.textContent = "browse…";
            }
        });
    });
    
    document.getElementById("input-width").value = params.width.formatInputValue();
    document.getElementById("input-height").value = params.height.formatInputValue();
    document.getElementById("input-block").value = params.blockSize.formatInputValue();
    setSelectedOption(document.getElementById("select-interp"), params.interpolation.value);

    const galleryTemplate = document.getElementById("template-gallery");
    document.querySelectorAll(".gallery-wrapper").forEach(wrapper => {
        const input = document.getElementById(wrapper.getAttribute("for"));
        const node = document.importNode(galleryTemplate.content, true);
        wrapper.appendChild(node);
        wrapper.querySelectorAll("video").forEach(video => {
            video.addEventListener("mouseenter", (e) => { video.play(); });
            video.addEventListener("mouseleave", (e) => { video.pause(); });
            video.addEventListener("click", (e) => { input.value = video.src });
        });
    });

    document.getElementById("menu-form").addEventListener("submit", onMenuFormSubmit);

}


function onMenuFormSubmit(event) {
    event.preventDefault();
    const width = parseInt(document.getElementById("input-width").value);
    const height = parseInt(document.getElementById("input-height").value);
    const blockSize = parseInt(document.getElementById("input-block").value);
    if (width % blockSize != 0 || height % blockSize != 0) {
        alert("dimensions must be divisble by block size!")
        return;
    }
    readParamsFromForm(event.target);
    startMain();
}


function startMain() {
    params.flowWidth.value = Math.floor(params.width.value / params.blockSize.value);
    params.flowHeight.value = Math.floor(params.height.value / params.blockSize.value);
    writeParamsToUrl(delayed=false, push=true);
    window.addEventListener("popstate", function (e) {
        window.location.reload();
    });
    document.getElementById("menu").style.display = "none";
    main();
}


/** MAIN FUNCTIONS ************************************************************/

function binParamToSelect(param, selectSelector, callback) {
    const selectEl = document.querySelector(selectSelector);
    function onChange() {
        const selectedOption = getSelectedOption(selectEl);
        params[param].parseInputValue(selectedOption);
        callback(selectedOption);
    }
    selectEl.addEventListener("change", onChange);
    onChange();
}


function bindParamToRangeInput(gl, param, varType, programs, methodFilter) {
    const locations = [];
    for (const program of programs) {
        locations.push(gl.getUniformLocation(program, param));
    }
    function updateValue(newValue) {
        for (let i = 0; i < programs.length; i++) {
            gl.useProgram(programs[i]);
            if (varType == tINT) {
                gl.uniform1i(locations[i], newValue);
            } else if (varType == tFLOAT) {
                gl.uniform1f(locations[i], newValue);
            }
        }
    }
    const el = params[param].create(document.getElementById("range-inputs"), varType == tINT ? 1 : 0.001, updateValue);
    if (methodFilter != undefined) {
        el.classList.add("method-input");
        el.setAttribute("method", methodFilter.join(","));
    }
    updateValue(params[param].value);
}


function createTextureSource(sourceType, sourceArg, sourceWidth, sourceHeight, onTextureSourceReady) {
    let source;
    if (sourceType == "webcam" || sourceType == "video" || sourceType == "vfile") {
        source = document.createElement("video");
        source.muted = true;
        source.autoplay = true;
        source.loop = true;
        source.addEventListener("canplay", onTextureSourceReady);
    } else if (sourceType == "image" || sourceType == "ifile" || sourceType == "random") {
        source = new Image();
        source.addEventListener("load", onTextureSourceReady);
    }
    source.crossOrigin = "anonymous";
    source.width = sourceWidth;
    source.height = sourceHeight;
    if (sourceType == "webcam") {
        getVideoDevices.then(devices => {
            navigator.mediaDevices.getUserMedia({
                video: {
                    deviceId: { exact: videoDevices[parseInt(sourceArg)].deviceId },
                    width: { ideal: sourceWidth },
                    height: { ideal: sourceHeight },
                }
            }).then(
                (stream) => {
                    try {
                        if ('srcObject' in source) source.srcObject = stream;
                        else source.src = window.URL.createObjectURL(stream);
                    } catch (err) { source.src = stream; }
                }, console.log);
        });
    } else if (sourceType == "video" || sourceType == "image") {
        source.src = sourceArg;
    } else if (sourceType == "vfile" || sourceType == "ifile") {
        source.src = URL.createObjectURL(sourceArg);
    } else if (sourceType == "random") {
        source.src = `https://picsum.photos/${sourceWidth}/${sourceHeight}`;
    }
    if (sourceType == "webcam" || sourceType == "video") source.play();
    return source;
}


function showToast(event, message) {
    const toast = create(document.body, "div", "toast");
    toast.textContent = message;
    toast.style.top = event.clientY + "px";
    toast.style.left = event.clientX + "px";
    const onMouseMove = (e) => {
        toast.style.top = e.clientY + "px";
        toast.style.left = e.clientX + "px";
    }
    window.addEventListener("mousemove", onMouseMove);
    setTimeout(() => {
        window.removeEventListener("mousemove", onMouseMove);
        document.body.removeChild(toast);
    }, 700);
}


async function main() {

    const displayCanvas = document.getElementById("display");
    const gl = displayCanvas.getContext("webgl2", { "preserveDrawingBuffer": true });
    displayCanvas.width = params.width.value;
    displayCanvas.height = params.height.value;

    document.getElementById("button-refresh").addEventListener("click", () => {
        window.location.reload();
    });

    document.getElementById("button-reset").addEventListener("click", () => {
        for (const param of rangeParams) {
            params[param].reset();
        }
    });

    document.getElementById("button-recall").addEventListener("click", () => {
        gl.activeTexture(gl.TEXTURE4);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, params.width.value, params.height.value, 0, gl.RGBA, gl.UNSIGNED_BYTE, createEmptyArrayInt16(params.width.value, params.height.value));
    });

    document.getElementById("button-export").addEventListener("click", () => {
        const link = document.createElement("a");
        link.setAttribute("download", `transflow-${parseInt((new Date()) * 1)}.png`);
        link.href = displayCanvas.toDataURL("image/png");
        link.click();
    });

    document.getElementById("button-share").addEventListener("click", (event) => {
        writeParamsToUrl(false, false);
        navigator.clipboard.writeText(window.location.href);
        showToast(event, "link copied to clipboard");
    });

    function onTextureSourceReady(event) {
        try {
            event.target.play();
        } catch (err) {
            //pass
        }
        textureSourcesReady++;
        if (textureSourcesReady == 2) {
            if (initialized) {
                console.log("All sources are ready, starting animation");
                document.body.removeChild(document.getElementById("wait"));
                requestAnimationFrame(animate);
            } else {
                console.log("Sources are ready, waiting for WebGL initialization");
            }
        }
    }

    initialized = false;
    textureSourcesReady = 0;
    const motionSource = createTextureSource(params.motionType.value, params.motionSource.value, params.flowWidth.value, params.flowHeight.value, onTextureSourceReady);
    const bitmapSource = createTextureSource(params.bitmapType.value, params.bitmapSource.value, params.width.value, params.height.value, onTextureSourceReady);

    const vertexShader = await createShader(gl, gl.VERTEX_SHADER, "shaders/base.vert");
    const flowLucasKanadeShader = await createShader(gl, gl.FRAGMENT_SHADER, "shaders/flowLucasKanade.frag");
    const flowHornSchunckShader = await createShader(gl, gl.FRAGMENT_SHADER, "shaders/flowHornSchunck.frag");
    const flowPointMatchingShader = await createShader(gl, gl.FRAGMENT_SHADER, "shaders/flowPointMatching.frag");
    const accShader = await createShader(gl, gl.FRAGMENT_SHADER, "shaders/acc.frag");
    const copyShader = await createShader(gl, gl.FRAGMENT_SHADER, "shaders/copy.frag");
    const remapShader = await createShader(gl, gl.FRAGMENT_SHADER, "shaders/remap.frag");

    const pastProgram = createProgram(gl, vertexShader, copyShader);
    const flowLucasKanadeProgram = createProgram(gl, vertexShader, flowLucasKanadeShader);
    const flowHornSchunckProgram = createProgram(gl, vertexShader, flowHornSchunckShader);
    const flowPointMatchingProgram = createProgram(gl, vertexShader, flowPointMatchingShader);
    const flowProgram = {
        lk: flowLucasKanadeProgram,
        hs: flowHornSchunckProgram,
        pm: flowPointMatchingProgram,
    };
    const copyFlowProgram = createProgram(gl, vertexShader, copyShader);
    const accProgram = createProgram(gl, vertexShader, accShader);
    const feedbackProgram = createProgram(gl, vertexShader, copyShader);
    const remapProgram = createProgram(gl, vertexShader, remapShader);

    bindParamToRangeInput(gl, "windowSize", tINT, [flowPointMatchingProgram, flowLucasKanadeProgram], ["pm", "lk"]);
    bindParamToRangeInput(gl, "gaussianSize", tINT, [flowPointMatchingProgram], ["pm"]);
    bindParamToRangeInput(gl, "minimumMovement", tFLOAT, [flowPointMatchingProgram], ["pm"])
    bindParamToRangeInput(gl, "flowDecay", tFLOAT, [flowHornSchunckProgram], ["hs"]);
    bindParamToRangeInput(gl, "alpha", tFLOAT, [flowHornSchunckProgram], ["hs"]);
    bindParamToRangeInput(gl, "scale", tFLOAT, [accProgram]);
    bindParamToRangeInput(gl, "blurSize", tINT, [accProgram]);
    bindParamToRangeInput(gl, "decay", tFLOAT, [accProgram]);
    bindParamToRangeInput(gl, "flowMirrorX", tINT, [flowPointMatchingProgram, flowLucasKanadeProgram, flowHornSchunckProgram]);
    bindParamToRangeInput(gl, "flowMirrorY", tINT, [flowPointMatchingProgram, flowLucasKanadeProgram, flowHornSchunckProgram]);
    bindParamToRangeInput(gl, "bitmapMirrorX", tINT, [remapProgram]);
    bindParamToRangeInput(gl, "bitmapMirrorY", tINT, [remapProgram]);
    const rangeInputs = document.getElementById("range-inputs");
    params.showFlow.create(rangeInputs);
    params.lockFlow.create(rangeInputs);
    params.lockBitmap.create(rangeInputs);

    binParamToSelect("method", "#select-method", (methodName => {
        document.querySelectorAll(".method-input").forEach(methodInput => {
            if (methodInput.getAttribute("method").includes(methodName)) {
                methodInput.style.display = null;
            } else {
                methodInput.style.display = "none";
            }
        });
    }));

    const nextTexture = gl.createTexture();
    initializeTexture(gl, nextTexture);

    const pastTexture = gl.createTexture();
    initializeTexture(gl, pastTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, params.flowWidth.value, params.flowHeight.value, 0, gl.RGBA, gl.UNSIGNED_BYTE, createEmptyArray(params.flowWidth.value, params.flowHeight.value));
    const pastFrameBuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, pastFrameBuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, pastTexture, 0);

    const baseFlowData = [];
    for (let k = 0; k < params.flowHeight.value * params.flowWidth.value; k++) baseFlowData.push(...[0, 128, 0, 128]);

    const flowATexture = gl.createTexture();
    initializeTexture(gl, flowATexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, params.flowWidth.value, params.flowHeight.value, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(baseFlowData));
    const flowAFrameBuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, flowAFrameBuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, flowATexture, 0);

    const flowBTexture = gl.createTexture();
    initializeTexture(gl, flowBTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, params.flowWidth.value, params.flowHeight.value, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(baseFlowData));
    const flowBFrameBuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, flowBFrameBuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, flowBTexture, 0);

    const baseMappingData = [];
    for (let k = 0; k < params.height.value * params.width.value; k++) baseMappingData.push(...[0, 128, 0, 128]);

    const feedbackTexture = gl.createTexture();
    initializeTexture(gl, feedbackTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, params.width.value, params.height.value, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(baseMappingData));
    const feedbackFrameBuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, feedbackFrameBuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, feedbackTexture, 0);

    const mappingTexture = gl.createTexture();
    initializeTexture(gl, mappingTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, params.width.value, params.height.value, 0, gl.RGBA, gl.UNSIGNED_BYTE, createEmptyArray(params.width.value, params.height.value));
    const outputFrameBuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, outputFrameBuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, mappingTexture, 0);

    const bitmapTexture = gl.createTexture();
    initializeTexture(gl, bitmapTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, params.width.value, params.height.value, 0, gl.RGBA, gl.UNSIGNED_BYTE, createEmptyArray(params.width.value, params.height.value));

    gl.useProgram(pastProgram);
    initializeBuffer(gl);
    bindTexture(gl, pastProgram, "sampler", 0, nextTexture);

    for (const program of [flowLucasKanadeProgram, flowHornSchunckProgram, flowPointMatchingProgram]) {
        gl.useProgram(program);
        initializeBuffer(gl);
        gl.uniform1f(gl.getUniformLocation(program, "blockSize"), params.blockSize.value);
        gl.uniform1f(gl.getUniformLocation(program, "flowWidth"), params.flowWidth.value);
        gl.uniform1f(gl.getUniformLocation(program, "flowHeight"), params.flowHeight.value);
        bindTexture(gl, program, "next", 0, nextTexture);
        bindTexture(gl, program, "past", 1, pastTexture);
        bindTexture(gl, program, "inflow", 2, flowBTexture);
    }

    gl.useProgram(copyFlowProgram);
    initializeBuffer(gl);
    bindTexture(gl, copyFlowProgram, "sampler", 3, flowATexture);

    gl.useProgram(accProgram);
    initializeBuffer(gl);
    gl.uniform1f(gl.getUniformLocation(accProgram, "flowWidth"), params.flowWidth.value);
    gl.uniform1f(gl.getUniformLocation(accProgram, "flowHeight"), params.flowHeight.value);
    bindTexture(gl, accProgram, "flowTexture", 3, flowATexture);
    bindTexture(gl, accProgram, "feedbackTexture", 4, feedbackTexture);

    gl.useProgram(feedbackProgram);
    initializeBuffer(gl);
    bindTexture(gl, feedbackProgram, "sampler", 5, mappingTexture);

    gl.useProgram(remapProgram);
    initializeBuffer(gl);
    bindTexture(gl, remapProgram, "mapping", 5, mappingTexture);
    bindTexture(gl, remapProgram, "bitmap", 6, bitmapTexture);

    var firstFrame = true;
    var previousFrameTime = performance.now();

    function animate() {

        const currentTimeFrame = performance.now();
        if (currentTimeFrame - previousFrameTime < 1000.0 / params.fps.value) {
            requestAnimationFrame(animate);
            return;
        }
        previousFrameTime = currentTimeFrame;

        if (firstFrame) {
            firstFrame = false;
            gl.useProgram(flowProgram[params.method.value]);
            gl.activeTexture(gl.TEXTURE0);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, motionSource);
            gl.useProgram(pastProgram);
            gl.viewport(0, 0, params.flowWidth.value, params.flowHeight.value);
            gl.bindFramebuffer(gl.FRAMEBUFFER, pastFrameBuffer);
            gl.drawArrays(gl.TRIANGLES, 0, 6);
        }

        if (!params.lockFlow.value) {
            gl.useProgram(flowProgram[params.method.value]);
            gl.viewport(0, 0, params.flowWidth.value, params.flowHeight.value);
            gl.bindFramebuffer(gl.FRAMEBUFFER, flowAFrameBuffer);
            gl.activeTexture(gl.TEXTURE0);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, motionSource);
            gl.drawArrays(gl.TRIANGLES, 0, 6);
        }

        gl.useProgram(pastProgram);
        gl.viewport(0, 0, params.flowWidth.value, params.flowHeight.value);
        gl.bindFramebuffer(gl.FRAMEBUFFER, pastFrameBuffer);
        gl.drawArrays(gl.TRIANGLES, 0, 6);

        gl.useProgram(copyFlowProgram);
        gl.viewport(0, 0, params.flowWidth.value, params.flowHeight.value);
        gl.bindFramebuffer(gl.FRAMEBUFFER, flowBFrameBuffer);
        gl.drawArrays(gl.TRIANGLES, 0, 6);

        if (params.showFlow.value) {
            gl.useProgram(copyFlowProgram);
            gl.viewport(0, 0, params.flowWidth.value, params.flowHeight.value);
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.drawArrays(gl.TRIANGLES, 0, 6);
        } else {
            gl.useProgram(accProgram);
            gl.viewport(0, 0, params.width.value, params.height.value);
            gl.bindFramebuffer(gl.FRAMEBUFFER, outputFrameBuffer);
            gl.drawArrays(gl.TRIANGLES, 0, 6);

            gl.useProgram(feedbackProgram);
            gl.viewport(0, 0, params.width.value, params.height.value);
            gl.bindFramebuffer(gl.FRAMEBUFFER, feedbackFrameBuffer);
            gl.drawArrays(gl.TRIANGLES, 0, 6);

            gl.useProgram(remapProgram);
            gl.viewport(0, 0, params.width.value, params.height.value);
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.activeTexture(gl.TEXTURE6);
            if (!params.lockBitmap.value) {
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmapSource);
            }
            gl.drawArrays(gl.TRIANGLES, 0, 6);
        }

        requestAnimationFrame(animate);

    }

    if (textureSourcesReady == 2) {
        console.log("WebGL context has been initialized. Starting animation.");
        document.body.removeChild(document.getElementById("wait"));
        requestAnimationFrame(animate);
    } else {
        console.log("WebGL context has been initialized. Waiting for sources to load.");
        initialized = true;
    }

}


/** GLOBAL EVENTS *************************************************************/

window.addEventListener("mousemove", () => {
    if (cursorTimeout != null) {
        clearTimeout(cursorTimeout);
    } else {
        document.getElementById("display").classList.add("show-cursor");
    }
    cursorTimeout = setTimeout(() => {
        document.getElementById("display").classList.remove("show-cursor");
        cursorTimeout = null;
    }, 500);
});


window.addEventListener("load", () => {
    readParamsFromUrl();
    if (params.bypassMenu.value) {
        startMain();
    } else {
        inflateMenu();
    }
});
