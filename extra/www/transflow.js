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


class BooleanParameter extends Parameter {
    constructor(label, defl) {
        super(label, defl, parseInt, ident, parseInt, ident);
        this.inputEl = null;
        this.onInput = null;
    }

    create(parent, onInput=undefined) {
        const inputId = inputCounter;
        inputCounter++;
        const containerEl = create(parent, "label", "parameter-boolean");
        this.inputEl = create(containerEl, "input", "parameter-input");
        this.inputEl.setAttribute("id", `booleanInput-${inputId}`);
        this.inputEl.type = "checkbox";
        this.inputEl.checked = this.formatInputValue(this.value) == 1;
        containerEl.setAttribute("for", `booleanInput-${inputId}`);
        const labelEl = create(containerEl, "span");
        labelEl.textContent = this.label;
        var self = this;
        this.onInput = onInput;
        this.inputEl.addEventListener("input", () => {
            self.setValue(self.parseInputValue(self.inputEl.checked ? 1 : 0));
        });
        this.inputEl.addEventListener("dblclick", () => {
            self.reset();
        });
        return containerEl;
    }

    setValue(value) {
        this.value = value;
        this.inputEl.checked = this.formatInputValue(this.value) == 1;
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
    showFlow: new BooleanParameter("show flow", 0),
    lockFlow: new BooleanParameter("lock flow", 0),
    lockBitmap: new BooleanParameter("lock bitmap", 0),
    windowSize: new RangeParameter("window", 7, parseInt, ident, parseOdd, formatOdd, 1, 13),
    gaussianSize: new RangeParameter("neighborhood", 5, parseInt, ident, parseOdd, formatOdd, 1, 13),
    flowDecay: new RangeParameter("decay", 0.95, parseFloat, ident, parseFloat, ident),
    alpha: new RangeParameter("alpha", 1, parseFloat, ident, parseExp, formatExp, 0.01, 100),
    scale: new RangeParameter("scale", 1, parseFloat, ident, parseExp, formatExp, 0.01, 100),
    minimumMovement: new RangeParameter("threshold", 0.03, parseFloat, ident, parseEase(0.03), formatEase(0.03)),
    threshold: new RangeParameter("threshold", 0, parseFloat, ident, parseEase(0.001), formatEase(0.001), 0, 1),
    blurSize: new RangeParameter("blur", 1, parseInt, ident, parseOdd, formatOdd, 1, 13),
    decay: new RangeParameter("recall", 0.001, parseFloat, ident, parseEase(0.001), formatEase(0.001)),
    feedback: new RangeParameter("feedback", 1, parseInt, ident, parseInt, ident),
    iterations: new RangeParameter("iterations", 10, parseInt, ident, parseInt, ident, 1, 100),
    flowMirrorX: new BooleanParameter("flow flip x", 0),
    flowMirrorY: new BooleanParameter("flow flip y", 0),
    bitmapMirrorX: new BooleanParameter("bitmap flip x", 0),
    bitmapMirrorY: new BooleanParameter("bitmap flip y", 0),
    bypassMenu: new Parameter("bypass menu", 0, parseInt, ident, parseInt, ident),
}


const rangeParams = ["showFlow", "lockFlow", "lockBitmap", "windowSize",
    "gaussianSize", "minimumMovement", "flowDecay", "alpha", "scale",
    "blurSize", "decay", "flowMirrorX", "flowMirrorY", "bitmapMirrorX",
    "bitmapMirrorY", "feedback", "iterations", "threshold"];


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


function initializeTextureFloat(gl, texture, width, height) {
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(
        gl.TEXTURE_2D,      // Target
        0,                  // Mipmap level
        gl.RGBA32F,         // Internal format (high precision float)
        width, height,      // Width, Height
        0,                  // Border (must be 0)
        gl.RGBA,            // Format
        gl.FLOAT,           // Type
        null                // No initial data (allocate only)
    );
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
    return el;
}


function bindParamToBooleanInput(gl, param, programs, methodFilter) {
    const locations = [];
    for (const program of programs) {
        locations.push(gl.getUniformLocation(program, param));
    }
    function updateValue(newValue) {
        for (let i = 0; i < programs.length; i++) {
            gl.useProgram(programs[i]);
            gl.uniform1i(locations[i], newValue);
        }
    }
    const el = params[param].create(document.getElementById("boolean-inputs"), updateValue);
    if (methodFilter != undefined) {
        el.classList.add("method-input");
        el.setAttribute("method", methodFilter.join(","));
    }
    updateValue(params[param].value);
    return el;
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
    source.addEventListener("error", (event) => {
        console.error(`Error loading source: ${sourceType} ${sourceArg}`, event);
        alert(`Error loading source: ${sourceType} ${sourceArg}`);
    });
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
    if (!gl.getExtension('OES_texture_float_linear')) {
        console.warn("OES_texture_float_linear is not supported.");
    }
    if (!gl.getExtension("EXT_color_buffer_float")) {
        console.error("EXT_color_buffer_float is not supported on this device.");
    }
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
        const baseMappingData = new Float32Array(params.height.value * params.width.value * 4);
        for (let k = 0; k < params.height.value * params.width.value * 4; k++) baseMappingData[k] = 0;
        gl.activeTexture(gl.TEXTURE4);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, params.width.value, params.height.value, 0, gl.RGBA, gl.FLOAT, baseMappingData);
    });

    document.getElementById("button-screen").addEventListener("click", (event) => {
        const link = document.createElement("a");
        const fileName = `transflow-${parseInt((new Date()) * 1)}.png`;
        link.setAttribute("download", fileName);
        link.href = displayCanvas.toDataURL("image/png");
        link.click();
        showToast(event, `downloaded as ${fileName}`);
    });

    document.getElementById("button-share").addEventListener("click", (event) => {
        writeParamsToUrl(false, false);
        navigator.clipboard.writeText(window.location.href);
        showToast(event, "link copied to clipboard");
    });

    
    var isRecording = false;
    var recordedChunks = [];
    var mediaRecorder = null;

    function startRecording() {
        if (isRecording) return;
        isRecording = true;
        recordedChunks = [];
        return new Promise(function (res, rej) {
            var stream = displayCanvas.captureStream(params.fps.value);
            mediaRecorder = new MediaRecorder(stream, {
                mimeType: "video/webm;codecs:vp9",
                videoBitsPerSecond: 20000000
            });
            mediaRecorder.start(4000);
            mediaRecorder.ondataavailable = function (event) {
                recordedChunks.push(event.data);
            }
            mediaRecorder.onstop = function (event) {
                var blob = new Blob(recordedChunks, {type: "video/webm" });
                var a = document.createElement("a");
                a.setAttribute("download", `transflow-${parseInt((new Date()) * 1)}.webm`);
                a.href = URL.createObjectURL(blob); 
                a.click();
            }
        })
    }

    function stopRecording() {
        if (!isRecording) return;
        isRecording = false;
        mediaRecorder.stop();
    }

    document.getElementById("button-record").addEventListener("click", (event) => {
        if (isRecording) {
            stopRecording();
            event.target.textContent = "record";
        } else {
            startRecording();
            event.target.textContent = "recording…";
            showToast(event, `click again to stop recording`);
        }
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
    const showFlowShader = await createShader(gl, gl.FRAGMENT_SHADER, "shaders/showFlow.frag");
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
    const showFlowProgram = createProgram(gl, vertexShader, showFlowShader);
    const accProgram = createProgram(gl, vertexShader, accShader);
    const feedbackProgram = createProgram(gl, vertexShader, copyShader);
    const remapProgram = createProgram(gl, vertexShader, remapShader);

    const rangeInputs = document.getElementById("range-inputs");
    const booleanInputs = document.getElementById("boolean-inputs");
    bindParamToRangeInput(gl, "windowSize", tINT, [flowPointMatchingProgram, flowLucasKanadeProgram], ["pm", "lk"]);
    bindParamToRangeInput(gl, "gaussianSize", tINT, [flowPointMatchingProgram], ["pm"]);
    
    const feedbackInput = params.feedback.create(rangeInputs);
    const flowDecayInput = bindParamToRangeInput(gl, "flowDecay", tFLOAT, [flowHornSchunckProgram], ["hs"]);
    feedbackInput.classList.add("method-input");
    feedbackInput.setAttribute("method", "hs");
    const iterationsInput = params.iterations.create(rangeInputs);
    iterationsInput.classList.add("method-input");
    iterationsInput.setAttribute("method", "hs");
    function onFeedbackInput(feedbackState) {
        flowDecayInput.style.display = feedbackState == 0 ? "none" : "block";
        iterationsInput.style.display = feedbackState == 1 ? "none" : "block";
    }
    feedbackInput.querySelector("input").addEventListener("input", (event) => {
        onFeedbackInput(event.target.value);
    });
        
    bindParamToRangeInput(gl, "alpha", tFLOAT, [flowHornSchunckProgram], ["hs"]);
    bindParamToRangeInput(gl, "scale", tFLOAT, [accProgram]);
    bindParamToRangeInput(gl, "minimumMovement", tFLOAT, [flowPointMatchingProgram], ["pm"]);
    bindParamToRangeInput(gl, "threshold", tFLOAT, [flowLucasKanadeProgram, flowHornSchunckProgram], ["hs", "lk"]);
    bindParamToRangeInput(gl, "blurSize", tINT, [accProgram]);
    bindParamToRangeInput(gl, "decay", tFLOAT, [accProgram]);
    bindParamToBooleanInput(gl, "flowMirrorX", [flowPointMatchingProgram, flowLucasKanadeProgram, flowHornSchunckProgram]);
    bindParamToBooleanInput(gl, "flowMirrorY", [flowPointMatchingProgram, flowLucasKanadeProgram, flowHornSchunckProgram]);
    bindParamToBooleanInput(gl, "bitmapMirrorX", [remapProgram]);
    bindParamToBooleanInput(gl, "bitmapMirrorY", [remapProgram]);
    params.lockFlow.create(booleanInputs);
    params.lockBitmap.create(booleanInputs);
    params.showFlow.create(booleanInputs);

    binParamToSelect("method", "#select-method", (methodName => {
        document.querySelectorAll(".method-input").forEach(methodInput => {
            if (methodInput.getAttribute("method").includes(methodName)) {
                methodInput.style.display = null;
            } else {
                methodInput.style.display = "none";
            }
        });
    }));
    onFeedbackInput(params.feedback.value);

    const nextTexture = gl.createTexture();
    initializeTexture(gl, nextTexture);

    const pastTexture = gl.createTexture();
    initializeTexture(gl, pastTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, params.flowWidth.value, params.flowHeight.value, 0, gl.RGBA, gl.UNSIGNED_BYTE, createEmptyArray(params.flowWidth.value, params.flowHeight.value));
    const pastFrameBuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, pastFrameBuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, pastTexture, 0);

    const baseFlowData = new Float32Array(params.flowHeight.value * params.flowWidth.value * 4);
    for (let k = 0; k < params.flowHeight.value * params.flowWidth.value * 4; k++) baseFlowData[k] = 0;

    const flowATexture = gl.createTexture();
    initializeTextureFloat(gl, flowATexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, params.flowWidth.value, params.flowHeight.value, 0, gl.RGBA, gl.FLOAT, baseFlowData);
    const flowADepthBuffer = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, flowADepthBuffer);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, params.flowWidth.value, params.flowHeight.value);
    const flowAFrameBuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, flowAFrameBuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, flowATexture, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, flowADepthBuffer);

    const flowBTexture = gl.createTexture();
    initializeTextureFloat(gl, flowBTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, params.flowWidth.value, params.flowHeight.value, 0, gl.RGBA, gl.FLOAT, baseFlowData);
    const flowBDepthBuffer = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, flowBDepthBuffer);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, params.flowWidth.value, params.flowHeight.value);
    const flowBFrameBuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, flowBFrameBuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, flowBTexture, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, flowBDepthBuffer);

    const baseMappingData = new Float32Array(params.height.value * params.width.value * 4);
    for (let k = 0; k < params.height.value * params.width.value * 4; k++) baseMappingData[k] = 0;

    const feedbackTexture = gl.createTexture();
    initializeTextureFloat(gl, feedbackTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, params.width.value, params.height.value, 0, gl.RGBA, gl.FLOAT, baseMappingData);
    const feedbackFrameBuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, feedbackFrameBuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, feedbackTexture, 0);

    const mappingTexture = gl.createTexture();
    initializeTextureFloat(gl, mappingTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, params.width.value, params.height.value, 0, gl.RGBA, gl.FLOAT, baseMappingData);
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

    gl.useProgram(showFlowProgram);
    initializeBuffer(gl);
    bindTexture(gl, showFlowProgram, "sampler", 3, flowATexture);

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

    const flowDecayLocation = gl.getUniformLocation(flowHornSchunckProgram, "flowDecay");

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

        if (params.method.value == "hs" && !params.feedback.value && !params.lockFlow.value) {
            for (let i = 0; i < params.iterations.value; i++) {
                gl.useProgram(flowHornSchunckProgram);
                gl.uniform1f(flowDecayLocation, i == 0 ? 0.0 : 1.0);
                gl.viewport(0, 0, params.flowWidth.value, params.flowHeight.value);
                gl.bindFramebuffer(gl.FRAMEBUFFER, flowAFrameBuffer);
                gl.activeTexture(gl.TEXTURE0);
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, motionSource);
                gl.drawArrays(gl.TRIANGLES, 0, 6);
                gl.useProgram(copyFlowProgram);
                gl.viewport(0, 0, params.flowWidth.value, params.flowHeight.value);
                gl.bindFramebuffer(gl.FRAMEBUFFER, flowBFrameBuffer);
                gl.drawArrays(gl.TRIANGLES, 0, 6);
            }
        } else {
            if (!params.lockFlow.value) {
                gl.useProgram(flowProgram[params.method.value]);
                gl.viewport(0, 0, params.flowWidth.value, params.flowHeight.value);
                gl.bindFramebuffer(gl.FRAMEBUFFER, flowAFrameBuffer);
                gl.activeTexture(gl.TEXTURE0);
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, motionSource);
                gl.drawArrays(gl.TRIANGLES, 0, 6);
            }
            gl.useProgram(copyFlowProgram);
            gl.viewport(0, 0, params.flowWidth.value, params.flowHeight.value);
            gl.bindFramebuffer(gl.FRAMEBUFFER, flowBFrameBuffer);
            gl.drawArrays(gl.TRIANGLES, 0, 6);
        }

        gl.useProgram(pastProgram);
        gl.viewport(0, 0, params.flowWidth.value, params.flowHeight.value);
        gl.bindFramebuffer(gl.FRAMEBUFFER, pastFrameBuffer);
        gl.drawArrays(gl.TRIANGLES, 0, 6);

        if (params.showFlow.value) {
            gl.useProgram(showFlowProgram);
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

    const overlayCanvas = document.getElementById("overlay");
    const overlayContext = overlayCanvas.getContext("2d");
    overlayCanvas.width = params.width.value;
    overlayCanvas.height = params.height.value;

    const overlayData = {
        color: "#fff",
        title: "CAMERA 1",
        zoom: "     2.5",
        azimuth: " SSE 157°",
        lat: " 3.1223°W",
        lon: "45.8722°N",
        infoLeft: "TRANSFLOW",
        infoMid: "SYSTEM STATUS: NOMINAL",
        debugLogs: [
            "[OK] INIT :: Shader compilation successful [VTX:102ms, FRG:98ms]",
            "[DBG] Pipeline sync latency: Δ=+14.7μs",
            "[WRN] Texture #12 dropped: Mipmap inconsistency detected",
            "[SYS] GL_MAX_UNIFORM_BLOCK_SIZE: 65536",
            "[MEM] VRAM usage: 483.2MB / 2048MB (23.58%)",
            "[NET] Ping to telemetry endpoint (192.168.7.22): 2.8ms",
            "[GPU] Render pass completed with 3 draw calls [IDX: 1024]",
            "[I/O] Stream buffer flushed to /dev/gfx0 [BLOCKS: 42]",
            "[TEMP] Core1: 72.4°C | Core2: 71.8°C | GPU: 67.9°C",
            "[TRACE] FrameID: 3249 | Timestamp: 1696423131.089",
            "[SEC] No intrusion detected :: all keys synced",
            "[ENV] Atmosphere sample OK | Ion levels nominal",
            "[DBG] Allocated new transient buffer: UID=0x5A7E, Size=8192B, Align=64",
            "[SYS] Frame pacing adjusted :: vsync delta: -1.23ms",
            "[MEM] Heap fragmentation at 12.8% :: GC cycle postponed",
            "[I/O] Async flush committed to ring buffer [Channel: 3]",
            "[GPU] Shader unit latency stabilized at 37.2μs (±2.1μs)",
            "[WARN] Inferred LOD0 exceeds threshold texture size",
            "[NET] DNS prefetch success :: TTL: 472s",
            "[PHYS] Inertia tensor recalculated [Δ = 0.0034]",
            "[INFO] Spatial resolver converged after 7 iterations",
            "[AUX] Serial bus idle :: Port COM2 queued for wake",
            "[SEC] Entropy pool reseeded with 128-bit salt",
            "[TRACE] Event ID 0x9E14 triggered at T+492ms",
            "[VRAM] Atlas map realigned :: 4096x4096 > 8192x8192",
            "[GEO] Quaternion mismatch resolved via SLERP",
            "[PIPE] Swapchain validated :: Index = 2, Age = 3",
            "[OPT] SIMD vectorization bypassed due to register spill",
            "[CTRL] PID dampening applied :: Overshoot: 2.9%",
            "[MESH] Degenerate triangle culled :: ID#0442A",
            "[DEV] Build fingerprint: alpha-v18+unstable.3124",
            "[MON] Thermal deviation logged :: ΔCoreTemp = +4.3°C",
            "[BUF] Circular queue rewound to index 0x1F",
            "[PRE] Prologue cache prefetch :: Hit rate 98.1%",
            "[TIME] LocalClock: sync drift 2.3ms below UTC",
            "[LOGIC] Execution branch collapsed into single opcode",
            "[RENDER] Deferred lighting pass: 12.6ms (G-buffer)",
            "[HARD] I²C signal degraded :: Noise floor +12dB",
            "[ALERT] Stack trace anomaly at depth=14",
            "[LUX] Photon budget exceeded in forward pass",
            "[QUEUE] Semaphore signaled :: FIFO advanced",
            "[AUTH] Key handshake completed :: ECC-512 OK",
            "[TRACE] Stack ID: #BF33 :: Return @ 0x7FD09",
            "[SND] Audio underrun prevented :: Refill=16KB",
            "[DATA] Packet CRC32 verified :: 0xA42D39B4",
            "[MEM] Zero-page injection in segment 0x0042",
            "[SYS] Critical section exited :: Mutex#001F",
            "[NET] Keepalive ACK received :: seq=1108421",
            "[GPU] Tile cache invalidated :: Region [13:7]",
            "[IOCTL] Driver signal latched :: fd=12, ioctl=0x4004667A",
            "[ALGO] Heuristic threshold adapted: ε=0.00291",
            "[FX] Motion vector blur: ΔFrameID=+1.003ms",
            "[FIRM] ROM patch verified (sig:0xAAFF33CC)",
            "[SCHED] Tick rate throttled :: Load=91.2%",
            "[BIO] Synth impulse decoded :: Envelope: ADSR OK",
            "[NAV] Dead reckoning mode fallback :: GPS jitter",
            "[AUTH] Token refresh cycle initiated :: T-59s",
            "[STATE] Frame lock granted :: V-Sync pulse match",
            "[WARN] Latency buffer near overflow :: 93%",
            "[DEC] Codec pipeline rehydrated :: QP=23, B-frames=2",
            "[GRAPH] Frustum bounds recalibrated [θ=63.2°]",
            "[CACHE] L1-d Cache primed :: Bank[3] HitRate=99.4%",
        ],
    }

    overlayContext.fillStyle = overlayData.color;
    overlayContext.strokeStyle = overlayData.color;
    overlayContext.lineWidth = 2;
    overlayContext.font = "20px monospace";

    let maxDebugLogLength = 0;
    for (let i = 0; i < overlayData.debugLogs.length; i++) {
        maxDebugLogLength = Math.max(maxDebugLogLength, overlayData.debugLogs[i].length);
    }

    let currentLogIndex = 0;
    const maxVisibleLogs = 3;

    function drawOverlay() {
        overlayContext.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        
        // Frame
        overlayContext.strokeRect(20, 20, overlayCanvas.width - 40, overlayCanvas.height - 40);

        // Crosshair
        const cx = overlayCanvas.width / 2;
        const cy = overlayCanvas.height / 2;
        overlayContext.beginPath();
        overlayContext.moveTo(cx - 15, cy);
        overlayContext.lineTo(cx + 15, cy);
        overlayContext.moveTo(cx, cy - 15);
        overlayContext.lineTo(cx, cy + 15);
        overlayContext.stroke();
        
        overlayContext.textAlign = "left";
        // Top-left
        overlayContext.fillText(overlayData.infoLeft, 30, 42);
        // Mid-left

        const barX = 30;
        const barY = overlayCanvas.height / 2 - 100;
        const barWidth = 20;
        const barHeight = 200;
        const filledHeight1 = Math.random() * barHeight;
        overlayContext.fillRect(barX, barY + (barHeight - filledHeight1), barWidth, filledHeight1);
        overlayContext.strokeRect(barX, barY, barWidth, barHeight);
        const filledHeight2 = Math.random() * barHeight;
        overlayContext.fillRect(barX + 30, barY + (barHeight - filledHeight2), barWidth, filledHeight2);
        overlayContext.strokeRect(barX + 30, barY, barWidth, barHeight);

        // Bottom-left
        [
            `FLW: ${params.flowWidth.value}x${params.flowHeight.value}`,
            `BLK: ${params.blockSize.value}`,
            `FPS: ${params.fps.value}`,
            `RES: ${params.width.value}x${params.height.value}`,
        ].forEach((line, i) => {
            overlayContext.fillText(line, 30, overlayCanvas.height - 30 - i * 20);
        }); 

        overlayContext.textAlign = "center";
        // Top-mid
        overlayContext.fillText(overlayData.infoMid, overlayCanvas.width / 2, 42);
        // Bottom-mid
        for (let i = 0; i < maxVisibleLogs; i++) {
            const log = overlayData.debugLogs[(currentLogIndex + i) % overlayData.debugLogs.length];
            overlayContext.textAlign = 'center';
            overlayContext.fillText(log.padEnd(maxDebugLogLength), overlayCanvas.width / 2, overlayCanvas.height - 30 - (maxVisibleLogs - i - 1) * 20);
        }

        overlayContext.textAlign = "right";
        // Top-right
        overlayContext.font = "40px monospace";
        overlayContext.fillText(`${overlayData.title}`, overlayCanvas.width - 30, 56);
        overlayContext.font = "20px monospace";
        // Mid-right
        // Bottom-right     
        const now = new Date();
        const dateStr = `${now.getFullYear()}-${now.getMonth().toString().padStart(2, "0")}-${now.getDate().toString().padStart(2, "0")}`
        const timeStr = now.toLocaleTimeString();
        [
            `${dateStr} ${timeStr}`,
            `LON: ${overlayData.lon}`,
            `LAT: ${overlayData.lat}`,
            `AZM: ${overlayData.azimuth}`,
            `ZOM: ${overlayData.zoom}x`,
        ].forEach((line, i) => {
            overlayContext.fillText(line, overlayCanvas.width - 30, overlayCanvas.height - 30 - i * 20);
        });       
        
    }
    setInterval(() => {
        if (Math.random() < .5) {
            currentLogIndex = (currentLogIndex + 1) % overlayData.debugLogs.length;
        }
        drawOverlay();
    }, 1000);
    drawOverlay();

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
