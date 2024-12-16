precision highp float;

uniform sampler2D next;
uniform sampler2D past;
uniform sampler2D inflow;
varying vec2 uv;

uniform float blockSize;
uniform float flowWidth;
uniform float flowHeight;
uniform float flowDecay;
uniform float alpha;
uniform bool flowMirrorX;
uniform bool flowMirrorY;

const float textureScale = 128.0;

float gray(sampler2D texture, vec2 offset) {
    // flip x-axis for the flowMirrorX effect
    vec2 coords = uv + offset / vec2(flowWidth, flowHeight);
    if (flowMirrorX) {
        coords.x = 1.0 - coords.x;
    }
    if (!flowMirrorY) {
        coords.y = 1.0 - coords.y;
    }
    vec4 color = texture2D(texture, coords);
    return dot(color, vec4(1.0, 1.0, 1.0, 0.0)) / 3.0;
}

vec2 fromDoubleChannel(vec4 baseValue) {
    return (
        vec2(
            256.0 * baseValue.x + 256.0 * 255.0 * baseValue.y,
            256.0 * baseValue.z + 256.0 * 255.0 * baseValue.w
        ) - 32768.0) / textureScale;
}

vec2 flowAt(vec2 offset) {
    return flowDecay * fromDoubleChannel(texture2D(inflow, uv + offset / vec2(flowWidth, flowHeight))) / 256.0;
}

/**
 * Converts a vec2 of two values between -2^15/textureScale and 2^15/textureScale - 1
 * to a vec4 of four values between 0 and 1.
 */
vec4 toDoubleChannel(vec2 baseValue) {
    vec2 scaledAndShiftedValue = (baseValue * textureScale) + 32768.0;
    vec2 tenth = floor(scaledAndShiftedValue / 256.0);
    return vec4(scaledAndShiftedValue.x - 256.0 * tenth.x, tenth.x, scaledAndShiftedValue.y - 256.0 * tenth.y, tenth.y) / 255.0;
}

void main() {

    float ex = (gray(next, vec2(1.0, 0.0)) - gray(next, vec2(0.0, 0.0))
              + gray(next, vec2(1.0, 1.0)) - gray(next, vec2(0.0, 1.0))
              + gray(past, vec2(1.0, 0.0)) - gray(past, vec2(0.0, 0.0))
              + gray(past, vec2(1.0, 1.0)) - gray(past, vec2(0.0, 1.0))
              ) / 4.0;
    float ey = (gray(next, vec2(0.0, 1.0)) - gray(next, vec2(0.0, 0.0))
              + gray(next, vec2(1.0, 1.0)) - gray(next, vec2(1.0, 0.0))
              + gray(past, vec2(0.0, 1.0)) - gray(past, vec2(0.0, 0.0))
              + gray(past, vec2(1.0, 1.0)) - gray(past, vec2(1.0, 0.0))
              ) / 4.0;
    float et = (gray(past, vec2(0.0, 0.0)) - gray(next, vec2(0.0, 0.0))
              + gray(past, vec2(1.0, 0.0)) - gray(next, vec2(1.0, 0.0))
              + gray(past, vec2(0.0, 1.0)) - gray(next, vec2(0.0, 1.0))
              + gray(past, vec2(1.0, 1.0)) - gray(next, vec2(1.0, 1.0))
              ) / 4.0;
    vec2 hood = ((flowAt(vec2(-1.0, 0.0)) + flowAt(vec2(1.0, 0.0)) + flowAt(vec2(0.0, -1.0)) + flowAt(vec2(0.0, 1.0))) / 6.0
              + (flowAt(vec2(-1.0, -1.0)) + flowAt(vec2(-1.0, 1.0)) + flowAt(vec2(1.0, -1.0)) + flowAt(vec2(1.0, 1.0))) / 12.0);

    float b = (ex*hood.x + ey*hood.y + et) / (alpha + pow(ex, 2.0) + pow(ey, 2.0));

    vec2 result = hood - vec2(ex * b, ey * b);

    gl_FragColor = toDoubleChannel(256.0 * result / blockSize);


}
