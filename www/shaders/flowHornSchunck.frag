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

vec2 flowAt(vec2 offset) {
    return flowDecay * texture2D(inflow, uv + offset / vec2(flowWidth, flowHeight)).xy;
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

    // gl_FragColor = toDoubleChannel(256.0 * result / blockSize);
    gl_FragColor.xy = result / blockSize;

}
