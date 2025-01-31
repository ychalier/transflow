precision highp float;

uniform sampler2D next;
uniform sampler2D past;
uniform sampler2D inflow;
varying vec2 uv;

uniform float blockSize;
uniform float flowWidth;
uniform float flowHeight;
uniform bool flowMirrorX;
uniform bool flowMirrorY;

const int MAXIMUM_WINDOW_SIZE = 15;
uniform int windowSize;
const int MAXIMUM_GAUSSIAN_SIZE = 15;
uniform int gaussianSize;

uniform float minimumMovement;

float gray(sampler2D texture, vec2 coords) {
    // flip x-axis for the flowMirrorX effect
    if (flowMirrorX) {
        coords.x = 1.0 - coords.x;
    }
    if (!flowMirrorY) {
        coords.y = 1.0 - coords.y;
    }
    vec4 color = texture2D(texture, coords);
    return (color.x + color.y + color.z) / 3.0;
}

float differenceAround(vec2 offset) {
    float difference = 0.0;
    int halfGaussianSize = gaussianSize / 2;
    float total = 0.0;
    for (int i = 0; i < MAXIMUM_GAUSSIAN_SIZE; i++) {
        if (i >= gaussianSize)
            break;
        for (int j = 0; j < MAXIMUM_GAUSSIAN_SIZE; j++) {
            if (j >= gaussianSize)
                break;
            float weight = exp(-0.5 * (pow(float(i - halfGaussianSize), 2.0) + pow(float(j - halfGaussianSize), 2.0)));
            vec2 offset2 = vec2(float(j - halfGaussianSize) / flowWidth, float(i - halfGaussianSize) / flowHeight);
            difference += weight * pow(gray(next, uv + offset2) - gray(past, uv + offset + offset2), 2.0);
            total += weight;
        }
    }
    return difference / total;
}

void main() {
    float minDiff = 16.0;

    vec2 flow;
    vec2 offset = vec2(0.0, 0.0);
    int halfWindowSize = windowSize / 2;

    if(abs(gray(past, uv) - gray(next, uv)) < minimumMovement) {
        flow = vec2(0.0, 0.0);
    } else {
        for(int j = 0; j < MAXIMUM_WINDOW_SIZE; j++) {
            if(j >= windowSize)
                break;
            offset.x = float(j - halfWindowSize) / flowWidth;
            for(int i = 0; i < MAXIMUM_WINDOW_SIZE; i++) {
                if(i >= windowSize)
                    break;
                offset.y = float(i - halfWindowSize) / flowHeight;
                float diff = differenceAround(offset);
                if(diff < minDiff || (abs(diff - minDiff) < 0.005 && i == halfWindowSize && j == halfWindowSize)) {
                    minDiff = diff;
                    flow = offset / blockSize;
                }

            }
        }
    }

    gl_FragColor.xy = flow;

}
