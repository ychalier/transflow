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
uniform float threshold;

const int MAXIMUM_WINDOW_SIZE = 15;
uniform int windowSize;

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

float tr2(mat2 A) {
    return A[0][0] + A[1][1];
}

float det(mat2 A) {
    return (A[0][0] * A[1][1]) - (A[1][0] * A[0][1]);
}

vec2 eigenValues(mat2 A) {
    float m = 0.5 * tr2(A);
    float p = det(A);
    return vec2(m + sqrt(m * m - p), m - sqrt(m * m - p));
}


void main() {
    mat2 AtA = mat2(0.0, 0.0, 0.0, 0.0);
    vec2 Atb = vec2(0.0, 0.0);

    int halfWindowSize = windowSize / 2;

    vec2 offset = vec2(0.0, 0.0);

    for(int j = 0; j < MAXIMUM_WINDOW_SIZE; j++) {
        if(j >= windowSize)
            break;
        offset.x = float(j - halfWindowSize) / flowWidth;
        for(int i = 0; i < MAXIMUM_WINDOW_SIZE; i++) {
            if(i >= windowSize)
                break;
            offset.y = float(i - halfWindowSize) / flowHeight;
            float weight = exp(-0.5 * (pow(float(i - halfWindowSize), 2.0) + pow(float(j - halfWindowSize), 2.0)));
            float I =  gray(next, uv + offset);
            float It = gray(past, uv + offset) - I;
            float Ix = gray(next, uv + offset + vec2(1.0 / flowWidth, 0.0)) - I;
            float Iy = gray(next, uv + offset + vec2(0.0, 1.0 / flowHeight)) - I;
            AtA += weight * mat2(Ix * Ix, Ix * Iy, Ix * Iy, Iy * Iy);
            Atb -= weight * vec2(It * Ix, It * Iy);
        }
    }

    vec2 e = eigenValues(AtA);
    vec2 flow;
    if (e.x > 0.001 && e.y > 0.001) {
        mat2 AtAinv = mat2(AtA[1][1], -AtA[0][1], -AtA[1][0], AtA[0][0]) / (AtA[0][0] * AtA[1][1] - AtA[1][0] * AtA[0][1]);
        flow = AtAinv * Atb;
        flow /= (blockSize * flowWidth, blockSize * flowHeight);
        if (threshold > 0.0 && flow.x * flow.x + flow.y * flow.y < threshold) {
            flow = vec2(0.0);
        }
    } else {
        flow = vec2(0.0);
    }

    gl_FragColor.xy = flow;
}