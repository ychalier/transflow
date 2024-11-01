precision highp float;

uniform sampler2D feedbackTexture;
uniform sampler2D flowTexture;
varying vec2 uv;

uniform float scale;
uniform float decay;
uniform float flowWidth;
uniform float flowHeight;
uniform int blurSize;

const int MAX_BLUR_SIZE = 15;
const float textureScale = 128.0;
const float duMin = 1.0 / textureScale;

/**
 * Converts a vec4 of four values between 0 and 1
 * to a vec2 of two values between ca. -2^15/textureScale and 2^15/textureScale - 1
 */
vec2 fromDoubleChannel(vec4 baseValue) {
    return (
        vec2(
            255.0 * baseValue.x + 256.0 * 255.0 * baseValue.y,
            255.0 * baseValue.z + 256.0 * 255.0 * baseValue.w
        ) - 32768.0) / textureScale;
}


/**
 * Converts a vec2 of two values between -2^15/textureScale and 2^15/textureScale - 1
 * to a vec4 of four values between 0 and 1.
 */
vec4 toDoubleChannel(vec2 baseValue) {
    vec2 scaledAndShiftedValue = (baseValue * textureScale) + 32768.0;
    vec2 tenth = floor(scaledAndShiftedValue / 256.0);
    return vec4(
        scaledAndShiftedValue.x - 256.0 * tenth.x,
        tenth.x,
        scaledAndShiftedValue.y - 256.0 * tenth.y,
        tenth.y
    ) / 255.0;
}


void main() {

    float weight = 1.0 / pow(float(blurSize), 2.0);

    vec4 color = vec4(0.0);

    float halfBlurSize = float(blurSize / 2);
    vec2 offset = vec2(0.0, 0.0);

    for(int j = 0; j < MAX_BLUR_SIZE; j++) {
        if(j >= blurSize)
            break;
        offset.x = (float(j) - halfBlurSize) / flowWidth;
        for(int i = 0; i < MAX_BLUR_SIZE; i++) {
            if(i >= blurSize)
                break;
            offset.y = (float(i) - halfBlurSize) / flowHeight;
            color += texture2D(flowTexture, uv + offset) * weight;
        }
    }
    
    vec2 flow = scale * fromDoubleChannel(color) / 256.0;

    vec2 u = fromDoubleChannel(texture2D(feedbackTexture, uv + flow)) + 256.0 * flow;
    vec2 v = u - sign(u) * max(vec2(duMin, duMin), decay * abs(u));
    
    gl_FragColor = toDoubleChannel(v);

}