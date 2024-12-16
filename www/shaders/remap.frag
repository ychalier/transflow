precision highp float;

uniform sampler2D bitmap;
uniform sampler2D mapping;
varying vec2 uv;
uniform bool bitmapMirrorX;
uniform bool bitmapMirrorY;

const float textureScale = 128.0;

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

void main() {
    vec4 u = texture2D(mapping, uv);
    // gl_FragColor = u;
    vec2 v = uv + fromDoubleChannel(u) / 256.0;
    if (bitmapMirrorX) {
        v.x = 1.0 - v.x;
    }
    if (!bitmapMirrorY) {
        v.y = 1.0 - v.y;
    }
    gl_FragColor = texture2D(bitmap, v);    
}