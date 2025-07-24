precision highp float;

uniform sampler2D bitmap;
uniform sampler2D mapping;
varying vec2 uv;
uniform bool bitmapMirrorX;
uniform bool bitmapMirrorY;

void main() {
    vec2 u = texture2D(mapping, uv).xy;
    vec2 v = uv + u;
    if (bitmapMirrorX) {
        v.x = 1.0 - v.x;
    }
    if (!bitmapMirrorY) {
        v.y = 1.0 - v.y;
    }
    gl_FragColor = texture2D(bitmap, v);    
}