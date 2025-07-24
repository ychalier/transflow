precision highp float;

uniform sampler2D sampler;
varying vec2 uv;

void main() {
    gl_FragColor = texture2D(sampler, uv);
}