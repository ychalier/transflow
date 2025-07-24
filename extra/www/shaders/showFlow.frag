precision highp float;

uniform sampler2D sampler;
varying vec2 uv;

const float scale = 20.0;

void main() {
    vec2 flow = texture2D(sampler, uv).xy;
    gl_FragColor.x = scale * flow.x + 0.5;
    gl_FragColor.y = scale * flow.y + 0.5;
    gl_FragColor.zw = vec2(1.0, 1.0);
}