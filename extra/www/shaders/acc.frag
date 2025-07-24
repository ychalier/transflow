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

void main() {

    float weight = 1.0 / pow(float(blurSize), 2.0);

    vec2 color = vec2(0.0);

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
            color += texture2D(flowTexture, uv + offset).xy * weight;
        }
    }
    
    vec2 flow = scale * color;

    vec2 u = texture2D(feedbackTexture, uv + flow).xy + flow;
    vec2 v = u - sign(u) * decay * abs(u);
    
    gl_FragColor.xy = v;

}