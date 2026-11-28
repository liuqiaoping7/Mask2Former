#version 450
// Samples the composited SBS frame straight from the buffer the CUDA compose kernel wrote:
// tightly packed BGR bytes, row-major, no texture object and no copy. Bilinear when the window
// size differs from the frame; at 1:1 the weights are 0 and the bytes come through unchanged.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(std430, binding = 0) readonly buffer Frame { uint words[]; };
#ifdef VULKAN
layout(push_constant) uniform Push { ivec2 size; } push;
#define SRC_SIZE push.size
#else
layout(location = 0) uniform ivec2 src_size;
#define SRC_SIZE src_size
#endif

float byte_at(int i) { return float((words[i >> 2] >> ((i & 3) * 8)) & 255u); }

vec3 bgr_at(ivec2 p) {
    p = clamp(p, ivec2(0), SRC_SIZE - 1);
    int i = (p.y * SRC_SIZE.x + p.x) * 3;
    return vec3(byte_at(i), byte_at(i + 1), byte_at(i + 2));
}

void main() {
    vec2 s = uv * vec2(SRC_SIZE) - 0.5;
    ivec2 i = ivec2(floor(s));
    vec2 f = s - vec2(i);
    f = mix(f, step(0.5, f), lessThan(min(f, 1.0 - f), vec2(1e-3)));  // snap interpolation noise at 1:1
    vec3 bgr = mix(mix(bgr_at(i), bgr_at(i + ivec2(1, 0)), f.x),
                   mix(bgr_at(i + ivec2(0, 1)), bgr_at(i + ivec2(1, 1)), f.x), f.y);
    color = vec4(bgr.bgr / 255.0, 1.0);  // buffer is B, G, R
}
