#version 450
// One triangle that covers the viewport; no vertex buffer. Shared by the OpenGL and Vulkan backends.
#ifdef VULKAN
#define VERTEX_ID gl_VertexIndex
#else
#define VERTEX_ID gl_VertexID
#endif
layout(location = 0) out vec2 uv;  // (0, 0) = top-left of the source image

void main() {
    vec2 p = vec2((VERTEX_ID << 1) & 2, VERTEX_ID & 2);  // (0,0) (2,0) (0,2)
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
#ifdef VULKAN
    uv = p;                      // Vulkan: clip y = -1 is the top of the framebuffer
#else
    uv = vec2(p.x, 1.0 - p.y);   // OpenGL: clip y = -1 is the bottom
#endif
}
