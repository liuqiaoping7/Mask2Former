"""OpenGL 4.5 backend (cpp/render/gl_renderer.cpp), PyOpenGL.
Per slot a shader storage buffer registered with CUDA. mapped -> CUDA may write; unmapped -> OpenGL
may read. Only the consumer thread has the GL context, so it maps / unmaps; the producer orders its
stream against those operations with CUDA events, without a CPU wait."""
import ctypes as C
import os

import numpy as np

from . import SHADERS
from .cuda_rt import GL_DEVICE_LIST_ALL, GL_REGISTER_WRITE_DISCARD, H, NONBLOCKING_STREAM, event, fn

import glfw
from OpenGL import GL


class _Slot:
    res = ptr = written = mapped = None
    timed = False


class GLBackend:
    @staticmethod
    def window_hints():
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    def __init__(self, win, width, height, slots, zerocopy, vsync, check, validate):
        self.win, self.w, self.h, self.zerocopy = win, width, height, zerocopy
        self.bytes = width * height * 3
        glfw.make_context_current(win)
        glfw.swap_interval(1 if vsync else 0)
        n, dev = C.c_uint(), C.c_int()
        fn.cudaGLGetDevices(C.byref(n), C.byref(dev), 1, GL_DEVICE_LIST_ALL)  # fails on a non-CUDA GPU
        self.description = (f"OpenGL {GL.glGetString(GL.GL_VERSION).decode()} on "
                            f"{GL.glGetString(GL.GL_RENDERER).decode()}")
        self.prog = self._program()
        self.vao = GL.glGenVertexArrays(1)  # empty: the core profile needs one bound to draw
        GL.glDisable(GL.GL_DITHER)
        self.map_stream = H()
        fn.cudaStreamCreateWithFlags(C.byref(self.map_stream), NONBLOCKING_STREAM)
        self.slots = []
        for _ in range(slots):
            s = _Slot()
            s.buf = int(GL.glGenBuffers(1))
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, s.buf)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, self.bytes, None, GL.GL_DYNAMIC_DRAW)
            q = np.zeros(1, np.uint32)
            GL.glGenQueries(1, q)
            s.query = int(q[0])
            if zerocopy:
                s.res = H()
                fn.cudaGraphicsGLRegisterBuffer(C.byref(s.res), s.buf, GL_REGISTER_WRITE_DISCARD)
                s.written, s.mapped = event(), event()
                self._map(s)  # start out owned by CUDA
            self.slots.append(s)
        self.fbo = self.rbo = 0
        if check:  # offscreen target at source size for the read-back check
            self.fbo, self.rbo = GL.glGenFramebuffers(1), GL.glGenRenderbuffers(1)
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.rbo)
            GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGBA8, width, height)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
            GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, self.rbo)
            assert GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        self.gpu_draw_ms = None

    def _program(self):
        from OpenGL.GL import shaders

        src = lambda name: open(os.path.join(SHADERS, name)).read()  # noqa: E731
        return shaders.compileProgram(shaders.compileShader(src("sbs.vert"), GL.GL_VERTEX_SHADER),
                                      shaders.compileShader(src("sbs.frag"), GL.GL_FRAGMENT_SHADER), validate=False)

    def _map(self, s):
        # GL commands issued before this call finish before CUDA work queued after it on map_stream
        fn.cudaGraphicsMapResources(1, C.byref(s.res), self.map_stream)
        ptr, n = C.c_void_p(), C.c_size_t()
        fn.cudaGraphicsResourceGetMappedPointer(C.byref(ptr), C.byref(n), s.res)  # may move between maps
        s.ptr = ptr.value
        fn.cudaEventRecord(s.mapped, self.map_stream)

    # ------------------------------------------------------------------ producer thread (CUDA only)
    def cuda_begin(self, i, stream):
        s = self.slots[i]
        fn.cudaStreamWaitEvent(stream, s.mapped, 0)  # after the map, hence after OpenGL's reads
        return s.ptr

    def cuda_end(self, i, stream):
        fn.cudaEventRecord(self.slots[i].written, stream)

    # ------------------------------------------------------------------ consumer thread
    def draw(self, i, host, rgba):
        s = self.slots[i]
        if self.zerocopy:
            fn.cudaStreamWaitEvent(self.map_stream, s.written, 0)
            fn.cudaGraphicsUnmapResources(1, C.byref(s.res), self.map_stream)  # OpenGL sees the writes
        else:
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, s.buf)
            GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, self.bytes, host)  # the upload zero-copy avoids
        GL.glUseProgram(self.prog)
        GL.glUniform2i(0, self.w, self.h)
        GL.glBindVertexArray(self.vao)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 0, s.buf)
        if rgba is not None:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
            GL.glViewport(0, 0, self.w, self.h)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
            GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
            data = GL.glReadPixels(0, 0, self.w, self.h, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
            rgba[:] = np.frombuffer(data, np.uint8).reshape(self.h, self.w, 4)[::-1]  # bottom row first
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        if s.timed:  # result of the previous use of this slot: long finished
            ns = C.c_uint64()
            GL.glGetQueryObjectui64v(s.query, GL.GL_QUERY_RESULT, C.byref(ns))
            self.gpu_draw_ms = ns.value * 1e-6
        w, h = glfw.get_framebuffer_size(self.win)
        GL.glViewport(0, 0, w, h)
        GL.glClearColor(0, 0, 0, 1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        k = min(w / self.w, h / self.h)  # letterbox
        vw, vh = int(self.w * k), int(self.h * k)
        GL.glViewport((w - vw) // 2, (h - vh) // 2, vw, vh)
        GL.glBeginQuery(GL.GL_TIME_ELAPSED, s.query)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glEndQuery(GL.GL_TIME_ELAPSED)
        s.timed = True
        glfw.swap_buffers(self.win)
        if self.zerocopy:
            self._map(s)  # hand the slot back to CUDA once OpenGL's draw has read it

    def close(self):
        for s in self.slots:
            if s.res:
                fn.cudaGraphicsUnmapResources(1, C.byref(s.res), self.map_stream)
                fn.cudaGraphicsUnregisterResource(s.res)
                fn.cudaEventDestroy(s.written)
                fn.cudaEventDestroy(s.mapped)
            GL.glDeleteBuffers(1, [s.buf])
        if self.fbo:
            GL.glDeleteFramebuffers(1, [self.fbo])
            GL.glDeleteRenderbuffers(1, [self.rbo])
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteProgram(self.prog)
        fn.cudaStreamDestroy(self.map_stream)
