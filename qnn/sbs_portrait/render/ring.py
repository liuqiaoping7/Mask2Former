"""Producer-consumer half of the renderer, independent of the graphics API (cpp/render/queue.hpp,
presenter.cpp).

    producer thread (inference)              consumer (main) thread
    dst = sink.begin()      take a slot      take a ready slot
    <kernel writes dst>                      draw it (OpenGL / Vulkan), present
    sink.end(frame)         publish it       release it
"""
import collections
import ctypes as C
import threading
import time

import numpy as np

from . import SHADERS  # noqa: F401  (adds the binding directory to sys.path)
from .cuda_rt import D2H, event, fn, stream, tensor_at

import glfw


class SlotQueue:
    """Bounded queue of slot indices. A slot is always in exactly one place: free list -> producer ->
    ready FIFO -> consumer -> free list. drop_oldest: a producer without a free slot takes back the
    oldest frame not yet rendered; otherwise it blocks (back-pressure)."""

    def __init__(self, n, drop_oldest):
        self.free, self.ready = collections.deque(range(n)), collections.deque()
        self.drop, self.finished, self.closed, self.dropped = drop_oldest, False, False, 0
        self.cv = threading.Condition()

    def acquire(self):
        with self.cv:
            self.cv.wait_for(lambda: self.closed or self.free or (self.drop and self.ready))
            if self.closed:
                return -1
            if self.free:
                return self.free.popleft()
            self.dropped += 1
            return self.ready.popleft()

    def publish(self, i):
        with self.cv:
            self.ready.append(i)
            self.cv.notify_all()

    def take(self, timeout):
        """Next ready slot; None on timeout; -2 when the producer finished and all is drained."""
        with self.cv:
            self.cv.wait_for(lambda: self.ready or self.finished or self.closed, timeout)
            if self.ready:
                return self.ready.popleft()
            return -2 if self.finished or self.closed else None

    def release(self, i):
        with self.cv:
            self.free.append(i)
            self.cv.notify_all()

    def finish(self):
        with self.cv:
            self.finished = True
            self.cv.notify_all()

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify_all()


def _pinned(n):
    p = C.c_void_p()
    fn.cudaMallocHost(C.byref(p), n)
    return p, np.ctypeslib.as_array((C.c_uint8 * n).from_address(p.value))


class Presenter:
    """The window, a backend (OpenGL / Vulkan) and the slot queue. Construct and run() on the main
    thread; begin() / end() are the pipeline's sink and run on the producer thread."""

    def __init__(self, api, width, height, slots=3, zerocopy=True, vsync=True, drop=False, window=None,
                 check=0, validate=False):
        if not glfw.init():
            raise RuntimeError("glfw.init failed (is DISPLAY set?)")
        self.api, self.w, self.h, self.zerocopy, self.vsync, self.drop = api, width, height, zerocopy, vsync, drop
        self.n_slots, self.check, self.bytes = slots, check, width * height * 3
        ww = window[0] if window else min(width, 1920)
        wh = window[1] if window else ww * height // width
        glfw.default_window_hints()
        if api == "gl":
            from .gl_renderer import GLBackend as Backend
        elif api == "vulkan":
            from .vk_renderer import VKBackend as Backend
        else:
            raise ValueError(api)
        Backend.window_hints()
        title = f"SBS portrait (python) - {api} {'zero-copy' if zerocopy else 'host copy'}"
        self.win = glfw.create_window(ww, wh, title, None, None)
        if not self.win:
            raise RuntimeError("cannot create a window")
        self.backend = Backend(self.win, width, height, slots, zerocopy, vsync, check > 0, validate)
        self.q = SlotQueue(slots, drop)
        self.host = [None if zerocopy else _pinned(self.bytes) for _ in range(slots)]  # copy mode
        self.expect = [_pinned(self.bytes) if check else None for _ in range(slots)]  # correctness check
        self.ready = [event() for _ in range(slots)]
        self.published, self.frame, self.checking = [0.0] * slots, [0] * slots, [False] * slots
        self.views, self.cur, self.dst, self.produced = {}, -1, None, 0
        self.closed = False
        self.wait_ms, self.latency_ms, self.draw_ms, self.gpu_ms = [], [], [], []
        self.presented = self.checked = self.check_max = self.check_bad = 0
        self.on_checked = None  # callback(frame index, rgba (H, 2W, 4))

    # ------------------------------------------------------------------ producer thread
    def begin(self):
        """Returns the uint8 (H, 2W, 3) tensor to write into, or None (copy mode: use your own).
        Sets self.closed when the window is gone."""
        t0 = time.perf_counter()
        self.cur = self.q.acquire()
        self.wait_ms.append((time.perf_counter() - t0) * 1e3)
        if self.cur < 0:
            self.closed = True
            return None
        self.dst = None
        if self.zerocopy:
            ptr = self.backend.cuda_begin(self.cur, stream())
            if ptr not in self.views:
                self.views[ptr] = tensor_at(ptr, (self.h, self.w, 3))
            self.dst = self.views[ptr]
        return self.dst

    def end(self, frame):
        """frame: the (H, 2W, 3) tensor holding the result (the one from begin() in zero-copy mode)."""
        i, s = self.cur, stream()
        if self.dst is not None and frame.data_ptr() != self.dst.data_ptr():
            self.dst.copy_(frame)  # a compositor that returns its own tensor: one device copy
            frame = self.dst
        self.frame[i] = self.produced
        self.produced += 1
        self.checking[i] = self.frame[i] < self.check
        if self.checking[i]:
            fn.cudaMemcpyAsync(self.expect[i][0], frame.data_ptr(), self.bytes, D2H, s)
        if not self.zerocopy:
            fn.cudaMemcpyAsync(self.host[i][0], frame.data_ptr(), self.bytes, D2H, s)
        if self.checking[i] or not self.zerocopy:
            fn.cudaEventRecord(self.ready[i], s)
        if self.zerocopy:
            self.backend.cuda_end(i, s)
        self.published[i] = time.perf_counter()
        self.q.publish(i)

    def finish(self):
        self.q.finish()

    # ------------------------------------------------------------------ consumer thread
    def run(self):
        rgba = np.empty((self.h, self.w, 4), np.uint8) if self.check else None
        while not glfw.window_should_close(self.win):
            glfw.poll_events()
            if glfw.get_key(self.win, glfw.KEY_ESCAPE) == glfw.PRESS or glfw.get_key(self.win, glfw.KEY_Q) == glfw.PRESS:
                break
            i = self.q.take(0.01)
            if i == -2:
                break
            if i is None:
                continue
            if self.checking[i] or not self.zerocopy:
                fn.cudaEventSynchronize(self.ready[i])
            t0 = time.perf_counter()
            self.backend.draw(i, None if self.zerocopy else self.host[i][1], rgba if self.checking[i] else None)
            t1 = time.perf_counter()
            if self.checking[i]:  # the buffer is BGR, the read-back image R G B A
                exp = self.expect[i][1].reshape(self.h, self.w, 3)
                diff = np.abs(rgba[..., 2::-1].astype(np.int16) - exp)
                self.check_max = max(self.check_max, int(diff.max()))
                self.check_bad += int(np.count_nonzero(diff))
                self.checked += 1
                if self.on_checked:
                    self.on_checked(self.frame[i], rgba)
            else:
                self.latency_ms.append((t1 - self.published[i]) * 1e3)
                self.draw_ms.append((t1 - t0) * 1e3)
            if self.backend.gpu_draw_ms is not None:
                self.gpu_ms.append(self.backend.gpu_draw_ms)
            self.presented += 1
            self.q.release(i)
        self.q.close()

    def stats(self):
        med = lambda v: float(np.median(v)) if v else 0.0  # noqa: E731
        p95 = lambda v: float(np.percentile(v, 95)) if v else 0.0  # noqa: E731
        return {"api": self.api, "present": "zerocopy" if self.zerocopy else "copy", "vsync": int(self.vsync),
                "queue": "drop" if self.drop else "block", "slots": self.n_slots, "produced": self.produced,
                "presented": self.presented, "dropped": self.q.dropped,
                "acquire_wait_ms": {"median": med(self.wait_ms), "p95": p95(self.wait_ms)},
                "latency_ms": {"median": med(self.latency_ms), "p95": p95(self.latency_ms)},
                "draw_cpu_ms": med(self.draw_ms), "draw_gpu_ms": med(self.gpu_ms), "checked": self.checked,
                "check_max_diff": self.check_max, "check_diff_values": self.check_bad}

    @property
    def description(self):
        return self.backend.description

    def close(self):
        fn.cudaDeviceSynchronize()
        self.views.clear()
        self.backend.close()
        for h in self.host + self.expect:
            if h:
                fn.cudaFreeHost(h[0])
        for e in self.ready:
            fn.cudaEventDestroy(e)
        glfw.destroy_window(self.win)
        glfw.terminate()


def run_decoupled(presenter, produce):
    """Run produce() in a producer thread and the render loop here; returns produce()'s result."""
    box = {}

    def worker():
        try:
            box["result"] = produce()
        except BaseException as e:  # noqa: B902  (re-raised on the main thread)
            box["error"] = e
        finally:
            presenter.finish()

    t = threading.Thread(target=worker, name="producer")
    t.start()
    try:
        presenter.run()
    finally:
        presenter.q.close()
        t.join()
    if "error" in box:
        raise box["error"]
    return box["result"]
