"""The few CUDA runtime calls torch does not expose: graphics interop (OpenGL), external memory /
semaphores (Vulkan), events on raw stream handles. Called through ctypes in the libcudart torch has
already loaded, so streams and the primary context are the same ones torch uses."""
import ctypes as C

import torch

torch.cuda.init()
_path = next(line.split()[-1] for line in open("/proc/self/maps") if "libcudart" in line)
_lib = C.CDLL(_path)

H = C.c_void_p  # stream, event, resource, external memory / semaphore handles
D2H, D2D = 2, 3
NONBLOCKING_STREAM, EVENT_NO_TIMING = 1, 2
GL_REGISTER_WRITE_DISCARD, GL_DEVICE_LIST_ALL = 2, 1
OPAQUE_FD, TIMELINE_SEMAPHORE_FD, DEDICATED = 1, 9, 1


class _Win32(C.Structure):
    _fields_ = [("handle", C.c_void_p), ("name", C.c_void_p)]


class _Handle(C.Union):
    _fields_ = [("fd", C.c_int), ("win32", _Win32)]


class ExternalMemoryHandleDesc(C.Structure):
    _fields_ = [("type", C.c_int), ("handle", _Handle), ("size", C.c_ulonglong), ("flags", C.c_uint)]


class ExternalMemoryBufferDesc(C.Structure):
    _fields_ = [("offset", C.c_ulonglong), ("size", C.c_ulonglong), ("flags", C.c_uint)]


class ExternalSemaphoreHandleDesc(C.Structure):
    _fields_ = [("type", C.c_int), ("handle", _Handle), ("flags", C.c_uint)]


class _Fence(C.Structure):
    _fields_ = [("value", C.c_ulonglong)]


class _NvSci(C.Union):
    _fields_ = [("fence", C.c_void_p), ("reserved", C.c_ulonglong)]


class _SignalMutex(C.Structure):
    _fields_ = [("key", C.c_ulonglong)]


class _WaitMutex(C.Structure):
    _fields_ = [("key", C.c_ulonglong), ("timeoutMs", C.c_uint)]


class _SignalInner(C.Structure):
    _fields_ = [("fence", _Fence), ("nvSciSync", _NvSci), ("keyedMutex", _SignalMutex), ("reserved", C.c_uint * 12)]


class _WaitInner(C.Structure):
    _fields_ = [("fence", _Fence), ("nvSciSync", _NvSci), ("keyedMutex", _WaitMutex), ("reserved", C.c_uint * 10)]


class SignalParams(C.Structure):
    _fields_ = [("params", _SignalInner), ("flags", C.c_uint), ("reserved", C.c_uint * 16)]


class WaitParams(C.Structure):
    _fields_ = [("params", _WaitInner), ("flags", C.c_uint), ("reserved", C.c_uint * 16)]


assert C.sizeof(SignalParams) == 144 and C.sizeof(WaitParams) == 144  # sizeof in driver_types.h

P = C.POINTER
_SIG = {  # name: argument types (ctypes would otherwise truncate 64-bit handles to int)
    "cudaMallocHost": (P(C.c_void_p), C.c_size_t),
    "cudaFreeHost": (C.c_void_p,),
    "cudaFree": (C.c_void_p,),
    "cudaDeviceSynchronize": (),
    "cudaMemcpyAsync": (C.c_void_p, C.c_void_p, C.c_size_t, C.c_int, H),
    "cudaEventCreateWithFlags": (P(H), C.c_uint),
    "cudaEventDestroy": (H,),
    "cudaEventRecord": (H, H),
    "cudaEventSynchronize": (H,),
    "cudaStreamWaitEvent": (H, H, C.c_uint),
    "cudaStreamCreateWithFlags": (P(H), C.c_uint),
    "cudaStreamDestroy": (H,),
    "cudaStreamSynchronize": (H,),
    "cudaGLGetDevices": (P(C.c_uint), P(C.c_int), C.c_uint, C.c_int),
    "cudaGraphicsGLRegisterBuffer": (P(H), C.c_uint, C.c_uint),
    "cudaGraphicsMapResources": (C.c_int, P(H), H),
    "cudaGraphicsResourceGetMappedPointer": (P(C.c_void_p), P(C.c_size_t), H),
    "cudaGraphicsUnmapResources": (C.c_int, P(H), H),
    "cudaGraphicsUnregisterResource": (H,),
    "cudaImportExternalMemory": (P(H), P(ExternalMemoryHandleDesc)),
    "cudaExternalMemoryGetMappedBuffer": (P(C.c_void_p), H, P(ExternalMemoryBufferDesc)),
    "cudaDestroyExternalMemory": (H,),
    "cudaImportExternalSemaphore": (P(H), P(ExternalSemaphoreHandleDesc)),
    "cudaSignalExternalSemaphoresAsync_v2": (P(H), P(SignalParams), C.c_uint, H),
    "cudaWaitExternalSemaphoresAsync_v2": (P(H), P(WaitParams), C.c_uint, H),
    "cudaDestroyExternalSemaphore": (H,),
}


def _bind(name, args):
    fn = getattr(_lib, name)
    fn.argtypes, fn.restype = args, C.c_int
    _lib.cudaGetErrorString.restype = C.c_char_p

    def call(*a):
        err = fn(*a)
        if err:
            raise RuntimeError(f"{name}: {_lib.cudaGetErrorString(err).decode()}")

    return call


fn = type("Cuda", (), {n: staticmethod(_bind(n, a)) for n, a in _SIG.items()})


def stream():
    """Raw handle of torch's current stream of this thread."""
    return torch.cuda.current_stream().cuda_stream


def event():
    e = H()
    fn.cudaEventCreateWithFlags(C.byref(e), EVENT_NO_TIMING)
    return e


class _View:
    def __init__(self, ptr, shape):
        self.__cuda_array_interface__ = {"shape": shape, "typestr": "|u1", "data": (ptr, False), "version": 3,
                                         "strides": None}


def tensor_at(ptr, shape):
    """uint8 torch tensor over device memory torch does not own (no copy)."""
    t = torch.as_tensor(_View(ptr, shape), device="cuda")
    assert t.data_ptr() == ptr, "torch copied the external buffer"
    return t
