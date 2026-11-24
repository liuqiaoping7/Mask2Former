"""Dynamic-shape TensorRT engines with several optimization profiles (multi-resolution).

Eyes from VGA to 1080p give model inputs (padded to 32) from 480x640 to 800x1344. One engine
per stage covers all of them with a set of optimization profiles; TensorRT picks kernels per
profile for its `opt` shape, so a narrow profile around a frequent size (the 1080p input,
min = opt = max) runs like the former static engine while wider profiles cover the rest.

Profile sets (PROFILE_SETS): "exact" (default) -- one min = opt = max profile per discrete input
of the official / aligned size policies (768x1344 for 16:9, 800 x 1088..1344 for 4:3 .. 5:3,
768x1152) plus ranged buckets for anything else ("official" 768..800 x 1088..1344, "small"
480..608 lines, "mid" 608..800 lines; native policy); "multires" -- p1080 + the three ranged
buckets; "wide" -- a single profile 480x640 .. 800x1344 (comparison only).
Measured per eye (prof/profile_spike.py): exact profiles are as fast as the former static engine
(768x1344: 64.1 vs 63.6 ms) and 4-6% faster than a ranged bucket at the same shape; a single
wide profile is 3-16% slower than the buckets.

Profiles are given on the model input (H, W); each stage derives its input shapes from it:
  backbone      image (1, 3, H, W)
  pixel decoder res2..res5 (1, C_k, H / s_k, W / s_k), s = 4, 8, 16, 32
  decoder       ms0..ms2 NHWC at 1/32, 1/16, 1/8 and mask features (1, 256, H / 4, W / 4)

TRTRunner runs a plan on torch tensors:
  * picks the narrowest profile containing the input shapes (cached per shape),
  * one execution context per profile (created on first use); all contexts of all engines share one
    activation buffer (USER_MANAGED), grown to the largest profile actually used (not to the largest
    profile of the engine up front: 1080p needs 257 MB, the largest "exact" profile 478 MB),
  * one buffer per output sized for the largest shape of any profile; each call returns
    contiguous views of its first elements, so changing the resolution allocates nothing and
    the output addresses are set once per context; set_input_shape only when the shape changes,
  * calls from different streams are ordered (shared activation memory / outputs).
"""
import os
import time

import tensorrt as trt
import torch

DT = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32, trt.int64: torch.int64}

# model input (padded) H x W: (min, opt, max)
PROFILES = {
    "p1080": ((768, 1344), (768, 1344), (768, 1344)),     # 1080p eyes (and any 16:9 eye >= 800 lines)
    "official": ((768, 1088), (800, 1216), (800, 1344)),  # shortest edge 800 (official test size), 4:3 .. 16:9
    "small": ((480, 640), (544, 960), (608, 1088)),       # native size, 480 .. 608 lines
    "mid": ((608, 800), (704, 1216), (800, 1344)),        # native size, 608 .. 800 lines
}
PROFILE_SETS = {
    "1080p": ("p1080",),
    "official": ("p1080", "official"),
    "multires": ("p1080", "official", "small", "mid"),
    "wide": ("wide",),
}
PROFILES["wide"] = ((480, 640), (704, 1216), (800, 1344))  # one profile for everything (comparison)
# exact (min = opt = max) profiles for the discrete official-policy inputs of 4:3 .. 16:9 eyes:
# shortest edge 800 -> 800 x 1067..1333, padded 800 x 1088..1344; wider eyes -> 768 x 1344 (p1080)
for _w in range(1088, 1344 + 1, 32):
    PROFILES[f"o800x{_w}"] = ((800, _w),) * 3
PROFILES["a768x1152"] = ((768, 1152),) * 3  # size_policy "aligned": 4:3 .. 3:2 eyes
PROFILE_SETS["exact"] = ("p1080", *(f"o800x{w}" for w in range(1088, 1344 + 1, 32)), "a768x1152", "official", "small", "mid")


def with_batch(shapes, b):
    """Replace the leading (batch) dimension of every shape (engines are built for a fixed batch)."""
    return {k: (b, *v[1:]) for k, v in shapes.items()}


def profile_list(name_or_list):
    if isinstance(name_or_list, str):
        return [PROFILES[p] for p in PROFILE_SETS[name_or_list]]
    return list(name_or_list)


def plan_tag(profiles):
    """Short stable tag of a profile list for plan file names."""
    import hashlib

    return hashlib.sha1(repr([tuple(map(tuple, p)) for p in profiles]).encode()).hexdigest()[:8]


def build(onnx_path, plan, profiles, stage_shapes, fp16=True, in_dtypes=None, out_dtypes=None,
          configure=None, workspace_gb=2, plugins=False, batch=1, detailed=False, log=print):
    """Parse onnx_path and build plan with one optimization profile per entry of `profiles`
    (model-input (min, opt, max) HxW); stage_shapes(H, W) -> {input name: shape}.
    configure(net, cfg): extra network / config edits (precision pins)."""
    cache_path = os.path.join(os.path.dirname(plan), "trt_timing.cache")
    t0 = time.time()
    logger = trt.Logger(trt.Logger.WARNING)
    if plugins:
        trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, logger)
    if not parser.parse_from_file(onnx_path):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    for i in range(net.num_inputs):
        t = net.get_input(i)
        if in_dtypes and t.name in in_dtypes:
            t.dtype = in_dtypes[t.name]
    for i in range(net.num_outputs):
        t = net.get_output(i)
        if out_dtypes and t.name in out_dtypes:
            t.dtype = out_dtypes[t.name]
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    else:  # strict FP32 (conversion checks): no TF32 tactics
        cfg.clear_flag(trt.BuilderFlag.TF32)
    if detailed:  # per-layer precision / formats visible to the engine inspector
        cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if configure:
        configure(net, cfg)
    for lo, opt, hi in profiles:
        p = builder.create_optimization_profile()
        s_lo, s_opt, s_hi = (stage_shapes(*x) if batch == 1 else with_batch(stage_shapes(*x), batch) for x in (lo, opt, hi))
        for name in s_lo:
            p.set_shape(name, s_lo[name], s_opt[name], s_hi[name])
        cfg.add_optimization_profile(p)
    cache = cfg.create_timing_cache(open(cache_path, "rb").read() if os.path.exists(cache_path) else b"")
    cfg.set_timing_cache(cache, ignore_mismatch=False)
    ser = builder.build_serialized_network(net, cfg)
    if ser is None:
        raise RuntimeError("TensorRT engine build failed")
    with open(plan, "wb") as f:
        f.write(ser)
    with open(cache_path, "wb") as f:
        f.write(cfg.get_timing_cache().serialize())
    log(f"[trt] built {plan} ({ser.nbytes / 2**20:.0f} MB, {len(profiles)} profiles) in {time.time() - t0:.0f} s")


class _Pool:
    """One activation buffer shared by every context of every engine (stages run in sequence)."""

    def __init__(self):
        self.buf, self.ctxs, self.last = None, [], None

    def ensure(self, size):
        if self.buf is None or self.buf.numel() < size:
            torch.cuda.synchronize()
            self.buf = None  # free the old block first (no old + new peak, no stale cached block)
            torch.cuda.empty_cache()
            self.buf = torch.empty(size, dtype=torch.uint8, device="cuda")
            for ctx in self.ctxs:
                ctx.set_device_memory(self.buf.data_ptr(), self.buf.numel())

    def register(self, ctx):
        ctx.set_device_memory(self.buf.data_ptr(), self.buf.numel())
        self.ctxs.append(ctx)

    def order(self, stream):
        if self.last is not None and self.last != stream:
            stream.wait_stream(self.last)
        self.last = stream


POOL = _Pool()


def release():
    """Drop the shared activation buffer (tests building several engines in one process)."""
    global POOL
    POOL = _Pool()
    torch.cuda.empty_cache()


class TRTRunner:
    def __init__(self, plan, plugins=False):
        self.logger = trt.Logger(trt.Logger.WARNING)
        if plugins:
            trt.init_libnvinfer_plugins(self.logger, "")
        with open(plan, "rb") as f:
            self.engine = e = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        names = [e.get_tensor_name(i) for i in range(e.num_io_tensors)]
        self.inputs = [n for n in names if e.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if e.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.in_dtype = {n: DT[e.get_tensor_dtype(n)] for n in self.inputs}
        self.out_dtype = {n: DT[e.get_tensor_dtype(n)] for n in self.outputs}
        n_prof = e.num_optimization_profiles
        self.ranges = [{n: e.get_tensor_profile_shape(n, p) for n in self.inputs} for p in range(n_prof)]
        self.pool = POOL  # grown when a profile's context is created (not to the largest profile up front)
        # output sizes at every profile's max shapes, from one probe context switched through the
        # profiles; the execution contexts are created on first use (each holds host memory: all
        # 3 x 14 of the "exact" set cost ~2 GB of RSS)
        stream = torch.cuda.current_stream()
        numel = {n: 0 for n in self.outputs}
        probe = e.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
        for p in range(n_prof):
            probe.set_optimization_profile_async(p, stream.cuda_stream)
            stream.synchronize()
            for n in self.inputs:
                probe.set_input_shape(n, self.ranges[p][n][2])
            for n in self.outputs:
                numel[n] = max(numel[n], _vol(probe.get_tensor_shape(n)))
        del probe
        self.numel = numel
        self.buf = None  # allocated on first use
        self.ctx = [None] * n_prof
        self.cur = [None] * n_prof        # input shapes currently set on each context
        self.out_shape = [None] * n_prof
        self.pick_cache = {}
        self.switches = 0  # set_input_shape events (resolution changes), for the multi-res stats

    def profile_for(self, shapes):
        """Narrowest profile (smallest max - min volume) whose range contains all input shapes."""
        if shapes in self.pick_cache:
            return self.pick_cache[shapes]
        best, best_v = None, None
        for p, rng in enumerate(self.ranges):
            ok = all(all(lo <= s <= hi for s, lo, hi in zip(shp, rng[n][0], rng[n][2])) for n, shp in zip(self.inputs, shapes))
            if not ok:
                continue
            v = sum(_vol(rng[n][2]) - _vol(rng[n][0]) for n in self.inputs)
            if best is None or v < best_v:
                best, best_v = p, v
        if best is None:
            raise ValueError(f"no optimization profile covers {dict(zip(self.inputs, shapes))}; ranges {self.ranges}")
        self.pick_cache[shapes] = best
        return best

    def __call__(self, feeds):
        stream = torch.cuda.current_stream()
        self.pool.order(stream)
        xs = [feeds[n].to(self.in_dtype[n]).contiguous() for n in self.inputs]
        shapes = tuple(tuple(x.shape) for x in xs)
        p = self.profile_for(shapes)
        if self.buf is None:
            self.buf = {n: torch.empty(self.numel[n], dtype=self.out_dtype[n], device="cuda") for n in self.outputs}
        ctx = self.ctx[p]
        if ctx is None:
            ctx = self.ctx[p] = self.engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
            ctx.set_optimization_profile_async(p, stream.cuda_stream)
            self.pool.ensure(self.engine.get_device_memory_size_for_profile_v2(p))
            self.pool.register(ctx)
            for n in self.outputs:
                ctx.set_tensor_address(n, self.buf[n].data_ptr())
        if self.cur[p] != shapes:
            for n, s in zip(self.inputs, shapes):
                ctx.set_input_shape(n, s)
            self.cur[p] = shapes
            self.out_shape[p] = {n: tuple(ctx.get_tensor_shape(n)) for n in self.outputs}
            self.switches += 1
        for n, x in zip(self.inputs, xs):
            ctx.set_tensor_address(n, x.data_ptr())
        if not ctx.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        for x in xs:
            x.record_stream(stream)
        # views of the shared output buffers: consume them before the next call of this engine
        return {n: self.buf[n][:_vol(s)].view(s) for n, s in self.out_shape[p].items()}


def _vol(s):
    v = 1
    for d in s:
        v *= d
    return v
