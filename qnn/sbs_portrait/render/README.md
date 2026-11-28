# 结果上屏：CUDA → OpenGL / Vulkan（Python 与 C++）

合成 kernel 直接写进图形 API 的缓冲区，片元着色器读同一块显存；推理是生产者线程，主线程是消费者。
原理与两种 API 的实现说明见 QNN 仓库 `docs/Mask2Former-sbs-portrait-opt.md` 第十四节。

## 文件（建议按此顺序读）

| 内容 | Python（`render/`） | C++（`cpp/render/`） |
| --- | --- | --- |
| 环形队列 | `ring.py` `SlotQueue` | `queue.hpp` |
| 生产者接口、渲染循环、校验与统计 | `ring.py` `Presenter` | `cpp/src/sink.hpp`、`renderer.hpp`、`presenter.cpp` |
| 着色器（两种语言共用） | `shaders/sbs.vert`、`shaders/sbs.frag` | 同左 |
| OpenGL 后端 | `gl_renderer.py` | `gl_renderer.cpp` |
| Vulkan 后端 | `vk_renderer.py` | `vk_renderer.cpp` |
| CUDA 互操作接口 | `cuda_rt.py`（ctypes） | 直接用 CUDA 运行时 |
| 只测渲染路径 | `render_bench.py` | `render_bench.cpp` |

## 使用

```bash
render/setup_deps.sh          # 开发包、GLFW、校验层、Python 绑定解到 ~/data/sbs/render（无需 root）
cpp/build_deps.sh             # 有上一步的产物时 C++ 版带渲染支持
export DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority     # 从 ssh 指向本机桌面会话

python run_pipeline.py --frames DIR --bg BG.jpg --out OUT --render gl|vulkan
~/data/sbs/cpp/build/sbs_portrait --assets A --frames DIR --bg BG.jpg --out OUT --render gl|vulkan
```

| 选项 | 作用 |
| --- | --- |
| `--present zerocopy\|copy` | `copy` = 经主机内存的传统路径（对照） |
| `--queue block\|drop` | 队列满：生产者等待 / 丢最旧帧 |
| `--slots N`、`--vsync 0\|1`、`--window WxH` | 缓冲槽数、垂直同步、窗口大小 |
| `--render-check N` | 前 N 帧回读比对，关键帧另存 `render_XXXX.png` |
| `--validate` | Vulkan 校验层（`VK_LAYER_PATH=~/data/sbs/render/sysroot/usr/share/vulkan/explicit_layer.d`，`LD_LIBRARY_PATH` 加 sysroot 的 lib） |

`bench.sh` 复现测量，结果见 `../results/mr/render/LOG.md`。
