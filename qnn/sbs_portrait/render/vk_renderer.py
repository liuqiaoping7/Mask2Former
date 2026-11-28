"""Vulkan 1.3 backend (cpp/render/vk_renderer.cpp), `vulkan` (cffi) package, dynamic rendering.
Per slot a VkBuffer whose memory is exported as a POSIX fd and imported by CUDA, plus a timeline
semaphore exported the same way. Ownership moves on that semaphore, entirely on the GPU:
    CUDA waits  value r (Vulkan's last release) -> writes -> signals w = counter + 1   (producer)
    Vulkan waits w at the fragment shader       -> draws  -> signals counter + 2       (consumer)
No map / unmap: the CUDA device pointer stays valid for the lifetime of the buffer."""
import ctypes as C
import hashlib
import os
import subprocess
import sys

import numpy as np
import torch

from . import DEPS, SHADERS
from . import cuda_rt as cu

import glfw
import vulkan as vk

FORMAT = vk.VK_FORMAT_B8G8R8A8_UNORM  # UNORM: bytes are shown as they are (no sRGB re-encoding)
EXTERNAL = vk.VK_QUEUE_FAMILY_EXTERNAL
IGNORED = vk.VK_QUEUE_FAMILY_IGNORED
WHOLE = 2**64 - 1  # VK_WHOLE_SIZE (the binding defines it as -1)


def compile_spirv(name):
    """GLSL -> SPIR-V with glslangValidator (cached by source hash)."""
    src = os.path.join(SHADERS, name)
    key = hashlib.sha1(open(src, "rb").read()).hexdigest()[:12]
    out = os.path.join(DEPS, "spv", f"{name}.{key}.spv")
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        root = os.path.join(DEPS, "sysroot", "usr")
        env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(root, "lib", "x86_64-linux-gnu"))
        subprocess.run([os.path.join(root, "bin", "glslangValidator"), "-V", "--target-env", "vulkan1.3", "-o", out, src],
                       check=True, env=env, stdout=subprocess.DEVNULL)
    return open(out, "rb").read()


@vk.ffi.callback("PFN_vkDebugUtilsMessengerCallbackEXT")
def _on_validation(severity, kind, data, user):
    print("VALIDATION:", vk.ffi.string(data.pMessage).decode(), file=sys.stderr, flush=True)
    return 0


class _Slot:
    buf = mem = staging = staging_mem = mapped = timeline = None
    cuda_mem = cuda_sem = None
    ptr = 0
    counter = written = released = 0  # timeline values
    timed = False


class VKBackend:
    @staticmethod
    def window_hints():
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)

    def __init__(self, win, width, height, slots, zerocopy, vsync, check, validate):
        self.win, self.w, self.h, self.n, self.zerocopy, self.vsync = win, width, height, slots, zerocopy, vsync
        self.bytes = width * height * 3
        self.gpu_draw_ms = None
        self._instance(validate)
        surface = vk.ffi.new("VkSurfaceKHR[1]")
        if glfw.create_window_surface(self.inst, win, None, surface) != vk.VK_SUCCESS:
            raise RuntimeError("glfwCreateWindowSurface failed")
        self.surface = surface[0]
        self._pick_device()
        self._device()
        self._swapchain()
        self._pipeline()
        self._slots()
        self.check_img = None
        if check:
            self._check_target()

    def _ifn(self, name):
        return vk.vkGetInstanceProcAddr(self.inst, name)

    def _dfn(self, name):
        return vk.vkGetDeviceProcAddr(self.dev, name)

    # ------------------------------------------------------------------ setup
    def _instance(self, validate):
        ext = list(glfw.get_required_instance_extensions())
        layers = []
        if validate:
            ext.append(vk.VK_EXT_DEBUG_UTILS_EXTENSION_NAME)
            layers.append("VK_LAYER_KHRONOS_validation")
        app = vk.VkApplicationInfo(apiVersion=vk.VK_MAKE_VERSION(1, 3, 0))
        self.inst = vk.vkCreateInstance(vk.VkInstanceCreateInfo(
            pApplicationInfo=app, enabledExtensionCount=len(ext), ppEnabledExtensionNames=ext,
            enabledLayerCount=len(layers), ppEnabledLayerNames=layers), None)
        self.messenger = None
        if validate:
            info = vk.VkDebugUtilsMessengerCreateInfoEXT(
                messageSeverity=vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                | vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                messageType=vk.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                | vk.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT, pfnUserCallback=_on_validation)
            self.messenger = self._ifn("vkCreateDebugUtilsMessengerEXT")(self.inst, info, None)

    def _pick_device(self):  # the physical device that is the CUDA device (same UUID)
        want = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid).replace("-", "")
        self.phys = None
        for d in vk.vkEnumeratePhysicalDevices(self.inst):
            ids = vk.ffi.new("VkPhysicalDeviceIDProperties*",
                             {"sType": vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES})
            p2 = vk.ffi.new("VkPhysicalDeviceProperties2*",
                            {"sType": vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, "pNext": ids})
            vk.vkGetPhysicalDeviceProperties2(d, p2)
            if bytes(vk.ffi.buffer(ids.deviceUUID)).hex() == want:
                self.phys = d
                p = p2.properties
                self.period_ns = p.limits.timestampPeriod
                self.description = (f"Vulkan {p.apiVersion >> 22}.{(p.apiVersion >> 12) & 1023} on "
                                    f"{vk.ffi.string(p.deviceName).decode()}")
        if self.phys is None:
            raise RuntimeError("no Vulkan device matches the CUDA device")
        support = self._ifn("vkGetPhysicalDeviceSurfaceSupportKHR")
        for i, q in enumerate(vk.vkGetPhysicalDeviceQueueFamilyProperties(self.phys)):
            if q.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT and support(self.phys, i, self.surface):
                self.family = i
                return
        raise RuntimeError("no graphics + present queue family")

    def _device(self):
        q = vk.VkDeviceQueueCreateInfo(queueFamilyIndex=self.family, queueCount=1, pQueuePriorities=[1.0])
        ext = ["VK_KHR_swapchain", "VK_KHR_external_memory_fd", "VK_KHR_external_semaphore_fd"]
        f13 = vk.VkPhysicalDeviceVulkan13Features(dynamicRendering=1)
        f12 = vk.VkPhysicalDeviceVulkan12Features(pNext=f13, timelineSemaphore=1)
        self.dev = vk.vkCreateDevice(self.phys, vk.VkDeviceCreateInfo(
            pNext=f12, queueCreateInfoCount=1, pQueueCreateInfos=[q], enabledExtensionCount=len(ext),
            ppEnabledExtensionNames=ext), None)
        self.queue = vk.vkGetDeviceQueue(self.dev, self.family, 0)
        self.pool = vk.vkCreateCommandPool(self.dev, vk.VkCommandPoolCreateInfo(
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, queueFamilyIndex=self.family), None)
        self.queries = vk.vkCreateQueryPool(self.dev, vk.VkQueryPoolCreateInfo(
            queryType=vk.VK_QUERY_TYPE_TIMESTAMP, queryCount=2 * self.n), None)
        self.acquire_fn = self._dfn("vkAcquireNextImageKHR")
        self.present_fn = self._dfn("vkQueuePresentKHR")

    def _view(self, img):
        return vk.vkCreateImageView(self.dev, vk.VkImageViewCreateInfo(
            image=img, viewType=vk.VK_IMAGE_VIEW_TYPE_2D, format=FORMAT,
            subresourceRange=vk.VkImageSubresourceRange(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, levelCount=1,
                                                        layerCount=1)), None)

    def _swapchain(self):
        caps = self._ifn("vkGetPhysicalDeviceSurfaceCapabilitiesKHR")(self.phys, self.surface)
        w, h = caps.currentExtent.width, caps.currentExtent.height
        if w == 0xFFFFFFFF:
            w, h = glfw.get_framebuffer_size(self.win)
        self.extent = (w, h)
        modes = list(self._ifn("vkGetPhysicalDeviceSurfacePresentModesKHR")(self.phys, self.surface))
        mode = vk.VK_PRESENT_MODE_FIFO_KHR  # vsync
        if not self.vsync:
            mode = vk.VK_PRESENT_MODE_IMMEDIATE_KHR if vk.VK_PRESENT_MODE_IMMEDIATE_KHR in modes \
                else vk.VK_PRESENT_MODE_MAILBOX_KHR
        count = caps.minImageCount + 1
        if caps.maxImageCount:
            count = min(count, caps.maxImageCount)
        self.swapchain = self._dfn("vkCreateSwapchainKHR")(self.dev, vk.VkSwapchainCreateInfoKHR(
            surface=self.surface, minImageCount=count, imageFormat=FORMAT,
            imageColorSpace=vk.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, imageExtent=vk.VkExtent2D(w, h), imageArrayLayers=1,
            imageUsage=vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, preTransform=caps.currentTransform,
            compositeAlpha=vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, presentMode=mode, clipped=1), None)
        self.images = list(self._dfn("vkGetSwapchainImagesKHR")(self.dev, self.swapchain))
        self.views = [self._view(i) for i in self.images]
        self.done = [vk.vkCreateSemaphore(self.dev, vk.VkSemaphoreCreateInfo(), None) for _ in self.images]

    def _destroy_swapchain(self):
        for v in self.views:
            vk.vkDestroyImageView(self.dev, v, None)
        for s in self.done:
            vk.vkDestroySemaphore(self.dev, s, None)
        self._dfn("vkDestroySwapchainKHR")(self.dev, self.swapchain, None)

    def _recreate_swapchain(self):  # window resized
        vk.vkDeviceWaitIdle(self.dev)
        self._destroy_swapchain()
        self._swapchain()

    def _pipeline(self):
        stage_frag = vk.VK_SHADER_STAGE_FRAGMENT_BIT
        binding = vk.VkDescriptorSetLayoutBinding(
            binding=0, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=1, stageFlags=stage_frag)
        self.dlayout = vk.vkCreateDescriptorSetLayout(
            self.dev, vk.VkDescriptorSetLayoutCreateInfo(bindingCount=1, pBindings=[binding]), None)
        self.layout = vk.vkCreatePipelineLayout(self.dev, vk.VkPipelineLayoutCreateInfo(
            setLayoutCount=1, pSetLayouts=[self.dlayout], pushConstantRangeCount=1,
            pPushConstantRanges=[vk.VkPushConstantRange(stageFlags=stage_frag, offset=0, size=8)]), None)
        modules = []
        for name in ("sbs.vert", "sbs.frag"):
            code = compile_spirv(name)
            modules.append(vk.vkCreateShaderModule(
                self.dev, vk.VkShaderModuleCreateInfo(codeSize=len(code), pCode=code), None))
        stages = [vk.VkPipelineShaderStageCreateInfo(stage=s, module=m, pName="main")
                  for s, m in zip((vk.VK_SHADER_STAGE_VERTEX_BIT, stage_frag), modules)]
        # everything OpenGL keeps as mutable context state is baked into the pipeline object here
        dyn = [vk.VK_DYNAMIC_STATE_VIEWPORT, vk.VK_DYNAMIC_STATE_SCISSOR]
        info = vk.VkGraphicsPipelineCreateInfo(
            pNext=vk.VkPipelineRenderingCreateInfo(colorAttachmentCount=1, pColorAttachmentFormats=[FORMAT]),
            stageCount=2, pStages=stages,
            pVertexInputState=vk.VkPipelineVertexInputStateCreateInfo(),
            pInputAssemblyState=vk.VkPipelineInputAssemblyStateCreateInfo(topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST),
            pViewportState=vk.VkPipelineViewportStateCreateInfo(viewportCount=1, scissorCount=1),
            pRasterizationState=vk.VkPipelineRasterizationStateCreateInfo(
                polygonMode=vk.VK_POLYGON_MODE_FILL, cullMode=vk.VK_CULL_MODE_NONE, lineWidth=1.0),
            pMultisampleState=vk.VkPipelineMultisampleStateCreateInfo(rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT),
            pColorBlendState=vk.VkPipelineColorBlendStateCreateInfo(
                attachmentCount=1, pAttachments=[vk.VkPipelineColorBlendAttachmentState(colorWriteMask=0xF)]),
            pDynamicState=vk.VkPipelineDynamicStateCreateInfo(dynamicStateCount=2, pDynamicStates=dyn),
            layout=self.layout)
        self.pipe = vk.vkCreateGraphicsPipelines(self.dev, vk.VK_NULL_HANDLE, 1, [info], None)[0]
        for m in modules:
            vk.vkDestroyShaderModule(self.dev, m, None)

    def _memory_type(self, bits, want):
        mp = vk.vkGetPhysicalDeviceMemoryProperties(self.phys)
        for i in range(mp.memoryTypeCount):
            if bits >> i & 1 and mp.memoryTypes[i].propertyFlags & want == want:
                return i
        raise RuntimeError("no suitable memory type")

    def _buffer(self, size, usage, props, exportable):
        fd_bit = vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
        ext = vk.VkExternalMemoryBufferCreateInfo(handleTypes=fd_bit) if exportable else None
        buf = vk.vkCreateBuffer(self.dev, vk.VkBufferCreateInfo(pNext=ext, size=size, usage=usage), None)
        req = vk.vkGetBufferMemoryRequirements(self.dev, buf)
        chain = None
        if exportable:
            ded = vk.VkMemoryDedicatedAllocateInfo(buffer=buf)
            chain = vk.VkExportMemoryAllocateInfo(pNext=ded, handleTypes=fd_bit)
        mem = vk.vkAllocateMemory(self.dev, vk.VkMemoryAllocateInfo(
            pNext=chain, allocationSize=req.size, memoryTypeIndex=self._memory_type(req.memoryTypeBits, props)), None)
        vk.vkBindBufferMemory(self.dev, buf, mem, 0)
        return buf, mem, req.size

    def _export_to_cuda(self, s, alloc):
        # memory: VkDeviceMemory -> fd -> cudaExternalMemory -> device pointer
        fd = self._dfn("vkGetMemoryFdKHR")(self.dev, vk.VkMemoryGetFdInfoKHR(
            memory=s.mem, handleType=vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT))
        md = cu.ExternalMemoryHandleDesc(type=cu.OPAQUE_FD, size=alloc, flags=cu.DEDICATED)
        md.handle.fd = fd  # CUDA owns the fd after a successful import
        s.cuda_mem = cu.H()
        cu.fn.cudaImportExternalMemory(C.byref(s.cuda_mem), C.byref(md))
        ptr = C.c_void_p()
        cu.fn.cudaExternalMemoryGetMappedBuffer(C.byref(ptr), s.cuda_mem,
                                                C.byref(cu.ExternalMemoryBufferDesc(size=self.bytes)))
        s.ptr = ptr.value
        # semaphore: the same route
        fd = self._dfn("vkGetSemaphoreFdKHR")(self.dev, vk.VkSemaphoreGetFdInfoKHR(
            semaphore=s.timeline, handleType=vk.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT))
        sd = cu.ExternalSemaphoreHandleDesc(type=cu.TIMELINE_SEMAPHORE_FD)
        sd.handle.fd = fd
        s.cuda_sem = cu.H()
        cu.fn.cudaImportExternalSemaphore(C.byref(s.cuda_sem), C.byref(sd))

    def _slots(self):
        self.dpool = vk.vkCreateDescriptorPool(self.dev, vk.VkDescriptorPoolCreateInfo(
            maxSets=self.n, poolSizeCount=1, pPoolSizes=[vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=self.n)]), None)
        size = (self.bytes + 3) // 4 * 4
        cmds = vk.vkAllocateCommandBuffers(self.dev, vk.VkCommandBufferAllocateInfo(
            commandPool=self.pool, level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=self.n))
        device_local = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        host = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        self.slots = []
        for i in range(self.n):
            s = _Slot()
            s.cmd = cmds[i]
            s.fence = vk.vkCreateFence(self.dev, vk.VkFenceCreateInfo(flags=vk.VK_FENCE_CREATE_SIGNALED_BIT), None)
            s.acquired = vk.vkCreateSemaphore(self.dev, vk.VkSemaphoreCreateInfo(), None)
            s.buf, s.mem, alloc = self._buffer(
                size, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT, device_local,
                self.zerocopy)
            if self.zerocopy:
                es = vk.VkExportSemaphoreCreateInfo(handleTypes=vk.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT)
                ty = vk.VkSemaphoreTypeCreateInfo(pNext=es, semaphoreType=vk.VK_SEMAPHORE_TYPE_TIMELINE)
                s.timeline = vk.vkCreateSemaphore(self.dev, vk.VkSemaphoreCreateInfo(pNext=ty), None)
                self._export_to_cuda(s, alloc)
            else:
                s.staging, s.staging_mem, _ = self._buffer(size, vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host, False)
                s.mapped = np.frombuffer(vk.vkMapMemory(self.dev, s.staging_mem, 0, size, 0), np.uint8)
            s.set = vk.vkAllocateDescriptorSets(self.dev, vk.VkDescriptorSetAllocateInfo(
                descriptorPool=self.dpool, descriptorSetCount=1, pSetLayouts=[self.dlayout]))[0]
            vk.vkUpdateDescriptorSets(self.dev, 1, [vk.VkWriteDescriptorSet(
                dstSet=s.set, descriptorCount=1, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[vk.VkDescriptorBufferInfo(buffer=s.buf, offset=0, range=WHOLE)])], 0, None)
            self.slots.append(s)

    def _check_target(self):
        self.check_img = vk.vkCreateImage(self.dev, vk.VkImageCreateInfo(
            imageType=vk.VK_IMAGE_TYPE_2D, format=FORMAT, extent=vk.VkExtent3D(self.w, self.h, 1), mipLevels=1,
            arrayLayers=1, samples=vk.VK_SAMPLE_COUNT_1_BIT,
            usage=vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT), None)
        req = vk.vkGetImageMemoryRequirements(self.dev, self.check_img)
        self.check_mem = vk.vkAllocateMemory(self.dev, vk.VkMemoryAllocateInfo(
            allocationSize=req.size,
            memoryTypeIndex=self._memory_type(req.memoryTypeBits, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)), None)
        vk.vkBindImageMemory(self.dev, self.check_img, self.check_mem, 0)
        self.check_view = self._view(self.check_img)
        n = self.w * self.h * 4
        self.readback, self.readback_mem, _ = self._buffer(
            n, vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT, False)
        self.readback_map = np.frombuffer(vk.vkMapMemory(self.dev, self.readback_mem, 0, n, 0), np.uint8)

    # ------------------------------------------------------------------ producer thread (CUDA only)
    def cuda_begin(self, i, stream):
        s = self.slots[i]
        p = cu.WaitParams()
        p.params.fence.value = s.released
        cu.fn.cudaWaitExternalSemaphoresAsync_v2(C.byref(s.cuda_sem), C.byref(p), 1, stream)
        return s.ptr

    def cuda_end(self, i, stream):
        s = self.slots[i]
        s.counter += 1
        s.written = s.counter
        p = cu.SignalParams()
        p.params.fence.value = s.written
        cu.fn.cudaSignalExternalSemaphoresAsync_v2(C.byref(s.cuda_sem), C.byref(p), 1, stream)  # flushes writes

    # ------------------------------------------------------------------ recording
    def _buffer_barrier(self, cb, buf, src, dst, sa, da, sq=IGNORED, dq=IGNORED):
        b = vk.VkBufferMemoryBarrier(srcAccessMask=sa, dstAccessMask=da, srcQueueFamilyIndex=sq,
                                     dstQueueFamilyIndex=dq, buffer=buf, offset=0, size=WHOLE)
        vk.vkCmdPipelineBarrier(cb, src, dst, 0, 0, None, 1, [b], 0, None)

    def _image_barrier(self, cb, img, old, new, src, dst, sa, da):
        b = vk.VkImageMemoryBarrier(
            srcAccessMask=sa, dstAccessMask=da, oldLayout=old, newLayout=new, srcQueueFamilyIndex=IGNORED,
            dstQueueFamilyIndex=IGNORED, image=img,
            subresourceRange=vk.VkImageSubresourceRange(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, levelCount=1,
                                                        layerCount=1))
        vk.vkCmdPipelineBarrier(cb, src, dst, 0, 0, None, 0, None, 1, [b])

    def _record_draw(self, cb, s, view, size, vp):
        area = vk.VkRect2D(offset=vk.VkOffset2D(0, 0), extent=vk.VkExtent2D(*size))
        att = vk.VkRenderingAttachmentInfo(
            imageView=view, imageLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR, storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE)  # black letterbox bars
        vk.vkCmdBeginRendering(cb, vk.VkRenderingInfo(renderArea=area, layerCount=1, colorAttachmentCount=1,
                                                      pColorAttachments=[att]))
        vk.vkCmdBindPipeline(cb, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipe)
        vk.vkCmdBindDescriptorSets(cb, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self.layout, 0, 1, [s.set], 0, None)
        vk.vkCmdPushConstants(cb, self.layout, vk.VK_SHADER_STAGE_FRAGMENT_BIT, 0, 8,
                              vk.ffi.new("int[2]", [self.w, self.h]))
        vk.vkCmdSetViewport(cb, 0, 1, [vk.VkViewport(x=vp[0], y=vp[1], width=vp[2], height=vp[3], minDepth=0.0,
                                                     maxDepth=1.0)])
        vk.vkCmdSetScissor(cb, 0, 1, [area])
        vk.vkCmdDraw(cb, 3, 1, 0, 0)
        vk.vkCmdEndRendering(cb)

    def _record_check(self, cb, s):
        color = vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        out_stage = vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        transfer = vk.VK_PIPELINE_STAGE_TRANSFER_BIT
        write = vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
        self._image_barrier(cb, self.check_img, vk.VK_IMAGE_LAYOUT_UNDEFINED, color, transfer, out_stage, 0, write)
        self._record_draw(cb, s, self.check_view, (self.w, self.h), (0, 0, self.w, self.h))
        self._image_barrier(cb, self.check_img, color, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, out_stage, transfer,
                            write, vk.VK_ACCESS_TRANSFER_READ_BIT)
        region = vk.VkBufferImageCopy(
            imageSubresource=vk.VkImageSubresourceLayers(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, layerCount=1),
            imageExtent=vk.VkExtent3D(self.w, self.h, 1))
        vk.vkCmdCopyImageToBuffer(cb, self.check_img, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, self.readback, 1,
                                  [region])
        self._buffer_barrier(cb, self.readback, transfer, vk.VK_PIPELINE_STAGE_HOST_BIT,
                             vk.VK_ACCESS_TRANSFER_WRITE_BIT, vk.VK_ACCESS_HOST_READ_BIT)

    # ------------------------------------------------------------------ consumer thread
    def draw(self, i, host, rgba):
        s, dev = self.slots[i], self.dev
        forever = 2**64 - 1
        vk.vkWaitForFences(dev, 1, [s.fence], vk.VK_TRUE, forever)  # this slot's command buffer is idle
        if s.timed:
            t = vk.ffi.new("uint64_t[2]")
            vk.vkGetQueryPoolResults(dev, self.queries, 2 * i, 2, 16, t, 8,
                                     vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT)
            self.gpu_draw_ms = (t[1] - t[0]) * self.period_ns * 1e-6
        index = vk.ffi.new("uint32_t*")
        while True:
            try:
                self.acquire_fn(dev, self.swapchain, forever, s.acquired, vk.VK_NULL_HANDLE, index)
                break
            except vk.VkSuboptimalKhr:
                break
            except vk.VkErrorOutOfDateKhr:
                self._recreate_swapchain()
        img = index[0]
        vk.vkResetFences(dev, 1, [s.fence])

        frag = vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
        out_stage = vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        bottom = vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
        transfer = vk.VK_PIPELINE_STAGE_TRANSFER_BIT
        read, write = vk.VK_ACCESS_SHADER_READ_BIT, vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
        color = vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        cb = s.cmd
        vk.vkBeginCommandBuffer(cb, vk.VkCommandBufferBeginInfo(flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        vk.vkCmdResetQueryPool(cb, self.queries, 2 * i, 2)
        if self.zerocopy:  # take the buffer over from the "external" owner (CUDA)
            self._buffer_barrier(cb, s.buf, frag, frag, 0, read, EXTERNAL, self.family)
        else:  # host -> staging (CPU copy) -> device-local buffer (GPU copy over PCIe)
            s.mapped[:self.bytes] = host
            self._buffer_barrier(cb, s.buf, frag, transfer, 0, 0)  # after the previous frame's reads
            vk.vkCmdCopyBuffer(cb, s.staging, s.buf, 1, [vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=self.bytes)])
            self._buffer_barrier(cb, s.buf, transfer, frag, vk.VK_ACCESS_TRANSFER_WRITE_BIT, read)
        if rgba is not None:
            self._record_check(cb, s)
        self._image_barrier(cb, self.images[img], vk.VK_IMAGE_LAYOUT_UNDEFINED, color, out_stage, out_stage, 0, write)
        ew, eh = self.extent
        k = min(ew / self.w, eh / self.h)  # letterbox
        vw, vh = int(self.w * k), int(self.h * k)
        vk.vkCmdWriteTimestamp(cb, vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, self.queries, 2 * i)
        self._record_draw(cb, s, self.views[img], self.extent, ((ew - vw) / 2, (eh - vh) / 2, vw, vh))
        vk.vkCmdWriteTimestamp(cb, bottom, self.queries, 2 * i + 1)
        s.timed = True
        self._image_barrier(cb, self.images[img], color, vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, out_stage, bottom, write, 0)
        if self.zerocopy:  # give it back
            self._buffer_barrier(cb, s.buf, frag, bottom, read, 0, self.family, EXTERNAL)
        vk.vkEndCommandBuffer(cb)

        # wait: swapchain image available (binary) + CUDA finished writing (timeline)
        # signal: ready to present (binary)    + slot released to CUDA (timeline)
        waits, signals, stages = [s.acquired], [self.done[img]], [out_stage]
        wait_values, signal_values = [0], [0]  # values of binary semaphores are ignored
        if self.zerocopy:
            s.counter += 1
            s.released = s.counter
            waits.append(s.timeline)
            signals.append(s.timeline)
            stages.append(frag)
            wait_values.append(s.written)
            signal_values.append(s.released)
        timeline = vk.VkTimelineSemaphoreSubmitInfo(
            waitSemaphoreValueCount=len(waits), pWaitSemaphoreValues=wait_values,
            signalSemaphoreValueCount=len(signals), pSignalSemaphoreValues=signal_values)
        vk.vkQueueSubmit(self.queue, 1, [vk.VkSubmitInfo(
            pNext=timeline, waitSemaphoreCount=len(waits), pWaitSemaphores=waits, pWaitDstStageMask=stages,
            commandBufferCount=1, pCommandBuffers=[cb], signalSemaphoreCount=len(signals),
            pSignalSemaphores=signals)], s.fence)
        stale = False
        try:
            self.present_fn(self.queue, vk.VkPresentInfoKHR(
                waitSemaphoreCount=1, pWaitSemaphores=[self.done[img]], swapchainCount=1,
                pSwapchains=[self.swapchain], pImageIndices=[img]))
        except (vk.VkSuboptimalKhr, vk.VkErrorOutOfDateKhr):
            stale = True
        if rgba is not None:
            vk.vkWaitForFences(dev, 1, [s.fence], vk.VK_TRUE, forever)
            bgra = self.readback_map.reshape(self.h, self.w, 4)
            rgba[..., :3] = bgra[..., 2::-1]
            rgba[..., 3] = bgra[..., 3]
        if stale:
            self._recreate_swapchain()

    def close(self):
        dev = self.dev
        vk.vkDeviceWaitIdle(dev)
        cu.fn.cudaDeviceSynchronize()
        for s in self.slots:
            if s.ptr:
                cu.fn.cudaFree(s.ptr)  # releases the mapping, then the imported objects
                cu.fn.cudaDestroyExternalMemory(s.cuda_mem)
                cu.fn.cudaDestroyExternalSemaphore(s.cuda_sem)
                vk.vkDestroySemaphore(dev, s.timeline, None)
            if s.staging is not None:
                s.mapped = None
                vk.vkDestroyBuffer(dev, s.staging, None)
                vk.vkFreeMemory(dev, s.staging_mem, None)
            vk.vkDestroyBuffer(dev, s.buf, None)
            vk.vkFreeMemory(dev, s.mem, None)
            vk.vkDestroySemaphore(dev, s.acquired, None)
            vk.vkDestroyFence(dev, s.fence, None)
        if self.check_img is not None:
            self.readback_map = None
            vk.vkDestroyBuffer(dev, self.readback, None)
            vk.vkFreeMemory(dev, self.readback_mem, None)
            vk.vkDestroyImageView(dev, self.check_view, None)
            vk.vkDestroyImage(dev, self.check_img, None)
            vk.vkFreeMemory(dev, self.check_mem, None)
        self._destroy_swapchain()
        vk.vkDestroyQueryPool(dev, self.queries, None)
        vk.vkDestroyCommandPool(dev, self.pool, None)
        vk.vkDestroyDescriptorPool(dev, self.dpool, None)
        vk.vkDestroyPipeline(dev, self.pipe, None)
        vk.vkDestroyPipelineLayout(dev, self.layout, None)
        vk.vkDestroyDescriptorSetLayout(dev, self.dlayout, None)
        vk.vkDestroyDevice(dev, None)
        self._ifn("vkDestroySurfaceKHR")(self.inst, self.surface, None)
        if self.messenger:
            self._ifn("vkDestroyDebugUtilsMessengerEXT")(self.inst, self.messenger, None)
        vk.vkDestroyInstance(self.inst, None)
