// Vulkan 1.3 backend (render/vk_renderer.py), dynamic rendering (no VkRenderPass / VkFramebuffer).
//   per slot: a VkBuffer whose memory is exported as a POSIX fd and imported by CUDA, plus a
//   timeline semaphore exported the same way.
// Ownership of a slot is passed back and forth on that semaphore, entirely on the GPU:
//   CUDA waits  value r (Vulkan's last release) -> writes -> signals r' = counter+1   (producer)
//   Vulkan waits r' at the fragment shader      -> draws  -> signals counter+2        (consumer)
// No map / unmap: the CUDA device pointer stays valid for the lifetime of the buffer.
#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_VULKAN
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cstring>
#include "gpu.hpp"
#include "renderer.hpp"
#include "shader_file.hpp"

namespace render {
namespace {

#define VK(x)                                                                                              \
  do {                                                                                                     \
    VkResult r_ = (x);                                                                                     \
    if (r_ != VK_SUCCESS) throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                                                   " VkResult " + std::to_string(r_));                     \
  } while (0)

VKAPI_ATTR VkBool32 VKAPI_CALL on_validation(VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
                                             const VkDebugUtilsMessengerCallbackDataEXT* d, void*) {
  std::fprintf(stderr, "VALIDATION: %s\n", d->pMessage);
  return VK_FALSE;
}

struct Buffer {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkDeviceSize alloc = 0;
};

class VkBackend : public Backend {
 public:
  VkBackend(const Config& c, GLFWwindow* win) : c_(c), win_(win), bytes_((VkDeviceSize)c.width * c.height * 3) {
    create_instance();
    VK(glfwCreateWindowSurface(inst_, win_, nullptr, &surface_));
    pick_device();
    create_device();
    create_swapchain();
    create_pipeline();
    create_slots();
    if (c.check) create_check_target();
  }

  ~VkBackend() override {
    vkDeviceWaitIdle(dev_);
    cudaDeviceSynchronize();
    for (auto& s : slots_) {
      if (s.cuda_ptr) cudaFree(s.cuda_ptr);  // releases the mapping, then the imported objects
      if (s.cuda_mem) cudaDestroyExternalMemory(s.cuda_mem);
      if (s.cuda_sem) cudaDestroyExternalSemaphore(s.cuda_sem);
      destroy(s.frame);
      destroy(s.staging);
      if (s.timeline) vkDestroySemaphore(dev_, s.timeline, nullptr);
      vkDestroySemaphore(dev_, s.acquired, nullptr);
      vkDestroyFence(dev_, s.fence, nullptr);
    }
    destroy(readback_);
    if (check_view_) vkDestroyImageView(dev_, check_view_, nullptr);
    if (check_img_) vkDestroyImage(dev_, check_img_, nullptr);
    if (check_mem_) vkFreeMemory(dev_, check_mem_, nullptr);
    destroy_swapchain();
    vkDestroyQueryPool(dev_, queries_, nullptr);
    vkDestroyCommandPool(dev_, pool_, nullptr);
    vkDestroyDescriptorPool(dev_, dpool_, nullptr);
    vkDestroyPipeline(dev_, pipe_, nullptr);
    vkDestroyPipelineLayout(dev_, layout_, nullptr);
    vkDestroyDescriptorSetLayout(dev_, dlayout_, nullptr);
    vkDestroyDevice(dev_, nullptr);
    vkDestroySurfaceKHR(inst_, surface_, nullptr);
    if (messenger_)
      ((PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(inst_, "vkDestroyDebugUtilsMessengerEXT"))(
          inst_, messenger_, nullptr);
    vkDestroyInstance(inst_, nullptr);
  }

  // ---- producer thread: only CUDA calls
  uint8_t* cuda_begin(int i, cudaStream_t st) override {
    Slot& s = slots_[i];
    cudaExternalSemaphoreWaitParams p{};
    p.params.fence.value = s.released;
    CU(cudaWaitExternalSemaphoresAsync(&s.cuda_sem, &p, 1, st));
    return s.cuda_ptr;
  }
  void cuda_end(int i, cudaStream_t st) override {
    Slot& s = slots_[i];
    s.written = ++s.counter;
    cudaExternalSemaphoreSignalParams p{};
    p.params.fence.value = s.written;
    CU(cudaSignalExternalSemaphoresAsync(&s.cuda_sem, &p, 1, st));  // also flushes the CUDA writes
  }

  // ---- consumer thread
  void draw(int i, const uint8_t* host, uint8_t* rgba) override {
    Slot& s = slots_[i];
    VK(vkWaitForFences(dev_, 1, &s.fence, VK_TRUE, UINT64_MAX));  // this slot's command buffer is idle
    if (s.timed) {
      uint64_t t[2];
      VK(vkGetQueryPoolResults(dev_, queries_, 2 * i, 2, sizeof t, t, sizeof(uint64_t),
                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
      gpu_ms_ = (t[1] - t[0]) * period_ns_ * 1e-6;
    }
    uint32_t img = 0;
    for (;;) {
      VkResult r = vkAcquireNextImageKHR(dev_, swapchain_, UINT64_MAX, s.acquired, VK_NULL_HANDLE, &img);
      if (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR) break;
      if (r != VK_ERROR_OUT_OF_DATE_KHR) VK(r);
      recreate_swapchain();
    }
    VK(vkResetFences(dev_, 1, &s.fence));

    VkCommandBuffer cb = s.cmd;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK(vkBeginCommandBuffer(cb, &bi));
    vkCmdResetQueryPool(cb, queries_, 2 * i, 2);
    if (c_.zerocopy) {  // take the buffer over from the "external" owner (CUDA)
      buffer_barrier(cb, s.frame.buf, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                     VK_ACCESS_SHADER_READ_BIT, VK_QUEUE_FAMILY_EXTERNAL, family_);
    } else {  // host -> staging (CPU copy) -> device-local buffer (GPU copy over PCIe)
      std::memcpy(s.mapped, host, bytes_);
      // the previous frame in this slot was read by the fragment shader: order the overwrite after it
      buffer_barrier(cb, s.frame.buf, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0);
      VkBufferCopy region{0, 0, bytes_};
      vkCmdCopyBuffer(cb, s.staging.buf, s.frame.buf, 1, &region);
      buffer_barrier(cb, s.frame.buf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                     VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    if (rgba) record_check(cb, s);

    image_barrier(cb, images_[img], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                  VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    double k = std::min((double)extent_.width / c_.width, (double)extent_.height / c_.height);  // letterbox
    float vw = (float)(int)(c_.width * k), vh = (float)(int)(c_.height * k);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries_, 2 * i);
    record_draw(cb, s, views_[img], extent_, {(extent_.width - vw) / 2, (extent_.height - vh) / 2, vw, vh, 0, 1});
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries_, 2 * i + 1);
    s.timed = true;
    image_barrier(cb, images_[img], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                  VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0);
    if (c_.zerocopy)  // give it back
      buffer_barrier(cb, s.frame.buf, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                     VK_ACCESS_SHADER_READ_BIT, 0, family_, VK_QUEUE_FAMILY_EXTERNAL);
    VK(vkEndCommandBuffer(cb));

    // wait: swapchain image available (binary) + CUDA finished writing (timeline)
    // signal: ready to present (binary)    + slot released to CUDA (timeline)
    VkSemaphore waits[2] = {s.acquired, s.timeline}, signals[2] = {done_[img], s.timeline};
    VkPipelineStageFlags stages[2] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT};
    uint64_t wait_values[2] = {0, s.written}, signal_values[2] = {0, 0};  // values of binary semaphores are ignored
    uint32_t n = c_.zerocopy ? 2 : 1;
    if (c_.zerocopy) signal_values[1] = s.released = ++s.counter;
    VkTimelineSemaphoreSubmitInfo ti{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    ti.waitSemaphoreValueCount = n;
    ti.pWaitSemaphoreValues = wait_values;
    ti.signalSemaphoreValueCount = n;
    ti.pSignalSemaphoreValues = signal_values;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, &ti};
    si.waitSemaphoreCount = n;
    si.pWaitSemaphores = waits;
    si.pWaitDstStageMask = stages;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    si.signalSemaphoreCount = n;
    si.pSignalSemaphores = signals;
    VK(vkQueueSubmit(queue_, 1, &si, s.fence));

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &done_[img];
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain_;
    pi.pImageIndices = &img;
    VkResult r = vkQueuePresentKHR(queue_, &pi);

    if (rgba) {
      VK(vkWaitForFences(dev_, 1, &s.fence, VK_TRUE, UINT64_MAX));
      const uint8_t* p = static_cast<const uint8_t*>(readback_map_);
      for (size_t j = 0, m = (size_t)c_.width * c_.height; j < m; ++j) {  // image is B G R A
        rgba[4 * j + 0] = p[4 * j + 2];
        rgba[4 * j + 1] = p[4 * j + 1];
        rgba[4 * j + 2] = p[4 * j + 0];
        rgba[4 * j + 3] = p[4 * j + 3];
      }
    }
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) recreate_swapchain();
    else VK(r);
  }

  double gpu_draw_ms() const override { return gpu_ms_; }
  std::string description() const override { return name_; }

 private:
  struct Slot {
    Buffer frame, staging;
    void* mapped = nullptr;  // staging
    VkSemaphore timeline = VK_NULL_HANDLE, acquired = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    cudaExternalMemory_t cuda_mem = nullptr;
    cudaExternalSemaphore_t cuda_sem = nullptr;
    uint8_t* cuda_ptr = nullptr;
    uint64_t counter = 0, written = 0, released = 0;  // timeline values
    bool timed = false;
  };

  // ------------------------------------------------------------------ setup
  void create_instance() {
    uint32_t n = 0;
    const char** req = glfwGetRequiredInstanceExtensions(&n);
    if (!req) throw std::runtime_error("GLFW reports no Vulkan surface support");
    std::vector<const char*> ext(req, req + n), layers;
    if (c_.validate) {
      ext.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
      layers.push_back("VK_LAYER_KHRONOS_validation");
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = (uint32_t)ext.size();
    ci.ppEnabledExtensionNames = ext.data();
    ci.enabledLayerCount = (uint32_t)layers.size();
    ci.ppEnabledLayerNames = layers.data();
    VK(vkCreateInstance(&ci, nullptr, &inst_));
    if (c_.validate) {
      VkDebugUtilsMessengerCreateInfoEXT d{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
      d.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
      d.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
      d.pfnUserCallback = on_validation;
      VK(((PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(inst_, "vkCreateDebugUtilsMessengerEXT"))(
          inst_, &d, nullptr, &messenger_));
    }
  }

  void pick_device() {  // the physical device that is the CUDA device (same UUID)
    cudaDeviceProp prop;
    int cur = 0;
    CU(cudaGetDevice(&cur));
    CU(cudaGetDeviceProperties(&prop, cur));
    uint32_t n = 0;
    VK(vkEnumeratePhysicalDevices(inst_, &n, nullptr));
    std::vector<VkPhysicalDevice> devs(n);
    VK(vkEnumeratePhysicalDevices(inst_, &n, devs.data()));
    for (auto d : devs) {
      VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
      VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
      vkGetPhysicalDeviceProperties2(d, &p2);
      if (std::memcmp(id.deviceUUID, prop.uuid.bytes, VK_UUID_SIZE)) continue;
      phys_ = d;
      period_ns_ = p2.properties.limits.timestampPeriod;
      name_ = std::string("Vulkan ") + std::to_string(VK_API_VERSION_MAJOR(p2.properties.apiVersion)) + "." +
              std::to_string(VK_API_VERSION_MINOR(p2.properties.apiVersion)) + " on " + p2.properties.deviceName;
    }
    if (!phys_) throw std::runtime_error("no Vulkan device matches the CUDA device");
    vkGetPhysicalDeviceQueueFamilyProperties(phys_, &n, nullptr);
    std::vector<VkQueueFamilyProperties> q(n);
    vkGetPhysicalDeviceQueueFamilyProperties(phys_, &n, q.data());
    for (uint32_t i = 0; i < n; ++i) {
      VkBool32 present = VK_FALSE;
      VK(vkGetPhysicalDeviceSurfaceSupportKHR(phys_, i, surface_, &present));
      if (present && (q[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
        family_ = i;
        return;
      }
    }
    throw std::runtime_error("no graphics + present queue family");
  }

  void create_device() {
    float prio = 1.f;
    VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    q.queueFamilyIndex = family_;
    q.queueCount = 1;
    q.pQueuePriorities = &prio;
    const char* ext[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
                         VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    f12.timelineSemaphore = VK_TRUE;
    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f12};
    ci.queueCreateInfoCount = 1;
    ci.pQueueCreateInfos = &q;
    ci.enabledExtensionCount = 3;
    ci.ppEnabledExtensionNames = ext;
    VK(vkCreateDevice(phys_, &ci, nullptr, &dev_));
    vkGetDeviceQueue(dev_, family_, 0, &queue_);
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pi.queueFamilyIndex = family_;
    VK(vkCreateCommandPool(dev_, &pi, nullptr, &pool_));
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = 2 * c_.slots;
    VK(vkCreateQueryPool(dev_, &qi, nullptr, &queries_));
  }

  void create_swapchain() {
    VkSurfaceCapabilitiesKHR caps;
    VK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys_, surface_, &caps));
    extent_ = caps.currentExtent;
    if (extent_.width == UINT32_MAX) {
      int w, h;
      glfwGetFramebufferSize(win_, &w, &h);
      extent_ = {(uint32_t)w, (uint32_t)h};
    }
    uint32_t n = 0;
    VK(vkGetPhysicalDeviceSurfacePresentModesKHR(phys_, surface_, &n, nullptr));
    std::vector<VkPresentModeKHR> modes(n);
    VK(vkGetPhysicalDeviceSurfacePresentModesKHR(phys_, surface_, &n, modes.data()));
    auto has = [&](VkPresentModeKHR m) { return std::find(modes.begin(), modes.end(), m) != modes.end(); };
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;  // vsync
    if (!c_.vsync) mode = has(VK_PRESENT_MODE_IMMEDIATE_KHR) ? VK_PRESENT_MODE_IMMEDIATE_KHR : VK_PRESENT_MODE_MAILBOX_KHR;

    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = surface_;
    ci.minImageCount = caps.minImageCount + 1;
    if (caps.maxImageCount) ci.minImageCount = std::min(ci.minImageCount, caps.maxImageCount);
    ci.imageFormat = kFormat;  // UNORM: the bytes are shown as they are (no sRGB re-encoding)
    ci.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    ci.imageExtent = extent_;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = mode;
    ci.clipped = VK_TRUE;
    VK(vkCreateSwapchainKHR(dev_, &ci, nullptr, &swapchain_));
    VK(vkGetSwapchainImagesKHR(dev_, swapchain_, &n, nullptr));
    images_.resize(n);
    VK(vkGetSwapchainImagesKHR(dev_, swapchain_, &n, images_.data()));
    views_.resize(n);
    done_.resize(n);
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (uint32_t i = 0; i < n; ++i) {
      views_[i] = view_of(images_[i]);
      VK(vkCreateSemaphore(dev_, &sci, nullptr, &done_[i]));
    }
  }
  void destroy_swapchain() {
    for (auto v : views_) vkDestroyImageView(dev_, v, nullptr);
    for (auto s : done_) vkDestroySemaphore(dev_, s, nullptr);
    vkDestroySwapchainKHR(dev_, swapchain_, nullptr);
  }
  void recreate_swapchain() {  // window resized
    vkDeviceWaitIdle(dev_);
    destroy_swapchain();
    create_swapchain();
  }

  VkImageView view_of(VkImage img) {
    VkImageViewCreateInfo v{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    v.image = img;
    v.viewType = VK_IMAGE_VIEW_TYPE_2D;
    v.format = kFormat;
    v.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView out;
    VK(vkCreateImageView(dev_, &v, nullptr, &out));
    return out;
  }

  VkShaderModule module_of(const std::string& path) {
    std::string code = read_file(path);
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule m;
    VK(vkCreateShaderModule(dev_, &ci, nullptr, &m));
    return m;
  }

  void create_pipeline() {
    VkDescriptorSetLayoutBinding b{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dl.bindingCount = 1;
    dl.pBindings = &b;
    VK(vkCreateDescriptorSetLayout(dev_, &dl, nullptr, &dlayout_));
    VkPushConstantRange pc{VK_SHADER_STAGE_FRAGMENT_BIT, 0, 2 * sizeof(int)};
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &dlayout_;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &pc;
    VK(vkCreatePipelineLayout(dev_, &pl, nullptr, &layout_));

    VkShaderModule vs = module_of(c_.shader_dir + "/sbs.vert.spv"), fs = module_of(c_.shader_dir + "/sbs.frag.spv");
    VkPipelineShaderStageCreateInfo stages[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                                 {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs;
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs;
    stages[1].pName = "main";
    // everything OpenGL keeps as mutable context state is baked into the pipeline object here
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.lineWidth = 1.f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState att{};
    att.colorWriteMask = 0xf;
    VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cbs.attachmentCount = 1;
    cbs.pAttachments = &att;
    VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    ds.dynamicStateCount = 2;
    ds.pDynamicStates = dyn;
    VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    ri.colorAttachmentCount = 1;
    ri.pColorAttachmentFormats = &kFormat;
    VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &ri};
    gp.stageCount = 2;
    gp.pStages = stages;
    gp.pVertexInputState = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vp;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms;
    gp.pColorBlendState = &cbs;
    gp.pDynamicState = &ds;
    gp.layout = layout_;
    VK(vkCreateGraphicsPipelines(dev_, VK_NULL_HANDLE, 1, &gp, nullptr, &pipe_));
    vkDestroyShaderModule(dev_, vs, nullptr);
    vkDestroyShaderModule(dev_, fs, nullptr);
  }

  uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
      if ((bits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    throw std::runtime_error("no suitable memory type");
  }

  Buffer make_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, bool exportable) {
    Buffer b;
    VkExternalMemoryBufferCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, exportable ? &ext : nullptr};
    ci.size = size;
    ci.usage = usage;
    VK(vkCreateBuffer(dev_, &ci, nullptr, &b.buf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev_, b.buf, &req);
    VkMemoryDedicatedAllocateInfo ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    ded.buffer = b.buf;
    VkExportMemoryAllocateInfo exp{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, &ded};
    exp.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, exportable ? &exp : nullptr};
    ai.allocationSize = b.alloc = req.size;
    ai.memoryTypeIndex = memory_type(req.memoryTypeBits, props);
    VK(vkAllocateMemory(dev_, &ai, nullptr, &b.mem));
    VK(vkBindBufferMemory(dev_, b.buf, b.mem, 0));
    return b;
  }
  void destroy(Buffer& b) {
    if (b.buf) vkDestroyBuffer(dev_, b.buf, nullptr);
    if (b.mem) vkFreeMemory(dev_, b.mem, nullptr);
    b = Buffer{};
  }

  void export_to_cuda(Slot& s) {
    // memory: VkDeviceMemory -> fd -> cudaExternalMemory -> device pointer
    VkMemoryGetFdInfoKHR gm{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    gm.memory = s.frame.mem;
    gm.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    int fd = -1;
    VK(((PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(dev_, "vkGetMemoryFdKHR"))(dev_, &gm, &fd));
    cudaExternalMemoryHandleDesc md{};
    md.type = cudaExternalMemoryHandleTypeOpaqueFd;
    md.handle.fd = fd;  // CUDA owns the fd after a successful import
    md.size = s.frame.alloc;
    md.flags = cudaExternalMemoryDedicated;
    CU(cudaImportExternalMemory(&s.cuda_mem, &md));
    cudaExternalMemoryBufferDesc bd{};
    bd.size = bytes_;
    CU(cudaExternalMemoryGetMappedBuffer((void**)&s.cuda_ptr, s.cuda_mem, &bd));
    // semaphore: the same route
    VkSemaphoreGetFdInfoKHR gs{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
    gs.semaphore = s.timeline;
    gs.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    VK(((PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(dev_, "vkGetSemaphoreFdKHR"))(dev_, &gs, &fd));
    cudaExternalSemaphoreHandleDesc sd{};
    sd.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
    sd.handle.fd = fd;
    CU(cudaImportExternalSemaphore(&s.cuda_sem, &sd));
  }

  void create_slots() {
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)c_.slots};
    VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dp.maxSets = c_.slots;
    dp.poolSizeCount = 1;
    dp.pPoolSizes = &ps;
    VK(vkCreateDescriptorPool(dev_, &dp, nullptr, &dpool_));
    VkDeviceSize size = (bytes_ + 3) / 4 * 4;
    slots_.resize(c_.slots);
    for (auto& s : slots_) {
      VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      ca.commandPool = pool_;
      ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      ca.commandBufferCount = 1;
      VK(vkAllocateCommandBuffers(dev_, &ca, &s.cmd));
      VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
      VK(vkCreateFence(dev_, &fi, nullptr, &s.fence));
      VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
      VK(vkCreateSemaphore(dev_, &si, nullptr, &s.acquired));
      s.frame = make_buffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, c_.zerocopy);
      if (c_.zerocopy) {
        VkExportSemaphoreCreateInfo es{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
        es.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
        VkSemaphoreTypeCreateInfo ty{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, &es};
        ty.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo ts{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &ty};
        VK(vkCreateSemaphore(dev_, &ts, nullptr, &s.timeline));
        export_to_cuda(s);
      } else {
        s.staging = make_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, false);
        VK(vkMapMemory(dev_, s.staging.mem, 0, VK_WHOLE_SIZE, 0, &s.mapped));
      }
      VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
      da.descriptorPool = dpool_;
      da.descriptorSetCount = 1;
      da.pSetLayouts = &dlayout_;
      VK(vkAllocateDescriptorSets(dev_, &da, &s.set));
      VkDescriptorBufferInfo bi{s.frame.buf, 0, VK_WHOLE_SIZE};
      VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      w.dstSet = s.set;
      w.descriptorCount = 1;
      w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      w.pBufferInfo = &bi;
      vkUpdateDescriptorSets(dev_, 1, &w, 0, nullptr);
    }
  }

  void create_check_target() {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = kFormat;
    ci.extent = {(uint32_t)c_.width, (uint32_t)c_.height, 1};
    ci.mipLevels = ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VK(vkCreateImage(dev_, &ci, nullptr, &check_img_));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev_, check_img_, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK(vkAllocateMemory(dev_, &ai, nullptr, &check_mem_));
    VK(vkBindImageMemory(dev_, check_img_, check_mem_, 0));
    check_view_ = view_of(check_img_);
    readback_ = make_buffer((VkDeviceSize)c_.width * c_.height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                VK_MEMORY_PROPERTY_HOST_CACHED_BIT, false);
    VK(vkMapMemory(dev_, readback_.mem, 0, VK_WHOLE_SIZE, 0, &readback_map_));
  }

  // ------------------------------------------------------------------ recording
  static void buffer_barrier(VkCommandBuffer cb, VkBuffer b, VkPipelineStageFlags src, VkPipelineStageFlags dst,
                             VkAccessFlags sa, VkAccessFlags da, uint32_t sq = VK_QUEUE_FAMILY_IGNORED,
                             uint32_t dq = VK_QUEUE_FAMILY_IGNORED) {
    VkBufferMemoryBarrier m{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    m.srcAccessMask = sa;
    m.dstAccessMask = da;
    m.srcQueueFamilyIndex = sq;
    m.dstQueueFamilyIndex = dq;
    m.buffer = b;
    m.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cb, src, dst, 0, 0, nullptr, 1, &m, 0, nullptr);
  }
  static void image_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to,
                            VkPipelineStageFlags src, VkPipelineStageFlags dst, VkAccessFlags sa, VkAccessFlags da) {
    VkImageMemoryBarrier m{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    m.srcAccessMask = sa;
    m.dstAccessMask = da;
    m.oldLayout = from;
    m.newLayout = to;
    m.srcQueueFamilyIndex = m.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    m.image = img;
    m.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, src, dst, 0, 0, nullptr, 0, nullptr, 1, &m);
  }

  void record_draw(VkCommandBuffer cb, Slot& s, VkImageView target, VkExtent2D size, VkViewport vp) {
    VkRenderingAttachmentInfo att{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    att.imageView = target;
    att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;  // black bars of the letterbox
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, size};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &att;
    vkCmdBeginRendering(cb, &ri);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &s.set, 0, nullptr);
    int src[2] = {c_.width, c_.height};
    vkCmdPushConstants(cb, layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof src, src);
    vkCmdSetViewport(cb, 0, 1, &vp);
    VkRect2D sc{{0, 0}, size};
    vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
  }

  void record_check(VkCommandBuffer cb, Slot& s) {
    VkExtent2D size{(uint32_t)c_.width, (uint32_t)c_.height};
    image_barrier(cb, check_img_, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                  VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    record_draw(cb, s, check_view_, size, {0, 0, (float)size.width, (float)size.height, 0, 1});
    image_barrier(cb, check_img_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    VkBufferImageCopy r{};
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent = {size.width, size.height, 1};
    vkCmdCopyImageToBuffer(cb, check_img_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback_.buf, 1, &r);
    buffer_barrier(cb, readback_.buf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                   VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
  }

  const VkFormat kFormat = VK_FORMAT_B8G8R8A8_UNORM;
  Config c_;
  GLFWwindow* win_;
  VkDeviceSize bytes_;
  std::string name_;
  VkInstance inst_ = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;
  VkPhysicalDevice phys_ = VK_NULL_HANDLE;
  VkDevice dev_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  uint32_t family_ = 0;
  double period_ns_ = 1, gpu_ms_ = -1;
  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  VkExtent2D extent_{};
  std::vector<VkImage> images_;
  std::vector<VkImageView> views_;
  std::vector<VkSemaphore> done_;
  VkCommandPool pool_ = VK_NULL_HANDLE;
  VkQueryPool queries_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout dlayout_ = VK_NULL_HANDLE;
  VkDescriptorPool dpool_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkPipeline pipe_ = VK_NULL_HANDLE;
  std::vector<Slot> slots_;
  VkImage check_img_ = VK_NULL_HANDLE;
  VkDeviceMemory check_mem_ = VK_NULL_HANDLE;
  VkImageView check_view_ = VK_NULL_HANDLE;
  Buffer readback_;
  void* readback_map_ = nullptr;
};

}  // namespace

std::unique_ptr<Backend> make_vulkan(const Config& c, GLFWwindow* win) { return std::make_unique<VkBackend>(c, win); }

}  // namespace render
