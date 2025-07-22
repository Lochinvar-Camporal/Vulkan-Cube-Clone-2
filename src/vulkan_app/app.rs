use ash::{vk, Entry};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use std::ffi::{CStr, CString};

use cgmath::{Matrix4, Point3, Vector3};
use std::time::Instant;

use super::debug::vulkan_debug_callback;
use super::queue::QueueFamilyIndices;
use super::ubo::UniformBufferObject;
use super::vertex::{Vertex, INDICES, VERTICES};

use super::swapchain_support::SwapchainSupportDetails;
pub struct VulkanApp {
    entry: Entry,
    instance: ash::Instance,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    surface: vk::SurfaceKHR,
    surface_loader: ash::extensions::khr::Surface,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,
    pub framebuffer_resized: bool,
    queue_family_indices: QueueFamilyIndices,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    start_time: Instant,
    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_image_view: vk::ImageView,
}

impl VulkanApp {
    pub fn new(window: &winit::window::Window) -> Self {
        let entry = unsafe { Entry::load().unwrap() };
        let instance = Self::create_instance(&entry, window);
        let (debug_utils_loader, debug_messenger) = Self::setup_debug_messenger(&entry, &instance);
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.raw_display_handle(),
                window.raw_window_handle(),
                None,
            )
            .unwrap()
        };
        let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);
        let (physical_device, queue_family_indices) =
            Self::pick_physical_device(&instance, &surface_loader, surface);
        let (device, graphics_queue, present_queue) =
            Self::create_logical_device(&instance, physical_device, &queue_family_indices);

        let (vertex_buffer, vertex_buffer_memory) = Self::create_vertex_buffer(
            &instance,
            &device,
            physical_device,
            &queue_family_indices,
            &VERTICES,
        );
        let (index_buffer, index_buffer_memory) = Self::create_index_buffer(
            &instance,
            &device,
            physical_device,
            &queue_family_indices,
            &INDICES,
        );

        let swapchain_loader = ash::extensions::khr::Swapchain::new(&instance, &device);
        let (swapchain, swapchain_format, swapchain_extent) = Self::create_swapchain(
            &instance,
            &device,
            physical_device,
            &surface_loader,
            surface,
            &queue_family_indices,
            &swapchain_loader,
            window,
        );
        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain).unwrap() };
        let swapchain_image_views =
            Self::create_image_views(&device, &swapchain_images, swapchain_format);
        let depth_format = Self::find_depth_format(&instance, physical_device);
        let descriptor_set_layout = Self::create_descriptor_set_layout(&device);
        let render_pass = Self::create_render_pass(&device, swapchain_format, depth_format);
        let (graphics_pipeline, pipeline_layout) = Self::create_graphics_pipeline(
            &device,
            render_pass,
            swapchain_extent,
            descriptor_set_layout,
        );
        let (depth_image, depth_image_memory, depth_image_view) =
            Self::create_depth_resources(&instance, &device, physical_device, swapchain_extent);
        let framebuffers = Self::create_framebuffers(
            &device,
            &swapchain_image_views,
            depth_image_view,
            render_pass,
            swapchain_extent,
        );
        let command_pool = Self::create_command_pool(&device, &queue_family_indices);
        let (vertex_buffer, vertex_buffer_memory) = Self::create_vertex_buffer(
            &instance,
            &device,
            physical_device,
            &queue_family_indices,
            &VERTICES,
        );
        let (index_buffer, index_buffer_memory) = Self::create_index_buffer(
            &instance,
            &device,
            physical_device,
            &queue_family_indices,
            &INDICES,
        );
        let descriptor_set_layout = Self::create_descriptor_set_layout(&device);
        let (descriptor_pool, descriptor_sets) =
            Self::create_descriptor_pool(&device, swapchain_images.len(), descriptor_set_layout);

        let (uniform_buffers, uniform_buffers_memory) = Self::create_uniform_buffers(
            &instance,
            &device,
            physical_device,
            swapchain_images.len(),
        );
        let command_buffers =
            Self::create_command_buffers(&device, command_pool, swapchain_images.len());
        let (image_available_semaphore, render_finished_semaphore, in_flight_fence) =
            Self::create_sync_objects(&device);

        let descriptor_sets = Self::create_descriptor_sets(
            &device,
            descriptor_pool,
            descriptor_set_layout,
            &uniform_buffers,
            swapchain_images.len(),
        );

        Self {
            entry,
            instance,
            debug_utils_loader,
            debug_messenger,
            surface,
            surface_loader,
            physical_device,
            device,
            graphics_queue,
            present_queue,
            swapchain_loader,
            swapchain,
            swapchain_images,
            swapchain_format,
            swapchain_extent,
            swapchain_image_views,
            render_pass,
            pipeline_layout,
            graphics_pipeline,
            framebuffers,
            command_pool,
            command_buffers,
            image_available_semaphore,
            render_finished_semaphore,
            in_flight_fence,
            framebuffer_resized: false,
            queue_family_indices,
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
            uniform_buffers,
            uniform_buffers_memory,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            start_time: Instant::now(),
            depth_image,
            depth_image_memory,
            depth_image_view,
        }
    }

    fn create_instance(entry: &Entry, window: &winit::window::Window) -> ash::Instance {
        let app_name = CString::new("Vulkan Triangle").unwrap();
        let engine_name = CString::new("No Engine").unwrap();
        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_0);

        let mut extension_names =
            ash_window::enumerate_required_extensions(window.raw_display_handle())
                .unwrap()
                .to_vec();
        extension_names.push(ash::extensions::ext::DebugUtils::name().as_ptr());

        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);

        unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("Failed to create instance")
        }
    }

    fn setup_debug_messenger(
        entry: &Entry,
        instance: &ash::Instance,
    ) -> (ash::extensions::ext::DebugUtils, vk::DebugUtilsMessengerEXT) {
        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vulkan_debug_callback));

        let debug_utils_loader = ash::extensions::ext::DebugUtils::new(entry, instance);
        let debug_messenger = unsafe {
            debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap()
        };

        (debug_utils_loader, debug_messenger)
    }

    fn pick_physical_device(
        instance: &ash::Instance,
        surface_loader: &ash::extensions::khr::Surface,
        surface: vk::SurfaceKHR,
    ) -> (vk::PhysicalDevice, QueueFamilyIndices) {
        let physical_devices = unsafe { instance.enumerate_physical_devices().unwrap() };
        let physical_device = physical_devices
            .into_iter()
            .find(|pdevice| Self::is_device_suitable(instance, surface_loader, surface, *pdevice))
            .expect("Failed to find a suitable GPU!");

        let indices = Self::find_queue_families(instance, surface_loader, surface, physical_device);
        (physical_device, indices)
    }

    fn is_device_suitable(
        instance: &ash::Instance,
        surface_loader: &ash::extensions::khr::Surface,
        surface: vk::SurfaceKHR,
        pdevice: vk::PhysicalDevice,
    ) -> bool {
        let indices = Self::find_queue_families(instance, surface_loader, surface, pdevice);
        let extensions_supported = Self::check_device_extension_support(instance, pdevice);

        let mut swapchain_adequate = false;
        if extensions_supported {
            let swapchain_support = Self::query_swapchain_support(surface_loader, pdevice, surface);
            swapchain_adequate = !swapchain_support.formats.is_empty()
                && !swapchain_support.present_modes.is_empty();
        }

        indices.is_complete() && extensions_supported && swapchain_adequate
    }

    fn check_device_extension_support(
        instance: &ash::Instance,
        pdevice: vk::PhysicalDevice,
    ) -> bool {
        let required_extensions = [ash::extensions::khr::Swapchain::name()];
        let available_extensions = unsafe {
            instance
                .enumerate_device_extension_properties(pdevice)
                .unwrap()
        };

        for required in required_extensions.iter() {
            let found = available_extensions.iter().any(|ext| {
                let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                required == &name
            });

            if !found {
                return false;
            }
        }

        true
    }

    fn find_queue_families(
        instance: &ash::Instance,
        surface_loader: &ash::extensions::khr::Surface,
        surface: vk::SurfaceKHR,
        pdevice: vk::PhysicalDevice,
    ) -> QueueFamilyIndices {
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(pdevice) };
        let mut indices = QueueFamilyIndices::new();

        for (i, queue_family) in queue_families.iter().enumerate() {
            if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                indices.graphics_family = Some(i as u32);
            }

            let present_support = unsafe {
                surface_loader
                    .get_physical_device_surface_support(pdevice, i as u32, surface)
                    .unwrap()
            };
            if present_support {
                indices.present_family = Some(i as u32);
            }

            if indices.is_complete() {
                break;
            }
        }

        indices
    }

    fn create_logical_device(
        instance: &ash::Instance,
        pdevice: vk::PhysicalDevice,
        indices: &QueueFamilyIndices,
    ) -> (ash::Device, vk::Queue, vk::Queue) {
        let mut unique_queue_families = std::collections::HashSet::new();
        unique_queue_families.insert(indices.graphics_family.unwrap());
        unique_queue_families.insert(indices.present_family.unwrap());

        let queue_priorities = [1.0];
        let mut queue_create_infos = vec![];
        for queue_family in unique_queue_families {
            let queue_create_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family)
                .queue_priorities(&queue_priorities)
                .build();
            queue_create_infos.push(queue_create_info);
        }

        let physical_device_features = vk::PhysicalDeviceFeatures::builder();
        let required_extensions = [ash::extensions::khr::Swapchain::name().as_ptr()];

        let create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&physical_device_features)
            .enabled_extension_names(&required_extensions);

        let device = unsafe { instance.create_device(pdevice, &create_info, None).unwrap() };

        let graphics_queue =
            unsafe { device.get_device_queue(indices.graphics_family.unwrap(), 0) };
        let present_queue = unsafe { device.get_device_queue(indices.present_family.unwrap(), 0) };

        (device, graphics_queue, present_queue)
    }

    fn create_swapchain(
        instance: &ash::Instance,
        device: &ash::Device,
        pdevice: vk::PhysicalDevice,
        surface_loader: &ash::extensions::khr::Surface,
        surface: vk::SurfaceKHR,
        indices: &QueueFamilyIndices,
        swapchain_loader: &ash::extensions::khr::Swapchain,
        window: &winit::window::Window,
    ) -> (vk::SwapchainKHR, vk::Format, vk::Extent2D) {
        let swapchain_support = Self::query_swapchain_support(surface_loader, pdevice, surface);
        let surface_format = Self::choose_swap_surface_format(&swapchain_support.formats);
        let present_mode = Self::choose_swap_present_mode(&swapchain_support.present_modes);
        let extent = Self::choose_swap_extent(&swapchain_support.capabilities, window);

        let mut image_count = swapchain_support.capabilities.min_image_count + 1;
        if swapchain_support.capabilities.max_image_count > 0
            && image_count > swapchain_support.capabilities.max_image_count
        {
            image_count = swapchain_support.capabilities.max_image_count;
        }

        let mut create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);

        let queue_family_indices = [
            indices.graphics_family.unwrap(),
            indices.present_family.unwrap(),
        ];

        if indices.graphics_family != indices.present_family {
            create_info = create_info
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&queue_family_indices);
        } else {
            create_info = create_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE);
        }

        let create_info = create_info
            .pre_transform(swapchain_support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&create_info, None)
                .unwrap()
        };

        (swapchain, surface_format.format, extent)
    }

    fn query_swapchain_support(
        surface_loader: &ash::extensions::khr::Surface,
        pdevice: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> SwapchainSupportDetails {
        let capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(pdevice, surface)
                .unwrap()
        };
        let formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(pdevice, surface)
                .unwrap()
        };
        let present_modes = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(pdevice, surface)
                .unwrap()
        };

        SwapchainSupportDetails {
            capabilities,
            formats,
            present_modes,
        }
    }

    fn choose_swap_surface_format(
        available_formats: &[vk::SurfaceFormatKHR],
    ) -> vk::SurfaceFormatKHR {
        *available_formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&available_formats[0])
    }

    fn choose_swap_present_mode(
        available_present_modes: &[vk::PresentModeKHR],
    ) -> vk::PresentModeKHR {
        *available_present_modes
            .iter()
            .find(|mode| **mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(&vk::PresentModeKHR::FIFO)
    }

    fn choose_swap_extent(
        capabilities: &vk::SurfaceCapabilitiesKHR,
        window: &winit::window::Window,
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            let inner_size = window.inner_size();
            vk::Extent2D {
                width: inner_size.width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: inner_size.height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        }
    }

    fn create_image_views(
        device: &ash::Device,
        images: &[vk::Image],
        format: vk::Format,
    ) -> Vec<vk::ImageView> {
        images
            .iter()
            .map(|&image| {
                let create_info = vk::ImageViewCreateInfo::builder()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                unsafe { device.create_image_view(&create_info, None).unwrap() }
            })
            .collect()
    }

    fn create_render_pass(
        device: &ash::Device,
        format: vk::Format,
        depth_format: vk::Format,
    ) -> vk::RenderPass {
        let color_attachment = vk::AttachmentDescription::builder()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_attachment_ref = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_attachment = vk::AttachmentDescription::builder()
            .format(depth_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let depth_attachment_ref = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_attachment_ref))
            .depth_stencil_attachment(&depth_attachment_ref);

        let dependency = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            );

        let attachments = [color_attachment.build(), depth_attachment.build()];
        let render_pass_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(std::slice::from_ref(&dependency));

        unsafe { device.create_render_pass(&render_pass_info, None).unwrap() }
    }

    fn create_graphics_pipeline(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        extent: vk::Extent2D,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        let vert_shader_code = include_bytes!(env!("VERT_SHADER_PATH"));
        let frag_shader_code = include_bytes!(env!("FRAG_SHADER_PATH"));

        let vert_shader_module = Self::create_shader_module(device, vert_shader_code);
        let frag_shader_module = Self::create_shader_module(device, frag_shader_code);

        let main_function_name = CString::new("main").unwrap();

        let vert_shader_stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader_module)
            .name(&main_function_name);

        let frag_shader_stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader_module)
            .name(&main_function_name);

        let shader_stages = [
            vert_shader_stage_info.build(),
            frag_shader_stage_info.build(),
        ];

        let binding_description = Vertex::get_binding_description();
        let attribute_descriptions = Vertex::get_attribute_descriptions();
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(std::slice::from_ref(&binding_description))
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(extent.width as f32)
            .height(extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        let scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(extent);

        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(std::slice::from_ref(&viewport))
            .scissors(std::slice::from_ref(&scissor));

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout));
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .unwrap()
        };

        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blending)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let graphics_pipeline = unsafe {
            device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_info),
                    None,
                )
                .unwrap()[0]
        };

        unsafe {
            device.destroy_shader_module(vert_shader_module, None);
            device.destroy_shader_module(frag_shader_module, None);
        }

        (graphics_pipeline, pipeline_layout)
    }

    fn create_shader_module(device: &ash::Device, code: &[u8]) -> vk::ShaderModule {
        let create_info = vk::ShaderModuleCreateInfo::builder().code(unsafe {
            std::slice::from_raw_parts(code.as_ptr() as *const u32, code.len() / 4)
        });
        unsafe { device.create_shader_module(&create_info, None).unwrap() }
    }

    fn create_framebuffers(
        device: &ash::Device,
        image_views: &[vk::ImageView],
        depth_image_view: vk::ImageView,
        render_pass: vk::RenderPass,
        extent: vk::Extent2D,
    ) -> Vec<vk::Framebuffer> {
        image_views
            .iter()
            .map(|&view| {
                let attachments = [view, depth_image_view];
                let framebuffer_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1);
                unsafe { device.create_framebuffer(&framebuffer_info, None).unwrap() }
            })
            .collect()
    }

    fn create_command_pool(device: &ash::Device, indices: &QueueFamilyIndices) -> vk::CommandPool {
        let pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(indices.graphics_family.unwrap())
            .flags(vk::CommandPoolCreateFlags::empty());
        unsafe { device.create_command_pool(&pool_info, None).unwrap() }
    }

    fn create_command_buffers(
        device: &ash::Device,
        command_pool: vk::CommandPool,
        framebuffer_count: usize,
    ) -> Vec<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(framebuffer_count as u32);
        unsafe { device.allocate_command_buffers(&alloc_info).unwrap() }
    }

    fn record_command_buffer(&self, command_buffer: vk::CommandBuffer, image_index: usize) {
        let begin_info = vk::CommandBufferBeginInfo::builder();
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .unwrap();
        }

        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };
        let depth_clear = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };
        let clear_values = [clear_color, depth_clear];
        let render_pass_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffers[image_index])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain_extent,
            })
            .clear_values(&clear_values);

        unsafe {
            self.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            );
            let vertex_buffers = [self.vertex_buffer];
            let offsets = [0];
            self.device
                .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
            self.device.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffer,
                0,
                vk::IndexType::UINT16,
            );
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[image_index]],
                &[],
            );
            self.device
                .cmd_draw_indexed(command_buffer, INDICES.len() as u32, 1, 0, 0, 0);
            self.device.cmd_end_render_pass(command_buffer);
            self.device.end_command_buffer(command_buffer).unwrap();
        }
    }

    fn create_sync_objects(device: &ash::Device) -> (vk::Semaphore, vk::Semaphore, vk::Fence) {
        let semaphore_info = vk::SemaphoreCreateInfo::builder();
        let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

        let image_available_semaphore =
            unsafe { device.create_semaphore(&semaphore_info, None).unwrap() };
        let render_finished_semaphore =
            unsafe { device.create_semaphore(&semaphore_info, None).unwrap() };
        let in_flight_fence = unsafe { device.create_fence(&fence_info, None).unwrap() };

        (
            image_available_semaphore,
            render_finished_semaphore,
            in_flight_fence,
        )
    }

    fn cleanup_swapchain(&mut self) {
        unsafe {
            for i in 0..self.uniform_buffers.len() {
                self.device.destroy_buffer(self.uniform_buffers[i], None);
                self.device
                    .free_memory(self.uniform_buffers_memory[i], None);
            }
            for framebuffer in self.framebuffers.iter() {
                self.device.destroy_framebuffer(*framebuffer, None);
            }
            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            for image_view in self.swapchain_image_views.iter() {
                self.device.destroy_image_view(*image_view, None);
            }
            self.device.destroy_image_view(self.depth_image_view, None);
            self.device.destroy_image(self.depth_image, None);
            self.device.free_memory(self.depth_image_memory, None);
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }

    fn recreate_swapchain(&mut self, window: &winit::window::Window) {
        unsafe {
            self.device.device_wait_idle().unwrap();
        }
        self.cleanup_swapchain();

        let depth_format = Self::find_depth_format(&self.instance, self.physical_device);
        self.render_pass =
            Self::create_render_pass(&self.device, self.swapchain_format, depth_format);

        let (swapchain, swapchain_format, swapchain_extent) = Self::create_swapchain(
            &self.instance,
            &self.device,
            self.physical_device,
            &self.surface_loader,
            self.surface,
            &self.queue_family_indices,
            &self.swapchain_loader,
            window,
        );
        self.swapchain = swapchain;
        self.swapchain_images = unsafe {
            self.swapchain_loader
                .get_swapchain_images(swapchain)
                .unwrap()
        };
        self.swapchain_format = swapchain_format;
        self.swapchain_extent = swapchain_extent;
        self.swapchain_image_views =
            Self::create_image_views(&self.device, &self.swapchain_images, self.swapchain_format);
        let depth_format = Self::find_depth_format(&self.instance, self.physical_device);
        self.render_pass =
            Self::create_render_pass(&self.device, self.swapchain_format, depth_format);
        let (graphics_pipeline, pipeline_layout) = Self::create_graphics_pipeline(
            &self.device,
            self.render_pass,
            self.swapchain_extent,
            self.descriptor_set_layout,
        );
        self.graphics_pipeline = graphics_pipeline;
        self.pipeline_layout = pipeline_layout;
        let (depth_image, depth_image_memory, depth_image_view) = Self::create_depth_resources(
            &self.instance,
            &self.device,
            self.physical_device,
            self.swapchain_extent,
        );
        self.depth_image = depth_image;
        self.depth_image_memory = depth_image_memory;
        self.depth_image_view = depth_image_view;
        self.framebuffers = Self::create_framebuffers(
            &self.device,
            &self.swapchain_image_views,
            self.depth_image_view,
            self.render_pass,
            self.swapchain_extent,
        );
        let (uniform_buffers, uniform_buffers_memory) = Self::create_uniform_buffers(
            &self.instance,
            &self.device,
            self.physical_device,
            self.swapchain_images.len(),
        );
        self.uniform_buffers = uniform_buffers;
        self.uniform_buffers_memory = uniform_buffers_memory;
        let (descriptor_pool, descriptor_sets) = Self::create_descriptor_pool(
            &self.device,
            self.swapchain_images.len(),
            self.descriptor_set_layout,
        );
        self.descriptor_pool = descriptor_pool;
        self.descriptor_sets = descriptor_sets;
        self.descriptor_sets = Self::create_descriptor_sets(
            &self.device,
            self.descriptor_pool,
            self.descriptor_set_layout,
            &self.uniform_buffers,
            self.swapchain_images.len(),
        );
    }

    pub fn draw_frame(&mut self, window: &winit::window::Window) {
        unsafe {
            self.device
                .wait_for_fences(std::slice::from_ref(&self.in_flight_fence), true, u64::MAX)
                .unwrap();

            let result = self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available_semaphore,
                vk::Fence::null(),
            );

            let image_index = match result {
                Ok((image_index, is_suboptimal)) => {
                    if is_suboptimal {
                        self.framebuffer_resized = true;
                    }
                    image_index
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreate_swapchain(window);
                    return;
                }
                Err(error) => panic!("Error acquiring swapchain image: {}", error),
            };

            self.update_uniform_buffer(image_index as usize);

            self.device
                .reset_fences(std::slice::from_ref(&self.in_flight_fence))
                .unwrap();

            self.device
                .reset_command_buffer(
                    self.command_buffers[image_index as usize],
                    vk::CommandBufferResetFlags::empty(),
                )
                .unwrap();
            self.record_command_buffer(
                self.command_buffers[image_index as usize],
                image_index as usize,
            );

            let wait_semaphores = [self.image_available_semaphore];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let signal_semaphores = [self.render_finished_semaphore];
            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(std::slice::from_ref(
                    &self.command_buffers[image_index as usize],
                ))
                .signal_semaphores(&signal_semaphores);

            self.device
                .queue_submit(
                    self.graphics_queue,
                    std::slice::from_ref(&submit_info),
                    self.in_flight_fence,
                )
                .unwrap();

            let swapchains = [self.swapchain];
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(std::slice::from_ref(&image_index));

            let result = self
                .swapchain_loader
                .queue_present(self.present_queue, &present_info);

            let mut recreate_swapchain = false;
            match result {
                Ok(is_suboptimal) => {
                    if is_suboptimal {
                        recreate_swapchain = true;
                    }
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                    recreate_swapchain = true;
                }
                Err(error) => panic!("Failed to present swapchain image: {}", error),
            }

            if self.framebuffer_resized || recreate_swapchain {
                self.framebuffer_resized = false;
                self.recreate_swapchain(window);
            }
        }
    }

    fn update_uniform_buffer(&self, current_image: usize) {
        let time = self.start_time.elapsed().as_secs_f32();

        let model = Matrix4::from_angle_z(cgmath::Deg(time * 90.0));
        let view = Matrix4::look_at_rh(
            Point3::new(2.0, 2.0, 2.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        );
        let mut proj = cgmath::perspective(
            cgmath::Deg(45.0),
            self.swapchain_extent.width as f32 / self.swapchain_extent.height as f32,
            0.1,
            10.0,
        );
        proj[1][1] *= -1.0;

        let ubo = UniformBufferObject { model, view, proj };

        unsafe {
            let data_ptr = self
                .device
                .map_memory(
                    self.uniform_buffers_memory[current_image],
                    0,
                    std::mem::size_of::<UniformBufferObject>() as vk::DeviceSize,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap();
            let mut align = ash::util::Align::new(
                data_ptr,
                std::mem::align_of::<UniformBufferObject>() as _,
                std::mem::size_of::<UniformBufferObject>() as vk::DeviceSize,
            );
            align.copy_from_slice(&[ubo]);
            self.device
                .unmap_memory(self.uniform_buffers_memory[current_image]);
        }
    }

    fn create_index_buffer(
        instance: &ash::Instance,
        device: &ash::Device,
        pdevice: vk::PhysicalDevice,
        _indices: &QueueFamilyIndices,
        data: &[u16],
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_size = (std::mem::size_of::<u16>() * INDICES.len()) as vk::DeviceSize;
        let (buffer, buffer_memory) = Self::create_buffer(
            instance,
            device,
            pdevice,
            buffer_size,
            vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data_ptr = device
                .map_memory(buffer_memory, 0, buffer_size, vk::MemoryMapFlags::empty())
                .unwrap();
            let mut align =
                ash::util::Align::new(data_ptr, std::mem::align_of::<u16>() as _, buffer_size);
            align.copy_from_slice(data);
            device.unmap_memory(buffer_memory);
        }

        (buffer, buffer_memory)
    }

    fn create_vertex_buffer(
        instance: &ash::Instance,
        device: &ash::Device,
        pdevice: vk::PhysicalDevice,
        _indices: &QueueFamilyIndices,
        data: &[Vertex],
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_size = (std::mem::size_of::<Vertex>() * VERTICES.len()) as vk::DeviceSize;
        let (buffer, buffer_memory) = Self::create_buffer(
            instance,
            device,
            pdevice,
            buffer_size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data_ptr = device
                .map_memory(buffer_memory, 0, buffer_size, vk::MemoryMapFlags::empty())
                .unwrap();
            let mut align =
                ash::util::Align::new(data_ptr, std::mem::align_of::<Vertex>() as _, buffer_size);
            align.copy_from_slice(data);
            device.unmap_memory(buffer_memory);
        }

        (buffer, buffer_memory)
    }

    fn create_buffer(
        instance: &ash::Instance,
        device: &ash::Device,
        pdevice: vk::PhysicalDevice,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.create_buffer(&buffer_info, None).unwrap() };
        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mem_type_index = Self::find_memory_type(
            instance,
            pdevice,
            mem_requirements.memory_type_bits,
            properties,
        );

        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index);

        let buffer_memory = unsafe { device.allocate_memory(&alloc_info, None).unwrap() };
        unsafe {
            device.bind_buffer_memory(buffer, buffer_memory, 0).unwrap();
        }

        (buffer, buffer_memory)
    }

    fn find_memory_type(
        instance: &ash::Instance,
        pdevice: vk::PhysicalDevice,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> u32 {
        let mem_properties = unsafe { instance.get_physical_device_memory_properties(pdevice) };
        for i in 0..mem_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && (mem_properties.memory_types[i as usize]
                    .property_flags
                    .contains(properties))
            {
                return i;
            }
        }
        panic!("Failed to find suitable memory type!");
    }

    fn create_depth_resources(
        instance: &ash::Instance,
        device: &ash::Device,
        pdevice: vk::PhysicalDevice,
        extent: vk::Extent2D,
    ) -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
        let depth_format = Self::find_depth_format(instance, pdevice);
        let (depth_image, depth_image_memory) = Self::create_image(
            instance,
            device,
            pdevice,
            extent.width,
            extent.height,
            depth_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let depth_image_view = Self::create_image_view(
            device,
            depth_image,
            depth_format,
            vk::ImageAspectFlags::DEPTH,
        );

        (depth_image, depth_image_memory, depth_image_view)
    }

    fn find_depth_format(instance: &ash::Instance, pdevice: vk::PhysicalDevice) -> vk::Format {
        Self::find_supported_format(
            instance,
            pdevice,
            &[
                vk::Format::D32_SFLOAT,
                vk::Format::D32_SFLOAT_S8_UINT,
                vk::Format::D24_UNORM_S8_UINT,
            ],
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        )
    }

    fn find_supported_format(
        instance: &ash::Instance,
        pdevice: vk::PhysicalDevice,
        candidates: &[vk::Format],
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> vk::Format {
        for &format in candidates {
            let props = unsafe { instance.get_physical_device_format_properties(pdevice, format) };

            if tiling == vk::ImageTiling::LINEAR && props.linear_tiling_features.contains(features)
            {
                return format;
            } else if tiling == vk::ImageTiling::OPTIMAL
                && props.optimal_tiling_features.contains(features)
            {
                return format;
            }
        }

        panic!("Failed to find supported format!");
    }

    fn create_image(
        instance: &ash::Instance,
        device: &ash::Device,
        pdevice: vk::PhysicalDevice,
        width: u32,
        height: u32,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> (vk::Image, vk::DeviceMemory) {
        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(format)
            .tiling(tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1);

        let image = unsafe { device.create_image(&image_info, None).unwrap() };

        let mem_requirements = unsafe { device.get_image_memory_requirements(image) };
        let mem_type_index = Self::find_memory_type(
            instance,
            pdevice,
            mem_requirements.memory_type_bits,
            properties,
        );

        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index);

        let image_memory = unsafe { device.allocate_memory(&alloc_info, None).unwrap() };
        unsafe {
            device.bind_image_memory(image, image_memory, 0).unwrap();
        }

        (image, image_memory)
    }

    fn create_image_view(
        device: &ash::Device,
        image: vk::Image,
        format: vk::Format,
        aspect_flags: vk::ImageAspectFlags,
    ) -> vk::ImageView {
        let view_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect_flags,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        unsafe { device.create_image_view(&view_info, None).unwrap() }
    }

    fn create_uniform_buffers(
        instance: &ash::Instance,
        device: &ash::Device,
        pdevice: vk::PhysicalDevice,
        num_images: usize,
    ) -> (Vec<vk::Buffer>, Vec<vk::DeviceMemory>) {
        let buffer_size = std::mem::size_of::<UniformBufferObject>();
        let mut uniform_buffers = Vec::with_capacity(num_images);
        let mut uniform_buffers_memory = Vec::with_capacity(num_images);

        for _ in 0..num_images {
            let (buffer, memory) = Self::create_buffer(
                instance,
                device,
                pdevice,
                buffer_size as vk::DeviceSize,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            uniform_buffers.push(buffer);
            uniform_buffers_memory.push(memory);
        }

        (uniform_buffers, uniform_buffers_memory)
    }

    fn create_descriptor_set_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
        let ubo_layout_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .build();

        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(std::slice::from_ref(&ubo_layout_binding));

        unsafe {
            device
                .create_descriptor_set_layout(&layout_info, None)
                .unwrap()
        }
    }

    fn create_descriptor_pool(
        device: &ash::Device,
        num_images: usize,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> (vk::DescriptorPool, Vec<vk::DescriptorSet>) {
        let pool_size = vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(100)
            .build();

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(std::slice::from_ref(&pool_size))
            .max_sets(100);

        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None).unwrap() };

        let layouts = vec![descriptor_set_layout; num_images];
        let allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);

        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&allocate_info).unwrap() };

        (descriptor_pool, descriptor_sets)
    }

    fn create_descriptor_sets(
        device: &ash::Device,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
        uniform_buffers: &[vk::Buffer],
        num_images: usize,
    ) -> Vec<vk::DescriptorSet> {
        let layouts = vec![descriptor_set_layout; num_images];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);

        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info).unwrap() };

        for (i, &descriptor_set) in descriptor_sets.iter().enumerate() {
            let buffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(uniform_buffers[i])
                .offset(0)
                .range(std::mem::size_of::<UniformBufferObject>() as vk::DeviceSize)
                .build();

            let descriptor_write = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_info))
                .build();

            unsafe { device.update_descriptor_sets(std::slice::from_ref(&descriptor_write), &[]) };
        }

        descriptor_sets
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.cleanup_swapchain();
            self.device.destroy_buffer(self.index_buffer, None);
            self.device.free_memory(self.index_buffer_memory, None);
            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device.free_memory(self.vertex_buffer_memory, None);
            self.device
                .destroy_semaphore(self.image_available_semaphore, None);
            self.device
                .destroy_semaphore(self.render_finished_semaphore, None);
            self.device.destroy_fence(self.in_flight_fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_image_view(self.depth_image_view, None);
            self.device.destroy_image(self.depth_image, None);
            self.device.free_memory(self.depth_image_memory, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            for i in 0..self.uniform_buffers.len() {
                self.device.destroy_buffer(self.uniform_buffers[i], None);
                self.device
                    .free_memory(self.uniform_buffers_memory[i], None);
            }
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}
