pub const WIDTH: u32 = 800;
pub const HEIGHT: u32 = 600;

pub use app::VulkanApp;

mod app;
mod debug;
mod queue;
mod swapchain_support;
mod ubo;
mod vertex;
