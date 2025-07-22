mod vulkan_app;
use vulkan_app::{VulkanApp, HEIGHT, WIDTH};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Vulkan Triangle")
        .with_inner_size(winit::dpi::LogicalSize::new(WIDTH, HEIGHT))
        .build(&event_loop)
        .unwrap();

    let mut app = VulkanApp::new(&window);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(new_size),
                ..
            } => {
                if new_size.width > 0 && new_size.height > 0 {
                    app.framebuffer_resized = true;
                }
            }
            Event::MainEventsCleared => {
                app.draw_frame(&window);
            }
            _ => {}
        }
    });
}
