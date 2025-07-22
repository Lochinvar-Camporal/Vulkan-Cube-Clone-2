use ash::vk;
use std::mem::offset_of;

#[derive(Clone, Debug, Copy)]
#[repr(C)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub color: [f32; 3],
}

impl Vertex {
    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Self, pos) as u32)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Self, color) as u32)
                .build(),
        ]
    }
}

pub const VERTICES: [Vertex; 8] = [
    Vertex {
        pos: [-0.5, -0.5, 0.0],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        pos: [0.5, -0.5, 0.0],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        pos: [0.5, 0.5, 0.0],
        color: [0.0, 0.0, 1.0],
    },
    Vertex {
        pos: [-0.5, 0.5, 0.0],
        color: [1.0, 1.0, 1.0],
    },
    Vertex {
        pos: [-0.5, -0.5, -0.5],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        pos: [0.5, -0.5, -0.5],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        pos: [0.5, 0.5, -0.5],
        color: [0.0, 0.0, 1.0],
    },
    Vertex {
        pos: [-0.5, 0.5, -0.5],
        color: [1.0, 1.0, 1.0],
    },
];

pub const INDICES: [u16; 36] = [
    0, 1, 2, 2, 3, 0, // front
    4, 6, 5, 4, 7, 6, // back
    0, 7, 4, 0, 3, 7, // left
    1, 5, 6, 6, 2, 1, // right
    3, 2, 6, 6, 7, 3, // top
    0, 5, 1, 5, 0, 4, // bottom
];
