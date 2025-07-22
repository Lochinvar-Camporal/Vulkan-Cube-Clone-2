use cgmath::Matrix4;

#[derive(Copy, Clone)]
pub struct UniformBufferObject {
    pub model: Matrix4<f32>,
    pub view: Matrix4<f32>,
    pub proj: Matrix4<f32>,
}
