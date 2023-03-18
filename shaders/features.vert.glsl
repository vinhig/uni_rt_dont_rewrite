#version 430 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

flat out int o_instance_id;

uniform mat4 transform;

uniform int instance_id;

out vec3 o_position;
out vec3 o_normal;
out vec2 o_uv;
out vec3 o_geo_normal;
out vec4 o_motion;

flat out int o_current_frame;

layout(binding = 0, std140) uniform Reprojection {
  mat4 view_proj;
  mat4 inv_view_proj;
  mat4 prev_view_proj;
  mat4 proj;
  vec4 view_pos;
  vec2 target_dim;
  float alpha_illum;
  float alpha_moments;
  float phi_depth;
  float phi_normal;
  float depth_tolerance;
  float normal_tolerance;
  float min_accum_weight;
  uint frame_number;
}
uniforms;

void main() {
  o_position = (transform * vec4(position, 1.0)).xyz;
  o_normal = normalize(inverse(transpose(mat3(transform))) * normal);
  o_uv = uv;
  o_geo_normal = normalize(normal);

  o_current_frame = int(uniforms.frame_number);

  o_instance_id = instance_id + 1;

  o_motion = uniforms.view_proj * vec4(o_position, 1.0) -
             uniforms.prev_view_proj * vec4(o_position, 1.0);

  gl_Position = uniforms.view_proj * transform * vec4(position, 1.0);
}
