#version 430 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

flat out int o_instance_id;

uniform mat4 transform;

uniform mat4 view_proj;

uniform int instance_id;

out vec3 o_position;
out vec3 o_normal;
out vec2 o_uv;

void main() {
  o_position = (transform * vec4(position, 1.0)).xyz;
  o_normal = normalize(inverse(transpose(mat3(transform))) * normal);
  o_uv = uv;

  o_instance_id = instance_id + 1;

  gl_Position = view_proj * transform * vec4(position, 1.0);
}
