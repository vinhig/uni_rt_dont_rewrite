#version 430 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

flat out int o_instance_id;
flat out int o_geometry_id;
flat out int o_prim_id;

uniform mat4 transform;

uniform mat4 view_proj;

uniform int instance_id;
uniform int geometry_id;
uniform int prim_id;

out vec3 o_position;
out vec3 o_normal;
out vec2 o_uv;

void main() {
  o_position = (transform * vec4(position, 1.0)).xyz;
  o_normal = normalize(normal);
  o_uv = uv;

  o_instance_id = instance_id + 1;
  o_geometry_id = geometry_id + 1;
  o_prim_id = prim_id + 1;

  gl_Position = view_proj * transform * vec4(position, 1.0);
}
