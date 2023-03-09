#version 430

in vec3 o_position;
in vec3 o_normal;
in vec2 o_uv;

flat in int o_instance_id;
flat in int o_geometry_id;
flat in int o_prim_id;

layout(location = 0) out vec4 position;
layout(location = 1) out vec4 normal;
// layout(location = 2) out uvec4 visibility;

void main() {
  position = vec4(o_position, 1.0);
  normal = vec4(o_normal, 1.0);
  // visibility = uvec4(o_instance_id, o_geometry_id, o_prim_id, 0);
}
