#version 430

in vec3 o_position;
in vec3 o_normal;
in vec2 o_uv;
in vec3 o_geo_normal;
in vec4 o_motion;

flat in int o_current_frame;

flat in int o_instance_id;

layout(location = 0) out vec4 position;
layout(location = 1) out vec4 normal;
layout(location = 2) out int visibility;
layout(location = 3) out vec4 geo_normal;
layout(location = 4) out vec4 motion;

void main() {
  position = vec4(o_position, 1.0);
  normal = vec4(normalize(o_normal), 1.0);
  visibility = o_instance_id;
  geo_normal = vec4(o_geo_normal, 1.0);
  motion = o_motion;
}
