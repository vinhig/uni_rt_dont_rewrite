#version 430

in vec3 o_position;
in vec3 o_normal;
in vec2 o_uv;

layout(location = 0) out vec4 position;
layout(location = 1) out vec4 normal;

void main() {
  position = vec4(o_position, 1.0);
  normal = vec4(o_normal, 1.0);
}
