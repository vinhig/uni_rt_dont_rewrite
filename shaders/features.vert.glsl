#version 430 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

uniform mat4 transform;

uniform mat4 view_proj;

out vec3 o_position;
out vec3 o_normal;
out vec2 o_uv;

void main() {
  o_position = position;
  o_normal = normalize(normal);
  o_uv = uv;

  gl_Position = view_proj * transform * vec4(position, 1.0);
}
