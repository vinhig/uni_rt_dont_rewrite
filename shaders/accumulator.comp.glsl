#version 430

layout(binding = 0) uniform sampler2D noisy_data;

layout(binding = 1, rgba32f) uniform image2D accum_data;

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

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
  float gradient_cap;
  uint frame_number;
}
uniforms;

float luminance(vec3 color) { return dot(color, vec3(0.2126, 0.7152, 0.0722)); }

void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID);

  vec4 new = texelFetch(noisy_data, coord, 0);

  if (luminance(new.xyz) > 10.0) {
    new = normalize(new) * 10.0;
  }

  float n = uniforms.frame_number;
  vec4 old = imageLoad(accum_data, coord);
  vec4 accum = (new + n *old) / (n + 1);

  if (uniforms.frame_number == 0) {
    imageStore(accum_data, coord, new);
  } else {
    imageStore(accum_data, coord, accum);
  }
}
