#version 430 core

// Original implementation from Q2RTX
// https://github.com/NVIDIA/Q2RTX/blob/master/src/refresh/vkpt/shader/asvgf_gradient_img.comp

layout(binding = 0) uniform sampler2D t_curr_sample;
layout(binding = 1) uniform sampler2D t_prev_luminance;

layout(binding = 2, rgba32f) uniform restrict image2D t_out_gradient;

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

#define GRAD_DWN 3

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

float get_gradient(float l_curr, float l_prev) {
  float l_max = max(l_curr, l_prev);

  if (l_max == 0)
    return 0;

  float ret = abs(l_curr - l_prev) / l_max;
  ret *= ret; // make small changes less significant

  return ret;
}

float luminance(vec3 color) { return dot(color, vec3(0.2126, 0.7152, 0.0722)); }

void main() {
  ivec2 ipos = ivec2(gl_GlobalInvocationID);
  // Check out of bounds
  if (any(greaterThanEqual(ipos, ivec2(1280 / 3, 720 / 3)))) {
    return;
  }

  float curr_luminance = 0.0;

  for (int offy = 0; offy < GRAD_DWN; offy++) {
    for (int offx = 0; offx < GRAD_DWN; offx++) {
      float new_lum = luminance(
          texelFetch(t_curr_sample, ipos * 3 + ivec2(offx, offy), 0).rgb);

      if (new_lum > curr_luminance) {
        curr_luminance = new_lum;
      }
    }
  }

  float prev_luminance = texelFetch(t_prev_luminance, ipos, 0).b;

  float gradient = get_gradient(curr_luminance, prev_luminance);

  imageStore(t_out_gradient, ipos, vec4(gradient));
}