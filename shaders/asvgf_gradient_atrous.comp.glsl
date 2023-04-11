#version 430 core

// Original implementation from Q2RTX
// https://github.com/NVIDIA/Q2RTX/blob/master/src/refresh/vkpt/shader/asvgf_gradient_img.comp

#define GRAD_DWN (3)

// Texture to read from
layout(binding = 0) uniform sampler2D tex_gradient_ping;
layout(binding = 1) uniform sampler2D tex_gradient_pong;

// Image to write to
layout(binding = 2, rgba32f) uniform restrict image2D img_gradient_ping;
layout(binding = 3, rgba32f) uniform restrict image2D img_gradient_pong;

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

uniform int push_iteration;

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

const float gaussian_kernel[2][2] = {{1.0 / 4.0, 1.0 / 8.0},
                                     {1.0 / 8.0, 1.0 / 16.0}};

const float wavelet_factor = 0.5;
const float wavelet_kernel[2][2] = {
    {1.0, wavelet_factor}, {wavelet_factor, wavelet_factor *wavelet_factor}};

// Blur the gradients
vec2 filter_image(sampler2D img) {
  ivec2 ipos = ivec2(gl_GlobalInvocationID);
  ivec2 grad_size = ivec2(1280, 720) / GRAD_DWN;

  vec2 color_center = texelFetch(img, ipos, 0).xy;

  float sum_w = 1;

  const int step_size = int(1u << push_iteration);

  vec2 sum_color = vec2(0);
  sum_w = 0;

  const int r = 1;
  for (int yy = -r; yy <= r; yy++) {
    for (int xx = -r; xx <= r; xx++) {
      ivec2 p = ipos + ivec2(xx, yy) * step_size;

      vec2 c = texelFetch(img, p, 0).xy;

      if (any(greaterThanEqual(p, grad_size)))
        c = vec2(0);

      float w = wavelet_kernel[abs(xx)][abs(yy)];

      sum_color += c * w;
      sum_w += w;
    }
  }

  sum_color /= sum_w;

  return sum_color;
}

// LF gradients are not normalized in the gradient_img shader - do it after the
// blur
float get_gradient(float l_curr, float l_prev) {
  float l_max = max(l_curr, l_prev);

  if (l_max == 0)
    return 0;

  float grad = abs(l_curr - l_prev) / l_max;
  return grad * grad;
}

void main() {
  ivec2 ipos = ivec2(gl_GlobalInvocationID);
  ivec2 grad_size = ivec2(1280, 720) / GRAD_DWN;
  if (any(greaterThanEqual(ipos, grad_size))) {
    return;
  }

  vec2 filtered = vec2(0);
  switch (push_iteration) {
  case 0:
    filtered = filter_image(tex_gradient_ping);
    break;
  case 1:
    filtered = filter_image(tex_gradient_pong);
    break;
  case 2:
    filtered = filter_image(tex_gradient_ping);
    break;
  }

  switch (push_iteration) {
  case 0:
    imageStore(img_gradient_pong, ipos, vec4(filtered, 0, 0));
    break;
  case 1:
    imageStore(img_gradient_ping, ipos, vec4(filtered, 0, 0));
    break;
  case 2:
    imageStore(img_gradient_pong, ipos, vec4(filtered, 0, 0));
    break;
  }
}
