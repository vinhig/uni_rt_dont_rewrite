#version 430 core

// Original implementation from Q2RTX
// https://github.com/NVIDIA/Q2RTX/blob/master/src/refresh/vkpt/shader/asvgf_atrous.comp

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(binding = 0) uniform sampler2D tex_curr_position;
layout(binding = 1) uniform sampler2D tex_prev_position;
layout(binding = 2) uniform sampler2D tex_curr_normal;
layout(binding = 3) uniform sampler2D tex_prev_normal;
layout(binding = 4) uniform sampler2D tex_curr_geo_normal;
layout(binding = 5) uniform sampler2D tex_prev_geo_normal;
layout(binding = 6) uniform sampler2D tex_curr_depth;
layout(binding = 7) uniform sampler2D tex_prev_depth;
layout(binding = 8) uniform sampler2D tex_curr_motion;
layout(binding = 9) uniform sampler2D tex_prev_motion;

layout(binding = 10) uniform sampler2D tex_curr_moments;
layout(binding = 11) uniform sampler2D tex_prev_moments;

layout(binding = 21) uniform usampler2D tex_hist_len;

layout(binding = 12) uniform sampler3D tex_blue_noise;

layout(binding = 13, rgba32f) uniform restrict image2D img_atrous_ping_moments;
layout(binding = 14, rgba32f) uniform restrict image2D img_atrous_pong_moments;

layout(binding = 15, rgba32f) uniform restrict image2D img_atrous_ping;
layout(binding = 16, rgba32f) uniform restrict image2D img_atrous_pong;

layout(binding = 17) uniform sampler2D tex_atrous_ping_moments;
layout(binding = 18) uniform sampler2D tex_atrous_pong_moments;

layout(binding = 19) uniform sampler2D tex_atrous_ping;
layout(binding = 20) uniform sampler2D tex_atrous_pong;

#define BLUE_NOISE_RES (256)
#define NUM_BLUE_NOISE_TEX (128)
// #define STORAGE_SCALE_HF 32

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

layout(binding = 1, std140) uniform Denoising {
  float flt_antilag;
  float flt_temporal;
  float flt_min_alpha_color;
  float flt_min_alpha_moments;
  float flt_atrous;
  float flt_atrous_lum;
  float flt_atrous_normal;
  float flt_atrous_depth;
}
denoising;

float get_view_depth(float depth, mat4 proj) {
  // Returns linear depth in [near, far]
  return proj[3][2] / (proj[2][2] + (depth * 2.0 - 1.0));
}

float luminance(vec3 color) { return dot(color, vec3(0.2126, 0.7152, 0.0722)); }

const float gaussian_kernel[2][2] = {{1.0 / 4.0, 1.0 / 8.0},
                                     {1.0 / 8.0, 1.0 / 16.0}};

const float wavelet_factor = 0.5;
const float wavelet_kernel[2][2] = {
    {1.0, wavelet_factor}, {wavelet_factor, wavelet_factor *wavelet_factor}};

bool in_bounds(vec2 coord) {
  return all(lessThan(coord, uniforms.target_dim)) &&
         all(greaterThanEqual(coord, ivec2(0)));
}

void filter_image(sampler2D img_color, sampler2D img_moments, out vec3 filtered,
                  out vec2 filtered_moments) {
  ivec2 ipos = ivec2(gl_GlobalInvocationID);

  vec3 color_center = texelFetch(img_color, ipos, 0).rgb;
  vec2 moments_center = texelFetch(img_moments, ipos, 0).rg;

  if (denoising.flt_atrous <= push_iteration) {
    filtered = color_center;
    filtered_moments = moments_center;
    return;
  }

  // Fetch features to compute weight for edge-avoiding
  vec3 normal_center = texelFetch(tex_curr_normal, ipos, 0).xyz;
  float depth_center =
      get_view_depth(texelFetch(tex_curr_depth, ipos, 0).x, uniforms.proj);
  float fwidth_depth = texelFetch(tex_curr_motion, ipos, 0).w;

  float lum_mean = 0.0;
  float sigma_l = 0.0;

  int hist_len = int(texelFetch(tex_hist_len, ipos, 0).r);

  if (denoising.flt_atrous_lum != 0 && hist_len > 1) {
    // Compute luminance variance from the statistical moments: Var(X) = E[X^2]
    // - E[X]^2 The `asvgf_temporal` shader computes a combination of temporal
    // and spatial (3x3) moments, and stores these into a texture. This shader
    // will combine moments of the surrounding pixels using the same weights as
    // for colors, and the combined moments are used on the next iteration.
    lum_mean = moments_center.x;
    float lum_variance =
        max(1e-8, moments_center.y - moments_center.x * moments_center.x);
    sigma_l = min(hist_len, denoising.flt_atrous_lum) / (2.0 * lum_variance);
  } else {
    // If there is no history, treat all moments as invalid, because 3x3 spatial
    // is just not enough to get reasonable filtering. Ignore luminance in this
    // case, and perform a depth-normal-guided bilateral blur.
    sigma_l = 0;
  }

  float normal_weight_scale = clamp(hist_len / 8.0, 0.0, 1.0);

  float normal_weight = denoising.flt_atrous_normal;
  normal_weight *= normal_weight_scale;

  const int step_size = int(1u << push_iteration);

  vec3 sum_color = color_center.rgb;
  vec2 sum_moments = moments_center;

  float sum_w = 1.0;

  ivec2 jitter = ivec2(0.0);
  {
    int texnum = int(uniforms.frame_number);
    ivec2 texpos = ipos & ivec2(BLUE_NOISE_RES - 1);
    float jitter_x =
        texelFetch(tex_blue_noise,
                   ivec3(texpos, (texnum + 0) & (NUM_BLUE_NOISE_TEX - 1)), 0)
            .r;
    float jitter_y =
        texelFetch(tex_blue_noise,
                   ivec3(texpos, (texnum + 1) & (NUM_BLUE_NOISE_TEX - 1)), 0)
            .r;
    jitter = ivec2((vec2(jitter_x, jitter_y) - 0.5) * float(step_size));
  }

  const int r = 1;

  for (int yy = -r; yy <= r; yy++) {
    for (int xx = -r; xx <= r; xx++) {
      ivec2 p = ipos + ivec2(xx, yy) * step_size + jitter;

      if (xx == 0 && yy == 0) {
        continue;
      }

      // TODO: ok not sure here
      float w = 1.0;

      vec3 normal = texelFetch(tex_curr_normal, p, 0).xyz;
      float depth =
          get_view_depth(texelFetch(tex_curr_depth, p, 0).x, uniforms.proj);

      float dist_z =
          abs(depth_center - depth) * fwidth_depth * denoising.flt_atrous_depth;
      w *= exp(-dist_z / float(step_size));
      w *= wavelet_kernel[abs(xx)][abs(yy)];

      vec3 c = texelFetch(img_color, p, 0).rgb;
      vec2 c_mom = texelFetch(img_moments, p, 0).xy;

      float l = luminance(c);
      float dist_l = abs(lum_mean - l);

      w *= exp(-dist_l * dist_l * sigma_l);

      float NdotN = max(0.0, dot(normal_center, normal));

      if (normal_weight >= 0) {
        w *= pow(NdotN, normal_weight);
      }

      // TODO: why this?
      if (denoising.flt_atrous <= push_iteration) {
        w = 0;
      }

      sum_color += c * w;
      sum_moments += c_mom * w;
      sum_w += w;
    }
  }

  filtered = sum_color / sum_w;
  filtered_moments = sum_moments / sum_w;
}

void main() {
  ivec2 ipos = ivec2(gl_GlobalInvocationID);

  if (!in_bounds(ipos)) {
    return;
  }

  vec3 filtered = vec3(0.0);
  vec2 filtered_moments = vec2(0.0);

  switch (push_iteration) {
  case 0:
    filter_image(tex_atrous_ping, tex_curr_moments, filtered, filtered_moments);
    break;
  case 1:
    filter_image(tex_atrous_pong, tex_atrous_pong_moments, filtered,
                 filtered_moments);
    break;
  case 2:
    filter_image(tex_atrous_ping, tex_atrous_ping_moments, filtered,
                 filtered_moments);
    break;
  case 3:
    filter_image(tex_atrous_pong, tex_atrous_pong_moments, filtered,
                 filtered_moments);
    break;
  case 4:
    filter_image(tex_atrous_ping, tex_atrous_ping_moments, filtered,
                 filtered_moments);
    break;
  case 5:
    filter_image(tex_atrous_pong, tex_atrous_pong_moments, filtered,
                 filtered_moments);
    break;
  }

  switch (push_iteration) {
  case 0:
    imageStore(img_atrous_pong, ipos, vec4(filtered, 1.0));
    imageStore(img_atrous_pong_moments, ipos, vec4(filtered_moments, 0, 0));
    break;
  case 1:
    imageStore(img_atrous_ping, ipos, vec4(filtered, 1.0));
    imageStore(img_atrous_ping_moments, ipos, vec4(filtered_moments, 0, 0));
    break;
  case 2:
    imageStore(img_atrous_pong, ipos, vec4(filtered, 1.0));
    imageStore(img_atrous_pong_moments, ipos, vec4(filtered_moments, 0, 0));
    break;
  case 3:
    imageStore(img_atrous_ping, ipos, vec4(filtered, 1.0));
    imageStore(img_atrous_ping_moments, ipos, vec4(filtered_moments, 0, 0));
    break;
  case 4:
    imageStore(img_atrous_pong, ipos, vec4(filtered, 1.0));
    imageStore(img_atrous_pong_moments, ipos, vec4(filtered_moments, 0, 0));
    break;
  case 5:
    imageStore(img_atrous_ping, ipos, vec4(filtered, 1.0));
    imageStore(img_atrous_ping_moments, ipos, vec4(filtered_moments, 0, 0));
    break;
  }
}
