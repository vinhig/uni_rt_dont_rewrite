#version 430

// Original implementation from Q2RTX
// https://github.com/NVIDIA/Q2RTX/blob/master/src/refresh/vkpt/shader/asvgf_temporal.comp
// - once again, Q2RTX supports multi GPU, but we don't, so support has been
// removed
// - previous position in reprojection is based on a motion vector, here we as
//   in asvgf_gradient_reproject.com.glsl

#extension GL_ARB_enhanced_layouts : enable // to be allowed to use compile time
                                            // expression in some place (as in
                                            // layout definition)

#define GRAD_DWN (3)
#define GROUP_SIZE 15
#define FILTER_RADIUS 1
#define SHARED_SIZE (GROUP_SIZE + FILTER_RADIUS * 2)

layout(local_size_x = GROUP_SIZE, local_size_y = GROUP_SIZE,
       local_size_z = 1) in;

layout(binding = 0) uniform sampler2D tex_curr_position;
layout(binding = 1) uniform sampler2D tex_prev_position;
layout(binding = 2) uniform sampler2D tex_curr_normal;
layout(binding = 3) uniform sampler2D tex_prev_normal;
layout(binding = 4) uniform sampler2D tex_curr_geo_normal;
layout(binding = 5) uniform sampler2D tex_prev_geo_normal;
layout(binding = 6) uniform sampler2D tex_curr_depth;
layout(binding = 7) uniform sampler2D tex_prev_depth;

layout(binding = 22) uniform sampler2D tex_curr_motion;
layout(binding = 23) uniform sampler2D tex_prev_motion;

layout(binding = 8) uniform isampler2D tex_curr_visibility;
layout(binding = 9) uniform isampler2D tex_prev_visibility;

layout(binding = 10) uniform sampler2D tex_curr_sample;
layout(binding = 11) uniform sampler2D tex_prev_sample;

layout(binding = 12) uniform sampler2D tex_curr_hist_color;
// layout(binding = 13) uniform sampler2D tex_prev_hist_color;
layout(binding = 14) uniform sampler2D tex_curr_hist_moments;
layout(binding = 15) uniform sampler2D tex_prev_hist_moments;

// layout(binding = 16) uniform sampler2D tex_gradient_atrous_ping;
// layout(binding = 17) uniform sampler2D tex_gradient_atrous_pong;

layout(binding = 18, rgba32f) uniform restrict image2D img_color_atrous_ping;
layout(binding = 19) uniform sampler2D tex_grad_atrous_pong;

layout(binding = 20, rgba32f) uniform restrict image2D img_curr_hist_moments;
layout(binding = 21, rgba32f) uniform restrict image2D img_atrous_ping_moments;

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

layout(binding = 1, std140) uniform Denoising {
  float flt_antilag;
  float flt_temporal;
  float flt_min_alpha_color;
  float flt_min_alpha_moments;
}
denoising;

shared vec4 s_normal_lum[SHARED_SIZE][SHARED_SIZE];
shared float s_depth[SHARED_SIZE][SHARED_SIZE];
shared float s_depth_width[GROUP_SIZE / GRAD_DWN][GROUP_SIZE / GRAD_DWN];

vec3 unproject_uv(float depth, vec2 uv, mat4 invXProj) {
  float z = depth * 2.0 - 1.0; // OpenGL Z convention
  vec4 ndc = vec4(uv * 2.0 - 1.0, z, 1.0);
  vec4 world = invXProj * ndc;
  return world.xyz / world.w;
}

float luminance(vec3 color) { return dot(color, vec3(0.2126, 0.7152, 0.0722)); }

void preload() {
  ivec2 groupBase = ivec2(gl_WorkGroupID) * GROUP_SIZE - FILTER_RADIUS;

  for (uint linear_idx = gl_LocalInvocationIndex;
       linear_idx < SHARED_SIZE * SHARED_SIZE;
       linear_idx += GROUP_SIZE * GROUP_SIZE) {
    // Convert the linear index to 2D index in a SHARED_SIZE x SHARED_SIZE
    // virtual group
    float t = (float(linear_idx) + 0.5) / float(SHARED_SIZE);
    int xx = int(floor(fract(t) * float(SHARED_SIZE)));
    int yy = int(floor(t));

    // Load
    ivec2 ipos = groupBase + ivec2(xx, yy);
    float depth = texelFetch(tex_curr_depth, ipos, 0).x;
    vec3 normal = texelFetch(tex_curr_normal, ipos, 0).rgb;
    vec3 color = texelFetch(tex_curr_sample, ipos, 0).rgb;

    // Store
    s_normal_lum[yy][xx] = vec4(normal.xyz, luminance(color.rgb));
    s_depth[yy][xx] = depth;
  }
}

void get_shared_data(ivec2 offset, out float depth, out vec3 normal,
                     out float lum) {
  ivec2 addr = ivec2(gl_LocalInvocationID) + ivec2(FILTER_RADIUS) + offset;

  vec4 normal_lum = s_normal_lum[addr.y][addr.x];
  depth = s_depth[addr.y][addr.x];

  normal = normal_lum.xyz;
  lum = normal_lum.w;
}

void main() {
  preload();
  barrier();

  ivec2 ipos = ivec2(gl_GlobalInvocationID);
  vec2 uv = (vec2(ipos) + 0.5) / uniforms.target_dim;

  // Doing the reprojection here
  // (not using a motion buffer, but rather the prev/curr view_proj)
  vec2 pos_prev;
  vec4 motion;
  {
    float current_depth = texelFetch(tex_curr_depth, ipos, 0).x;
    vec3 current_world_position =
        unproject_uv(current_depth, uv, uniforms.inv_view_proj);
    vec4 clip_pos_prev =
        uniforms.prev_view_proj * vec4(current_world_position, 1.0);
    vec3 ndc_pos_prev = clip_pos_prev.xyz / clip_pos_prev.w;
    vec3 reprojected_uv = ndc_pos_prev;
    reprojected_uv.xy = ndc_pos_prev.xy * 0.5 + 0.5;
    reprojected_uv.z = ndc_pos_prev.z * 0.5 + 0.5;

    motion = texelFetch(tex_curr_motion, ipos, 0);
    // motion /= motion.w;

    pos_prev = reprojected_uv.xy * uniforms.target_dim - 0.5;
  }

  float motion_length = length(motion.xy * vec2(1280, 720));

  float curr_depth;
  vec3 curr_normal;
  float curr_lum;
  get_shared_data(ivec2(0), curr_depth, curr_normal, curr_lum);
  vec3 curr_geo_normal = texelFetch(tex_curr_geo_normal, ipos, 0).rgb;

  bool temporal_sample_valid_diff = false;
  vec4 temporal_moments_histlen = vec4(0);
  vec3 temporal_color = vec3(0);
  float temporal_sum_w_diff = 0.0;
  {

    vec2 pos_ld = floor(pos_prev - vec2(0.5));
    vec2 subpix = fract(pos_prev - vec2(0.5) - pos_ld);

    const ivec2 off[4] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    float w[4] = {(1.0 - subpix.x) * (1.0 - subpix.y),
                  (subpix.x) * (1.0 - subpix.y), (1.0 - subpix.x) * (subpix.y),
                  (subpix.x) * (subpix.y)};

    for (int i = 0; i < 4; i++) {
      ivec2 p = ivec2(pos_ld) + off[i];

      float prev_depth = texelFetch(tex_prev_depth, p, 0).x;
      vec3 prev_normal = texelFetch(tex_prev_normal, p, 0).xyz;
      vec3 prev_geo_normal = texelFetch(tex_prev_geo_normal, p, 0).xyz;

      float dist_depth =
          abs(curr_depth - prev_depth + motion.z) / abs(curr_depth);
      float dot_normals = dot(curr_normal, prev_normal);
      float dot_geo_normals = dot(curr_geo_normal, prev_geo_normal);

      if (dist_depth < 0.4 && dot_geo_normals > 0.4) {
        float w_diff = w[i] * max(dot_normals, 0);

        temporal_color += texelFetch(tex_curr_hist_color, p, 0).rgb * w_diff;
        temporal_moments_histlen +=
            texelFetch(tex_prev_hist_moments, p, 0).rgba * w_diff;
        temporal_sum_w_diff += w_diff;
      }
    }

    if (temporal_sum_w_diff > 1e-6) {
      float inv_w_diff = 1.0 / temporal_sum_w_diff;
      temporal_color *= inv_w_diff;
      temporal_moments_histlen *= inv_w_diff;
      temporal_sample_valid_diff = true;
    }
  }

  vec2 spatial_moments = vec2(curr_lum, curr_lum * curr_lum);

  {
    float spatial_sum_w = 1.0;

    for (int yy = -FILTER_RADIUS; yy <= FILTER_RADIUS; yy++) {
      for (int xx = -FILTER_RADIUS; xx <= FILTER_RADIUS; xx++) {
        if (xx == 0 && yy == 0) {
          continue;
        }

        ivec2 p = ipos + ivec2(xx, yy);

        float depth;
        vec3 normal;
        float lum_p;
        get_shared_data(ivec2(xx, yy), depth, normal, lum_p);

        float dist_z = abs(curr_depth - depth) * motion.a;
        if (dist_z < 2.0) {
          float w = pow(max(0.0, dot(normal, curr_normal)), 128.0);

          spatial_moments += vec2(lum_p * w, lum_p * lum_p * w);
          spatial_sum_w += w;
        }
      }
    }

    float inv_w2 = 1.0 / spatial_sum_w;
    spatial_moments *= inv_w2;
  }

  vec3 color_curr = texelFetch(tex_curr_sample, ipos, 0).rgb;

  vec3 out_color;
  vec4 out_moments_histlen;

  float grad = texelFetch(tex_grad_atrous_pong, ipos / GRAD_DWN, 0).r;
  grad = clamp(grad, 0.0, 1.1);

  if (temporal_sample_valid_diff) {
    // Compute the antilag factors based on the gradients
    float antilag_alpha = clamp(
        mix(1.0, denoising.flt_antilag * grad, denoising.flt_temporal), 0, 1);

    // Adjust the history length, taking the antilag factors into account
    float hist_len = min(
        temporal_moments_histlen.b * pow(1.0 - antilag_alpha, 10) + 1.0, 256.0);

    // Compute the blending weights based on history length, so that the filter
    // converges faster. I.e. the first frame has weight of 1.0, the second
    // frame 1/2, third 1/3 and so on.
    float alpha_color = max(denoising.flt_min_alpha_color, 1.0 / hist_len);
    float alpha_moments = max(denoising.flt_min_alpha_moments, 1.0 / hist_len);

    // Adjust the blending factors, taking the antilag factors into account
    // again
    alpha_color = mix(alpha_color, 1.0, antilag_alpha);
    alpha_moments = mix(alpha_moments, 1.0, antilag_alpha);

    // Blend!
    out_color.rgb = mix(temporal_color.rgb, color_curr.rgb, alpha_color);

    out_moments_histlen.rg =
        mix(temporal_moments_histlen.rg, spatial_moments.rg, alpha_moments);
    out_moments_histlen.b = hist_len;
  } else {
    // red --> invalid
    out_color.rgb = color_curr;
    out_moments_histlen = vec4(spatial_moments, 1, 1);
  }

  // Store the outputs for further processing by the a-trous filter
  imageStore(img_curr_hist_moments, ipos, out_moments_histlen);
  // imageStore(img_color_atrous_ping, ipos, motion);
  imageStore(img_color_atrous_ping, ipos, vec4(out_color, 1.0));
  imageStore(img_atrous_ping_moments, ipos, vec4(out_moments_histlen.xy, 0, 0));
}
