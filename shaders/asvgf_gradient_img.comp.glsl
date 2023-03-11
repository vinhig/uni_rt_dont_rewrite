#version 430 core

layout(binding = 0) uniform sampler2D t_curr_position;
layout(binding = 1) uniform sampler2D t_prev_position;
layout(binding = 2) uniform sampler2D t_curr_normal;
layout(binding = 3) uniform sampler2D t_prev_normal;
layout(binding = 4) uniform sampler2D t_curr_depth;
layout(binding = 5) uniform sampler2D t_prev_depth;

layout(binding = 6) uniform isampler2D t_curr_visibility;
layout(binding = 7) uniform isampler2D t_prev_visibility;

layout(binding = 8) uniform sampler2D t_curr_sample;
layout(binding = 9) uniform sampler2D t_prev_sample;

layout(binding = 10, rgba32f) uniform restrict image2D t_out_gradient;

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
  uint frame_number;
}
uniforms;

float get_gradient(float l_curr, float l_prev) {
  float l_max = max(l_curr, l_prev);

  if (l_max == 0.0) {
    return 0.0;
  }

  float ret = abs(l_curr - l_prev) / l_max;
  ret *= ret; // make small changes less significant

  return ret;
}

float get_view_depth(float depth, mat4 proj) {
  // Returns linear depth in [near, far]
  return proj[3][2] / (proj[2][2] + (depth * 2.0 - 1.0));
}

bool in_bounds(vec2 coord) {
  return all(lessThan(coord, uniforms.target_dim)) &&
         all(greaterThanEqual(coord, ivec2(0)));
}

float depth_weight(float prev_depth, float curr_depth, vec3 curr_normal,
                   vec3 view_dir, mat4 proj, float phi) {
  float linear_depth_prv = get_view_depth(prev_depth, proj);

  float linear_depth = get_view_depth(curr_depth, proj);

  float angle_factor = max(0.25, -dot(curr_normal, view_dir));

  float diff = abs(linear_depth_prv - linear_depth);
  return exp(-diff * angle_factor / phi);
}

float luminance(vec3 color) { return dot(color, vec3(0.2126, 0.7152, 0.0722)); }

float normal_weight(vec3 prev_normal, vec3 curr_normal, float phi) {
  // float d = max(0, dot(curr_normal, prev_normal));
  // return d * d;
  vec3 dd = prev_normal - curr_normal;
  float d = dot(dd, dd);
  return exp(-d / phi);
}

vec3 unproject_uv(float depth, vec2 uv, mat4 inv_view_proj) {
  float z = depth * 2.0 - 1.0;
  vec4 ndc = vec4(uv * 2.0 - 1.0, z, 1.0);
  vec4 world = inv_view_proj * ndc;
  return world.xyz / world.w;
}

float reproject_pixel_luminance() {
  ivec2 curr_coord = ivec2(gl_GlobalInvocationID);

  if (!in_bounds(curr_coord)) {
    return -1.0;
  }

  vec2 uv = (vec2(curr_coord) + 0.5) / uniforms.target_dim;

  // Actual reprojection from current sample to previous one happens here
  float current_depth = texelFetch(t_curr_depth, curr_coord, 0).x;
  vec3 current_world_position =
      unproject_uv(current_depth, uv, uniforms.inv_view_proj);
  vec4 clip_pos_prev =
      uniforms.prev_view_proj * vec4(current_world_position, 1.0);
  vec3 ndc_pos_prev = clip_pos_prev.xyz / clip_pos_prev.w;
  vec3 reprojected_uv = ndc_pos_prev;
  reprojected_uv.xy = ndc_pos_prev.xy * 0.5 + 0.5;
  reprojected_uv.z = ndc_pos_prev.z * 0.5 + 0.5;
  ivec2 prev_coord = ivec2(reprojected_uv.xy * uniforms.target_dim - 0.5);

  vec3 ray_dir = normalize(current_world_position - uniforms.view_pos.xyz).xyz;

  // Check if computed sample is the one from current sample
  // According to normals and depth
  int curr_visibility = texelFetch(t_curr_visibility, curr_coord, 0).x;
  int prev_visibility = texelFetch(t_prev_visibility, prev_coord, 0).x;
  float curr_depth = texelFetch(t_curr_depth, curr_coord, 0).x;
  float prev_depth = texelFetch(t_prev_depth, prev_coord, 0).x;
  vec3 curr_normal = texelFetch(t_curr_normal, curr_coord, 0).rgb;
  vec3 prev_normal = texelFetch(t_prev_normal, prev_coord, 0).rgb;

  float depth_error = depth_weight(prev_depth, curr_depth, curr_normal, ray_dir,
                                   uniforms.proj, uniforms.phi_depth);
  float normal_error =
      normal_weight(prev_normal, curr_normal, uniforms.phi_normal);

  if (depth_error > uniforms.depth_tolerance &&
      normal_error > uniforms.normal_tolerance &&
      curr_visibility == prev_visibility) {
    return luminance(texelFetch(t_prev_sample, prev_coord, 0).rgb);
  }

  return -1.0;
}

void main() {
  ivec2 ipos = ivec2(gl_GlobalInvocationID);
  // Check out of bounds
  if (any(greaterThanEqual(ipos, ivec2(1280, 720)))) {
    return;
  }

  float curr_luminance = luminance(texelFetch(t_curr_sample, ipos, 0).rgb);
  float prev_luminance = reproject_pixel_luminance();

  if (prev_luminance == -1.0) {
    return;
  } else {
    float grad = prev_luminance - curr_luminance;
    imageStore(t_out_gradient, ipos, vec4((vec3(grad) + 1.0) / 2, 1.0));
  }
}