#version 430 core

// Original implementation from
// https://github.com/JuanDiegoMontoya/Fwog/blob/examples-refactor/example/shaders/rsm/Reproject2.comp.glsl
// Main modificiations:
// - use of a vibility buffer to early stop accumulation

const uint k_radius = 1;
const uint k_width = 1 + 2 * k_radius;
const float kernel1D[k_width] = {0.27901, 0.44198, 0.27901};
const float kernel[k_width][k_width] = {
    {kernel1D[0] * kernel1D[0], kernel1D[0] * kernel1D[1],
     kernel1D[0] * kernel1D[2]},
    {kernel1D[1] * kernel1D[0], kernel1D[1] * kernel1D[1],
     kernel1D[1] * kernel1D[2]},
    {kernel1D[2] * kernel1D[0], kernel1D[2] * kernel1D[1],
     kernel1D[2] * kernel1D[2]},
};

// layout(binding = 0) uniform sampler2D t_curr_position;
layout(binding = 1) uniform sampler2D t_curr_normal;
// layout(binding = 2) uniform sampler2D t_prev_position;
layout(binding = 3) uniform sampler2D t_prev_normal;

layout(binding = 4) uniform sampler2D t_prev_moments;

layout(binding = 5) uniform sampler2D t_curr_indirect;
layout(binding = 6) uniform sampler2D t_prev_accumulated;

layout(binding = 7) uniform restrict writeonly image2D t_out_accumulated;
layout(binding = 8) uniform restrict writeonly image2D t_out_moments;
layout(binding = 9, r8ui) uniform restrict uimage2D t_out_history_length;

layout(binding = 11) uniform sampler2D t_curr_depth;
layout(binding = 12) uniform sampler2D t_prev_depth;

layout(binding = 13) uniform isampler2D t_curr_visibility;
layout(binding = 14) uniform isampler2D t_prev_visibility;

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

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

float get_view_depth(float depth, mat4 proj) {
  // Returns linear depth in [near, far]
  return proj[3][2] / (proj[2][2] + (depth * 2.0 - 1.0));
}

float depth_weight(float depthPrev, float depthCur, vec3 normalCur,
                   vec3 viewDir, mat4 proj, float phi) {
  float linearDepthPrev = get_view_depth(depthPrev, proj);

  float linearDepth = get_view_depth(depthCur, proj);

  float angleFactor = max(0.25, -dot(normalCur, viewDir));

  float diff = abs(linearDepthPrev - linearDepth);
  return exp(-diff * angleFactor / phi);
}

float normal_weight(vec3 normalPrev, vec3 normalCur, float phi) {
  // float d = max(0, dot(normalCur, normalPrev));
  // return d * d;
  vec3 dd = normalPrev - normalCur;
  float d = dot(dd, dd);
  return exp(-d / phi);
}

vec3 unproject_uv(float depth, vec2 uv, mat4 invXProj) {
  float z = depth * 2.0 - 1.0; // OpenGL Z convention
  vec4 ndc = vec4(uv * 2.0 - 1.0, z, 1.0);
  vec4 world = invXProj * ndc;
  return world.xyz / world.w;
}

void accumulate(vec3 prev_color, vec3 curr_color, vec2 prev_moments,
                vec2 curr_moments, ivec2 gid) {
  uint history_length = min(1 + imageLoad(t_out_history_length, gid).x, 255);
  imageStore(t_out_history_length, gid, uvec4(history_length));

  float alpha_illum = max(uniforms.alpha_illum, 1.0 / float(history_length));
  float alpha_moments =
      max(uniforms.alpha_moments, 1.0 / float(history_length));

  vec3 out_color = mix(prev_color, curr_color, alpha_illum);
  vec2 out_moments = mix(prev_moments, curr_moments, alpha_moments);

  imageStore(t_out_accumulated, gid, vec4(out_color, 0.0));
  imageStore(t_out_moments, gid, vec4(out_moments, 0.0, 0.0));
}

vec3 bilerp(vec3 _00, vec3 _01, vec3 _10, vec3 _11, vec2 weight) {
  vec3 bottom = mix(_00, _10, weight.x);
  vec3 top = mix(_01, _11, weight.x);
  return mix(bottom, top, weight.y);
}

vec2 bilerp(vec2 _00, vec2 _01, vec2 _10, vec2 _11, vec2 weight) {
  vec2 bottom = mix(_00, _10, weight.x);
  vec2 top = mix(_01, _11, weight.x);
  return mix(bottom, top, weight.y);
}

float bilerp(float _00, float _01, float _10, float _11, vec2 weight) {
  float bottom = mix(_00, _10, weight.x);
  float top = mix(_01, _11, weight.x);
  return mix(bottom, top, weight.y);
}

float luminance(vec3 color) { return dot(color, vec3(0.2126, 0.7152, 0.0722)); }

bool in_bounds(vec2 coord) {
  return all(lessThan(coord, uniforms.target_dim)) &&
         all(greaterThanEqual(coord, ivec2(0)));
}

void main() {
  ivec2 curr_coord = ivec2(gl_GlobalInvocationID.xy);

  if (!in_bounds(curr_coord)) {
    return;
  }

  vec2 uv = (vec2(curr_coord) + 0.5) / uniforms.target_dim;

  // Reproject this pixel
  float current_depth = texelFetch(t_curr_depth, curr_coord, 0).x;
  vec3 current_world_position =
      unproject_uv(current_depth, uv, uniforms.inv_view_proj);
  vec4 clip_pos_prev =
      uniforms.prev_view_proj * vec4(current_world_position, 1.0);
  vec3 ndc_pos_prev = clip_pos_prev.xyz / clip_pos_prev.w;
  vec3 reprojected_uv = ndc_pos_prev;
  reprojected_uv.xy = ndc_pos_prev.xy * 0.5 + 0.5;
  reprojected_uv.z = ndc_pos_prev.z * 0.5 + 0.5;

  vec3 ray_dir = normalize(current_world_position - uniforms.view_pos.xyz).xyz;

  vec3 current_world_normal = texelFetch(t_curr_normal, curr_coord, 0).xyz;

  ivec2 bottom_left_pos = ivec2(reprojected_uv.xy * uniforms.target_dim - 0.5);
  vec3 colors[2][2] =
      vec3[2][2](vec3[2](vec3(0), vec3(0)), vec3[2](vec3(0), vec3(0)));
  vec2 moments[2][2] =
      vec2[2][2](vec2[2](vec2(0), vec2(0)), vec2[2](vec2(0), vec2(0)));
  float valid[2][2] = float[2][2](float[2](0, 0), float[2](0, 0));

  int valid_count = 0;

  for (int y = 0; y <= 1; y++) {
    for (int x = 0; x <= 1; x++) {
      ivec2 pos = bottom_left_pos + ivec2(x, y);

      if (!in_bounds(pos)) {
        continue;
      }

      float depth_prev = texelFetch(t_prev_depth, pos, 0).x;
      if (depth_weight(depth_prev, current_depth, current_world_normal.xyz,
                       ray_dir, uniforms.proj,
                       uniforms.phi_depth) < uniforms.depth_tolerance) {
        continue;
      }

      vec3 normal_prev = texelFetch(t_prev_normal, pos, 0).xyz;
      if (normal_weight(normal_prev, current_world_normal.xyz,
                        uniforms.phi_normal) < uniforms.normal_tolerance) {
        continue;
      }

      valid_count += 1;
      valid[x][y] = 1.0;
      colors[x][y] = texelFetch(t_prev_accumulated, pos, 0).xyz;
      moments[x][y] = texelFetch(t_prev_moments, pos, 0).xy;
    }
  }

  vec2 weight = fract(reprojected_uv.xy * uniforms.target_dim - 0.5);
  vec3 curr_color = texelFetch(t_curr_indirect, curr_coord, 0).xyz;
  float lum = luminance(curr_color);
  vec2 curr_moments = {lum, lum * lum};

  if (valid_count > 0) {
    float factor = max(0.01, bilerp(valid[0][0], valid[0][1], valid[1][0],
                                    valid[1][1], weight));
    vec3 prev_color =
        bilerp(colors[0][0], colors[0][1], colors[1][0], colors[1][1], weight) /
        factor;
    vec2 prev_moments = bilerp(moments[0][0], moments[0][1], moments[1][0],
                               moments[1][1], weight) /
                        factor;

    accumulate(prev_color, curr_color, prev_moments, curr_moments, curr_coord);
  } else {
    ivec2 center_pos = ivec2(reprojected_uv.xy * uniforms.target_dim);
    vec3 accum_illuminance = vec3(0.0);
    vec2 accum_moments = vec2(0.0);
    float accum_weight = 0.0;

    for (int col = 0; col < k_width; col++) {
      for (int row = 0; row < k_width; row++) {
        ivec2 offset = ivec2(row - k_radius, col - k_radius);
        ivec2 pos = center_pos + offset;

        if (!in_bounds(pos)) {
          continue;
        }

        float kernel_weight = kernel[row][col];

        vec3 o_color = texelFetch(t_prev_accumulated, pos, 0).rgb;
        vec2 o_moments = texelFetch(t_prev_moments, pos, 0).xy;
        vec3 o_normal = texelFetch(t_prev_normal, pos, 0).xyz;
        float o_depth = texelFetch(t_prev_depth, pos, 0).x;

        float phi_depth = offset == ivec2(0.0) ? 1.0 : length(vec2(offset));
        phi_depth *= uniforms.phi_depth;

        float v_normal_weight = normal_weight(
            o_normal, current_world_normal.xyz, uniforms.phi_normal);
        float v_depth_weight =
            depth_weight(o_depth, current_depth, current_world_normal.xyz,
                         ray_dir, uniforms.proj, uniforms.phi_depth);

        float weight = v_normal_weight * v_depth_weight;
        accum_illuminance += o_color * weight * kernel_weight;
        accum_moments += o_moments * weight * kernel_weight;
        accum_weight += weight * kernel_weight;
      }
    }

    if (accum_weight >= uniforms.min_accum_weight) {
      vec3 prev_color = accum_illuminance / accum_weight;
      vec2 prev_moments = accum_moments / accum_weight;

      accumulate(prev_color, curr_color, prev_moments, curr_moments,
                 curr_coord);
    } else {
      imageStore(t_out_accumulated, curr_coord, vec4(curr_color, 0.0));
      imageStore(t_out_moments, curr_coord, vec4(curr_moments, 0.0, 0.0));
      imageStore(t_out_history_length, curr_coord, uvec4(0));
    }
  }
}
