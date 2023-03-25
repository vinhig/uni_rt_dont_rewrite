#version 430 core

// Original implementation from Q2RTX
// https://github.com/NVIDIA/Q2RTX/blob/master/src/refresh/vkpt/shader/asvgf_gradient_reproject.comp
// Main changes:
// - reprojection slightly different (to be kept similar as the default
// reprojection in ta.comp.glsl, and the original svgf reprojection)
// - denoising isn't applied on separated noisy data, but rather on the whole
//   image generated by the embree backend (Q2RTX works on diffuse and specular,
//   while we work on both at the same time)
// - i still don't understand why every images are splitted into two parts in
//   q2rtx... here it's only one ofc (maybe for VR stuff idk)

#define GRAD_DWN (3)
#define GROUP_SIZE_GRAD 8
#define GROUP_SIZE_PIXELS (GROUP_SIZE_GRAD * GRAD_DWN)
#define BLUE_NOISE_RES (256)
#define NUM_BLUE_NOISE_TEX (128)
#define RNG_SEED_SHIFT_X 0u
#define RNG_SEED_SHIFT_Y 8u
#define RNG_SEED_SHIFT_ISODD 16u
#define RNG_SEED_SHIFT_FRAME 17u

#extension GL_ARB_enhanced_layouts : enable // to be allowed to use compile time
                                            // expression in some place (as in
                                            // layout definition)

layout(local_size_x = GROUP_SIZE_PIXELS, local_size_y = GROUP_SIZE_PIXELS,
       local_size_z = 1) in;

layout(binding = 0) uniform sampler2D t_curr_normal;
layout(binding = 1) uniform sampler2D t_prev_normal;
layout(binding = 2) uniform sampler2D t_curr_depth;
layout(binding = 3) uniform sampler2D t_prev_depth;

layout(binding = 4) uniform sampler2D t_curr_sample;
layout(binding = 5) uniform sampler2D t_prev_sample;

layout(binding = 6) uniform isampler2D t_curr_rng_seed;
layout(binding = 7) uniform isampler2D t_prev_rng_seed;

layout(binding = 8) uniform restrict writeonly image2D t_out_gradient;
layout(binding = 9, r32i) uniform restrict writeonly iimage2D t_out_rng_seed;

shared vec4 reprojected_pixels[GROUP_SIZE_PIXELS][GROUP_SIZE_PIXELS];

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

void reproject_pixel(ivec2 ipos) {
  // First pass, for all pixel in current frame, match the pixel of the previous
  // frame
  // All threads on the job
  ivec2 local_pos = ivec2(gl_LocalInvocationID);

  // Init
  reprojected_pixels[local_pos.y][local_pos.x] = vec4(0.0);

  vec2 uv = (vec2(ipos) + 0.5) / uniforms.target_dim;

  // Actual reprojection from current sample to previous one happens here
  float current_depth = texelFetch(t_curr_depth, ipos, 0).x;
  vec3 current_world_position =
      unproject_uv(current_depth, uv, uniforms.inv_view_proj);
  vec4 clip_pos_prev =
      uniforms.prev_view_proj * vec4(current_world_position, 1.0);
  vec3 ndc_pos_prev = clip_pos_prev.xyz / clip_pos_prev.w;
  vec3 reprojected_uv = ndc_pos_prev;
  reprojected_uv.xy = ndc_pos_prev.xy * 0.5 + 0.5;
  reprojected_uv.z = ndc_pos_prev.z * 0.5 + 0.5;
  ivec2 prev_coord = ivec2(reprojected_uv.xy * uniforms.target_dim - 0.5);

  if (!in_bounds(prev_coord)) {
    return;
  }

  vec3 ray_dir = normalize(current_world_position - uniforms.view_pos.xyz).xyz;

  // Check if computed sample is the one from current sample
  // According to normals and depth
  float curr_depth = texelFetch(t_curr_depth, ipos, 0).x;
  float prev_depth = texelFetch(t_prev_depth, prev_coord, 0).x;
  vec3 curr_normal = texelFetch(t_curr_normal, ipos, 0).rgb;
  vec3 prev_normal = texelFetch(t_prev_normal, prev_coord, 0).rgb;

  float depth_error = depth_weight(prev_depth, curr_depth, curr_normal, ray_dir,
                                   uniforms.proj, uniforms.phi_depth);
  float normal_error =
      normal_weight(prev_normal, curr_normal, uniforms.phi_normal);

  if (depth_error > uniforms.depth_tolerance &&
      normal_error > uniforms.normal_tolerance) {
    float prev_luminance =
        luminance(texelFetch(t_prev_sample, prev_coord, 0).rgb);

    reprojected_pixels[local_pos.y][local_pos.x] =
        vec4(ivec2(reprojected_uv.xy * uniforms.target_dim - 0.5),
             prev_luminance, 1.0);
  }
}

uint generate_rng_seed(ivec2 ipos) {
  int frame_num = int(uniforms.frame_number);

  uint frame_offset = frame_num / NUM_BLUE_NOISE_TEX;

  uint rng_seed = 0;
  rng_seed |= (uint(ipos.x + frame_offset) % BLUE_NOISE_RES)
              << RNG_SEED_SHIFT_X;
  rng_seed |= (uint(ipos.y + (frame_offset << 4)) % BLUE_NOISE_RES)
              << RNG_SEED_SHIFT_Y;
  rng_seed |= uint(false) << RNG_SEED_SHIFT_ISODD;
  rng_seed |= uint(frame_num) << RNG_SEED_SHIFT_FRAME;

  return rng_seed;
}

void main() {
  // ipos is the position inside the 1280x720 texture yes yes
  ivec2 ipos = ivec2(gl_GlobalInvocationID);
  // pos_grad is the position inside the 1280/3x720/3 texture yes yes
  ivec2 pos_grad = ipos / (GRAD_DWN);

  imageStore(t_out_rng_seed, ipos, ivec4(generate_rng_seed(ipos)));

  // Reproject the current pixel
  // Won't save it in the texture as i thought before
  // but rather in a shared buffer -> gonna be sorted in the second pass (find a
  // random sample in 3x3 block)
  reproject_pixel(ipos);

  // Waiting for all threads to finish
  barrier();

  ivec2 local_pos;
  local_pos.x = int(gl_LocalInvocationIndex) % GROUP_SIZE_GRAD;
  local_pos.y = int(gl_LocalInvocationIndex) / GROUP_SIZE_GRAD;

  if (local_pos.y >= GROUP_SIZE_GRAD)
    return;

  pos_grad = ivec2(gl_WorkGroupID) * GROUP_SIZE_GRAD + local_pos;
  ipos = pos_grad * GRAD_DWN;

  bool found = false;
  ivec2 found_offset = ivec2(0);
  ivec2 found_pos_prev = ivec2(0);
  vec4 found_prev_lum = vec4(0.0, 0.0, 0.0, 1.0);
  // Here is the implementation of Q2RTX which is picking the sample with the
  // highest luminance ...
  for (int offy = 0; offy < GRAD_DWN; offy++) {
    for (int offx = 0; offx < GRAD_DWN; offx++) {
      ivec2 p = local_pos * GRAD_DWN + ivec2(offx, offy);

      vec4 reprojected_pixel = reprojected_pixels[p.y][p.x];

      vec4 prev_lum = reprojected_pixel;

      if (prev_lum.z > found_prev_lum.z) {
        found_prev_lum = prev_lum;
        found_offset = ivec2(offx, offy);
        found_pos_prev = ivec2(reprojected_pixel.xy);
        found = true;
      }
    }
  }

  // ... but here is the theory of the original a-svgf paper: pick a random
  // sample
  // TODO

  if (!found) {
    imageStore(t_out_gradient, pos_grad, vec4(0.0));
    return;
  }

  imageStore(t_out_gradient, pos_grad, found_prev_lum);
  imageStore(t_out_rng_seed, ipos + found_offset,
             ivec4(texelFetch(t_prev_rng_seed, found_pos_prev, 0)));
}