#version 430 core

#define BLUE_NOISE_RES (256)
#define NUM_BLUE_NOISE_TEX (128)
#define RNG_SEED_SHIFT_X 0u
#define RNG_SEED_SHIFT_Y 8u
#define RNG_SEED_SHIFT_ISODD 16u
#define RNG_SEED_SHIFT_FRAME 17u

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(binding = 0, r32ui) uniform restrict writeonly uimage2D t_out_rng_seed;

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
  ivec2 curr_coord = ivec2(gl_GlobalInvocationID);

  imageStore(t_out_rng_seed, curr_coord, uvec4(generate_rng_seed(curr_coord)));
}
