#version 430

// Original implementation from
// https://github.com/gztong/BMFR-DXR-Denoiser/blob/master/BMFR_Denoiser/Data/regressionCP.hlsl
// https://github.com/maZZZu/bmfr/blob/master/opencl/bmfr.cl
// Major changes:
// - Whole picture is denoised instead of only the half
// - Translated to GLSL
// - A bunch of comments
// - Added depth buffer as a feature buffer
// - Ability to change block size

#define BUFFER_COUNT 13
#define FEATURES_COUNT 10
#define FEATURES_NOT_SCALED 4
#define BLOCK_PIXELS 1024
#define LOCAL_SIZE 512
#define BLOCK_EDGE_LENGTH 32
#define NOISE_AMOUNT 0.01
#define BLOCK_OFFSETS_COUNT 16

#define INBLOCK_ID sub_vector *LOCAL_SIZE + groupThreadId
#define BLOCK_OFFSET groupId.x *BUFFER_COUNT

layout(local_size_x = LOCAL_SIZE, local_size_y = 1, local_size_z = 1) in;

#if true
const ivec2 BLOCK_OFFSETS[BLOCK_OFFSETS_COUNT] =
    ivec2[](ivec2(-14, -14), ivec2(4, -6), ivec2(-8, 14), ivec2(8, 0),
            ivec2(-10, -8), ivec2(2, 12), ivec2(12, -12), ivec2(-10, 0),
            ivec2(12, 14), ivec2(-8, -16), ivec2(6, 6), ivec2(-2, -2),
            ivec2(6, -14), ivec2(-16, 12), ivec2(14, -4), ivec2(-6, 4));
#else
const ivec2 BLOCK_OFFSETS[BLOCK_OFFSETS_COUNT] =
    ivec2[](ivec2(-30),      //
            ivec2(-12, -22), //
            ivec2(-24, -2), ivec2(-8, -16), ivec2(-26, -24), ivec2(-14, -4),
            ivec2(-4, -28), ivec2(-26, -16),

            //  ivec2(-14, -8),
            //  ivec2(-16, -20),
            //  ivec2(-2, -18),
            //  ivec2(-30, -2),
            //  ivec2(-16, -16),
            //  ivec2(-4, -8),
            //  ivec2(-0, -0),
            //  ivec2(-26, -26),

            ivec2(-4, -2), ivec2(-24, -32), ivec2(-10), ivec2(-18),
            ivec2(-12, -30), ivec2(-32, -4), ivec2(-2, -20), ivec2(-22, -12));
#endif

// Feature buffers (from the G-Buffer)
layout(binding = 0) uniform sampler2D curr_position; // Current world position
layout(binding = 1) uniform sampler2D curr_normal;   // Current normal
layout(binding = 2) uniform sampler2D
    curr_depth; // Current depth (distance from camera to pixel sample)

// Noisy data accumulated by a previous pass
// Will be demodulated
layout(binding = 3) uniform sampler2D noisy_data;

// Temporary buffers to write
layout(binding = 4, r32f)
    uniform image2D out_data; // Values from feature buffers (with power applied
                              // for certain) Size is BLOCK_PIXELS *
                              // [(FEATURES_COUNT + color_channels) * blocks]
layout(binding = 5,
       r32f) uniform image2D tmp_data; // Values from QR decomposition

// Destination texture when the denoising is done
layout(binding = 6, rgba32f) uniform image2D render_texture;

layout(binding = 7, std140) uniform PerFrameCB {
  vec2 target_dim;
  int frame_number;
  int horizontal_blocks_count;
}
uniforms;

// Albedo (texture color), used to demodulate
// Otherwise textures will be blurred
layout(binding = 8) uniform sampler2D albedo;

shared float sum_vec[LOCAL_SIZE];
shared float block_max;
shared float block_min;
shared float uVec[BLOCK_PIXELS];
shared float vec_length;
shared float u_length_squared;
shared float rmat[FEATURES_COUNT][BUFFER_COUNT];
shared float dotV;
shared float gchannel[BLOCK_PIXELS];
shared float bchannel[BLOCK_PIXELS];

int mirror(inout int index, int size) {
  if (index < 0) {
    index = abs(index) - 1;
  } else {
    if (index >= size) {
      index = ((2 * size) - index) - 1;
    }
  }
  return index;
}

ivec2 mirror2(inout ivec2 index, ivec2 size) {
  index.x = mirror(index.x, size.x);
  index.y = mirror(index.y, size.y);

  return index;
}

float random(uint a) {
  a = (a + uint(2127912214u)) + (a << uint(12));
  a = (a ^ uint(3345072700u)) ^ (a >> uint(19));
  a = (a + uint(374761393u)) + (a << uint(5));
  a = (a + uint(3550635116u)) ^ (a << uint(9));
  a = (a + uint(4251993797u)) + (a << uint(3));
  a = (a ^ uint(3042594569u)) ^ (a >> uint(16));
  return float(a) / 4294967296.0;
}

float add_random(float value, int id, int sub_vector, int feature_buffer,
                 int frame_number) {
  return value + NOISE_AMOUNT * 2 *
                     (random(uint(id + sub_vector * LOCAL_SIZE +
                                  feature_buffer * BLOCK_EDGE_LENGTH *
                                      BLOCK_EDGE_LENGTH +
                                  uniforms.frame_number * BUFFER_COUNT *
                                      BLOCK_EDGE_LENGTH * BLOCK_EDGE_LENGTH)) -
                      0.5);
}

void main() {
  uvec3 groupId = gl_WorkGroupID;
  uint groupThreadId = gl_LocalInvocationIndex;

  for (uint sub_vector = 0; sub_vector < 4; sub_vector++) {
    uint index = (sub_vector * LOCAL_SIZE) + groupThreadId;
    ivec2 uv = ivec2(int(groupId.x % uint(uniforms.horizontal_blocks_count)),
                     int(groupId.x / uint(uniforms.horizontal_blocks_count)));
    uv *= ivec2(BLOCK_EDGE_LENGTH);
    uv += ivec2(int(index % BLOCK_EDGE_LENGTH), int(index / BLOCK_EDGE_LENGTH));
    uv += BLOCK_OFFSETS[uniforms.frame_number % BLOCK_OFFSETS_COUNT];

    // Avoid going out of screen when fetching sample
    uv = mirror2(uv, ivec2(uniforms.target_dim));

    // Extract value from feature buffers
    // And write them in tmp_data, apply power if needed
    float constant = 1.0;
    imageStore(tmp_data, ivec2(uvec2(index, 0 + (groupId.x * 13))),
               vec4(constant));

    vec4 normal = texelFetch(curr_normal, uv, 0);
    imageStore(tmp_data, ivec2(uvec2(index, 1 + BLOCK_OFFSET)), vec4(normal.x));
    imageStore(tmp_data, ivec2(uvec2(index, 2 + BLOCK_OFFSET)), vec4(normal.y));
    imageStore(tmp_data, ivec2(uvec2(index, 3 + BLOCK_OFFSET)), vec4(normal.z));

    vec4 depth = texelFetch(curr_depth, uv, 0);
    imageStore(tmp_data, ivec2(uvec2(index, 4 + BLOCK_OFFSET)), vec4(depth.x));
    imageStore(tmp_data, ivec2(uvec2(index, 5 + BLOCK_OFFSET)), vec4(depth.y));
    imageStore(tmp_data, ivec2(uvec2(index, 6 + BLOCK_OFFSET)), vec4(depth.z));

    vec4 position = texelFetch(curr_position, uv, 0);

    imageStore(tmp_data, ivec2(uvec2(index, 7 + BLOCK_OFFSET)),
               vec4(position.x * position.x));
    imageStore(tmp_data, ivec2(uvec2(index, 8 + BLOCK_OFFSET)),
               vec4(position.y * position.y));
    imageStore(tmp_data, ivec2(uvec2(index, 9 + BLOCK_OFFSET)),
               vec4(position.z * position.z));

    // vec4 albedo_value = texelFetch(albedo, uv, 0);
    vec4 noisy = texelFetch(noisy_data, uv, 0);

    float storeTemp_10 = noisy.x;
    imageStore(tmp_data, ivec2(uvec2(index, 10 + BLOCK_OFFSET)),
               vec4(storeTemp_10));
    float storeTemp_11 = noisy.x;
    imageStore(tmp_data, ivec2(uvec2(index, 11 + BLOCK_OFFSET)),
               vec4(storeTemp_11));
    float storeTemp_12 = noisy.x;
    imageStore(tmp_data, ivec2(uvec2(index, 12 + BLOCK_OFFSET)),
               vec4(storeTemp_12));
  }
  barrier();
  for (int feature_buffer = FEATURES_NOT_SCALED;
       feature_buffer < FEATURES_COUNT; feature_buffer++) {
    uint sub_vector = 0;

    float tmp_max =
        imageLoad(tmp_data, ivec2(INBLOCK_ID, feature_buffer + BLOCK_OFFSET)).x;
    float tmp_min = tmp_max;

    for (sub_vector = 1; sub_vector < 4; sub_vector++) {
      float value =
          imageLoad(tmp_data, ivec2(INBLOCK_ID, feature_buffer + BLOCK_OFFSET))
              .x;
      tmp_max = max(value, tmp_max);
      tmp_min = min(value, tmp_min);
    }
    sum_vec[groupThreadId] = tmp_max;
    barrier();
    if (groupThreadId < 256) {
      sum_vec[groupThreadId] =
          max(sum_vec[groupThreadId], sum_vec[groupThreadId + 256]);
    }
    barrier();
    if (groupThreadId < 128) {
      sum_vec[groupThreadId] =
          max(sum_vec[groupThreadId], sum_vec[groupThreadId + 128]);
    }
    barrier();
    if (groupThreadId < 64) {
      sum_vec[groupThreadId] =
          max(sum_vec[groupThreadId], sum_vec[groupThreadId + 64]);
    }
    barrier();
    if (groupThreadId < 32) {
      sum_vec[groupThreadId] =
          max(sum_vec[groupThreadId], sum_vec[groupThreadId + 32]);
    }
    barrier();
    if (groupThreadId < 16) {
      sum_vec[groupThreadId] =
          max(sum_vec[groupThreadId], sum_vec[groupThreadId + 16]);
    }
    barrier();
    if (groupThreadId < 8) {
      sum_vec[groupThreadId] =
          max(sum_vec[groupThreadId], sum_vec[groupThreadId + 8]);
    }
    barrier();
    if (groupThreadId < 4) {
      sum_vec[groupThreadId] =
          max(sum_vec[groupThreadId], sum_vec[groupThreadId + 4]);
    }
    barrier();
    if (groupThreadId < 2) {
      sum_vec[groupThreadId] =
          max(sum_vec[groupThreadId], sum_vec[groupThreadId + 2]);
    }
    barrier();
    if (groupThreadId == 0) {
      block_max = max(sum_vec[0], sum_vec[1]);
    }
    barrier();
    sum_vec[groupThreadId] = tmp_min;
    barrier();
    if (groupThreadId < 256) {
      sum_vec[groupThreadId] =
          min(sum_vec[groupThreadId], sum_vec[groupThreadId + 256]);
    }
    barrier();
    if (groupThreadId < 128) {
      sum_vec[groupThreadId] =
          min(sum_vec[groupThreadId], sum_vec[groupThreadId + 128]);
    }
    barrier();
    if (groupThreadId < 64) {
      sum_vec[groupThreadId] =
          min(sum_vec[groupThreadId], sum_vec[groupThreadId + 64]);
    }
    barrier();
    if (groupThreadId < 32) {
      sum_vec[groupThreadId] =
          min(sum_vec[groupThreadId], sum_vec[groupThreadId + 32]);
    }
    barrier();
    if (groupThreadId < 16) {
      sum_vec[groupThreadId] =
          min(sum_vec[groupThreadId], sum_vec[groupThreadId + 16]);
    }
    barrier();
    if (groupThreadId < 8) {
      sum_vec[groupThreadId] =
          min(sum_vec[groupThreadId], sum_vec[groupThreadId + 8]);
    }
    barrier();
    if (groupThreadId < 4) {
      sum_vec[groupThreadId] =
          min(sum_vec[groupThreadId], sum_vec[groupThreadId + 4]);
    }
    barrier();
    if (groupThreadId < 2) {
      sum_vec[groupThreadId] =
          min(sum_vec[groupThreadId], sum_vec[groupThreadId + 2]);
    }
    barrier();
    if (groupThreadId == 0) {
      block_min = min(sum_vec[0], sum_vec[1]);
    }
    barrier();
    if ((block_max - block_min) > 1.0) {
      for (uint sub_vector_2 = 0; sub_vector_2 < BLOCK_PIXELS / LOCAL_SIZE;
           sub_vector_2++) {
        float storeTemp_13 =
            (imageLoad(tmp_data,
                       ivec2(sub_vector_2 * LOCAL_SIZE + groupThreadId,
                             feature_buffer + BLOCK_OFFSET))
                 .x -
             block_min) /
            (block_max - block_min);
        imageStore(out_data,
                   ivec2(sub_vector_2 * LOCAL_SIZE + groupThreadId,
                         feature_buffer + BLOCK_OFFSET),
                   vec4(storeTemp_13));
        float storeTemp_14 =
            imageLoad(out_data, ivec2(sub_vector_2 * LOCAL_SIZE + groupThreadId,
                                      feature_buffer + BLOCK_OFFSET))
                .x;
        imageStore(tmp_data,
                   ivec2(sub_vector_2 * LOCAL_SIZE + groupThreadId,
                         feature_buffer + BLOCK_OFFSET),
                   vec4(storeTemp_14));
      }
    } else {
      for (uint sub_vector_3 = 0; sub_vector_3 < 4; sub_vector_3++) {
        float storeTemp_15 =
            imageLoad(tmp_data,
                      ivec2(uvec2(sub_vector_3 * LOCAL_SIZE + groupThreadId,
                                  feature_buffer + BLOCK_OFFSET)))
                .x -
            block_min;
        imageStore(out_data,
                   ivec2(uvec2(sub_vector_3 * LOCAL_SIZE + groupThreadId,
                               feature_buffer + BLOCK_OFFSET)),
                   vec4(storeTemp_15));
        float storeTemp_16 =
            imageLoad(out_data,
                      ivec2(uvec2(sub_vector_3 * LOCAL_SIZE + groupThreadId,
                                  feature_buffer + BLOCK_OFFSET)))
                .x;
        imageStore(tmp_data,
                   ivec2(uvec2(sub_vector_3 * LOCAL_SIZE + groupThreadId,
                               feature_buffer + BLOCK_OFFSET)),
                   vec4(storeTemp_16));
      }
    }
  }
  for (uint feature_buffer_1 = FEATURES_COUNT; feature_buffer_1 < BUFFER_COUNT;
       feature_buffer_1++) {
    for (uint sub_vector_4 = 0; sub_vector_4 < 4; sub_vector_4++) {
      float storeTemp_17 =
          imageLoad(tmp_data,
                    ivec2(uvec2(sub_vector_4 * LOCAL_SIZE + groupThreadId,
                                feature_buffer_1 + BLOCK_OFFSET)))
              .x;
      imageStore(out_data,
                 ivec2(uvec2(sub_vector_4 * LOCAL_SIZE + groupThreadId,
                             feature_buffer_1 + BLOCK_OFFSET)),
                 vec4(storeTemp_17));
    }
  }
  for (uint feature_buffer_2 = 0; feature_buffer_2 < FEATURES_NOT_SCALED;
       feature_buffer_2++) {
    for (uint sub_vector_5 = 0; sub_vector_5 < BLOCK_PIXELS / LOCAL_SIZE;
         sub_vector_5++) {
      float storeTemp_18 =
          imageLoad(tmp_data,
                    ivec2(uvec2(sub_vector_5 * LOCAL_SIZE + groupThreadId,
                                feature_buffer_2 + BLOCK_OFFSET)))
              .x;
      imageStore(out_data,
                 ivec2(uvec2(sub_vector_5 * LOCAL_SIZE + groupThreadId,
                             feature_buffer_2 + BLOCK_OFFSET)),
                 vec4(storeTemp_18));
    }
  }
  barrier();
  float r_value;
  float tmp_data_private_cache[BLOCK_PIXELS / LOCAL_SIZE];
  for (uint col = 0; col < FEATURES_COUNT; col++) {
    float tmp_sum_value = 0.0;
    for (uint sub_vector_6 = 0; sub_vector_6 < BLOCK_PIXELS / LOCAL_SIZE;
         sub_vector_6++) {
      int index_1 = int((sub_vector_6 * LOCAL_SIZE) + groupThreadId);
      float tmp =
          imageLoad(out_data, ivec2(uvec2(uint(index_1), col + BLOCK_OFFSET)))
              .x;
      uVec[index_1] = tmp;
      if (uint(index_1) >= (col + 1)) {
        tmp_sum_value += (tmp * tmp);
      }
    }
    sum_vec[groupThreadId] = tmp_sum_value;
    barrier();
    if (groupThreadId < 256) {
      sum_vec[groupThreadId] += sum_vec[groupThreadId + 256];
    }
    barrier();
    if (groupThreadId < 128) {
      sum_vec[groupThreadId] += sum_vec[groupThreadId + 128];
    }
    barrier();
    if (groupThreadId < 64) {
      sum_vec[groupThreadId] += sum_vec[groupThreadId + 64];
    }
    barrier();
    if (groupThreadId < 32) {
      sum_vec[groupThreadId] += sum_vec[groupThreadId + 32];
    }
    barrier();
    if (groupThreadId < 16) {
      sum_vec[groupThreadId] += sum_vec[groupThreadId + 16];
    }
    barrier();
    if (groupThreadId < 8) {
      sum_vec[groupThreadId] += sum_vec[groupThreadId + 8];
    }
    barrier();
    if (groupThreadId < 4) {
      sum_vec[groupThreadId] += sum_vec[groupThreadId + 4];
    }
    barrier();
    if (groupThreadId < 2) {
      sum_vec[groupThreadId] += sum_vec[groupThreadId + 2];
    }
    barrier();
    if (groupThreadId == 0) {
      vec_length = sum_vec[0] + sum_vec[1];
    }
    barrier();
    if (groupThreadId < col) {
      r_value = uVec[groupThreadId];
    } else {
      if (groupThreadId == col) {
        u_length_squared = vec_length;
        vec_length = sqrt(vec_length + (uVec[col] * uVec[col]));
        uVec[col] -= vec_length;
        u_length_squared += (uVec[col] * uVec[col]);
        r_value = vec_length;
      } else if (groupThreadId > col) {
        r_value = 0.0;
      }
    }
    if (groupThreadId < FEATURES_COUNT) {
      rmat[groupThreadId][col] = r_value;
    }

    for (uint feature_buffer_3 = col + 1; feature_buffer_3 < BUFFER_COUNT;
         feature_buffer_3++) {
      float tmp_sum_value_1 = 0.0;
      for (uint sub_vector_7 = 0; sub_vector_7 < BLOCK_PIXELS / LOCAL_SIZE;
           sub_vector_7++) {
        int index_2 = int((sub_vector_7 * LOCAL_SIZE) + groupThreadId);
        if (uint(index_2) >= col) {
          float tmp_1 =
              imageLoad(out_data, ivec2(uvec2(uint(index_2),
                                              feature_buffer_3 + BLOCK_OFFSET)))
                  .x;
          if ((col == 0) && (feature_buffer_3 < FEATURES_COUNT)) {
            tmp_1 =
                add_random(tmp_1, int(groupThreadId), int(sub_vector_7),
                           int(feature_buffer_3), int(uniforms.frame_number));
          }
          tmp_data_private_cache[sub_vector_7] = tmp_1;
          tmp_sum_value_1 += (tmp_1 * uVec[index_2]);
        }
      }
      sum_vec[groupThreadId] = tmp_sum_value_1;
      barrier();
      if (groupThreadId < 256) {
        sum_vec[groupThreadId] += sum_vec[groupThreadId + 256];
      }
      barrier();
      if (groupThreadId < 128) {
        sum_vec[groupThreadId] += sum_vec[groupThreadId + 128];
      }
      barrier();
      if (groupThreadId < 64) {
        sum_vec[groupThreadId] += sum_vec[groupThreadId + 64];
      }
      barrier();
      if (groupThreadId < 32) {
        sum_vec[groupThreadId] += sum_vec[groupThreadId + 32];
      }
      barrier();
      if (groupThreadId < 16) {
        sum_vec[groupThreadId] += sum_vec[groupThreadId + 16];
      }
      barrier();
      if (groupThreadId < 8) {
        sum_vec[groupThreadId] += sum_vec[groupThreadId + 8];
      }
      barrier();
      if (groupThreadId < 4) {
        sum_vec[groupThreadId] += sum_vec[groupThreadId + 4];
      }
      barrier();
      if (groupThreadId < 2) {
        sum_vec[groupThreadId] += sum_vec[groupThreadId + 2];
      }
      barrier();
      if (groupThreadId == 0) {
        dotV = sum_vec[0] + sum_vec[1];
      }
      barrier();
      for (uint sub_vector_8 = 0; sub_vector_8 < BLOCK_PIXELS / LOCAL_SIZE;
           sub_vector_8++) {
        int index_3 = int((sub_vector_8 * LOCAL_SIZE) + groupThreadId);
        if (uint(index_3) >= col) {
          float storeTemp_19 =
              tmp_data_private_cache[sub_vector_8] -
              (((2.0 * uVec[index_3]) * dotV) / u_length_squared);
          imageStore(
              out_data,
              ivec2(uvec2(uint(index_3), feature_buffer_3 + (groupId.x * 13))),
              vec4(storeTemp_19));
        }
      }
      barrier();
    }
  }
  if (groupThreadId < FEATURES_COUNT) {
    rmat[groupThreadId][FEATURES_COUNT] =
        imageLoad(out_data,
                  ivec2(uvec2(groupThreadId, FEATURES_COUNT + BLOCK_OFFSET)))
            .x;
  } else {
    uint tmpId = groupThreadId - FEATURES_COUNT;
    if (tmpId < FEATURES_COUNT) {
      rmat[tmpId][BUFFER_COUNT - 2] =
          imageLoad(out_data,
                    ivec2(uvec2(tmpId, BUFFER_COUNT - 2 + BLOCK_OFFSET)))
              .x;
    } else {
      tmpId = tmpId - FEATURES_COUNT;
      if (tmpId < FEATURES_COUNT) {
        rmat[tmpId][BUFFER_COUNT - 1] =
            imageLoad(out_data,
                      ivec2(uvec2(tmpId, BUFFER_COUNT - 1 + (groupId.x * 13))))
                .x;
      }
    }
  }
  barrier();
  for (int i = BUFFER_COUNT - 4; i >= 0; i--) {
    if (groupThreadId < 3) {
      rmat[i][(BUFFER_COUNT - groupThreadId) - 1] /= rmat[i][i];
    }
    barrier();
    if (groupThreadId < uint(3 * i)) {
      uint rowId = (uint(i) - (groupThreadId / 3)) - 1;
      uint channel = (BUFFER_COUNT - (groupThreadId % 3)) - 1;
      rmat[rowId][channel] -= (rmat[i][channel] * rmat[rowId][i]);
    }
    barrier();
  }
  for (uint sub_vector_9 = 0; sub_vector_9 < BLOCK_PIXELS / LOCAL_SIZE;
       sub_vector_9++) {
    uint index_4 = (sub_vector_9 * LOCAL_SIZE) + groupThreadId;
    uVec[index_4] = 0.0;
    gchannel[index_4] = 0.0;
    bchannel[index_4] = 0.0;
  }
  for (int col_1 = 0; col_1 < FEATURES_COUNT; col_1++) {
    for (uint sub_vector_10 = 0; sub_vector_10 < BLOCK_PIXELS / LOCAL_SIZE;
         sub_vector_10++) {
      uint index_5 = (sub_vector_10 * LOCAL_SIZE) + groupThreadId;
      float tmp_2 =
          imageLoad(tmp_data,
                    ivec2(uvec2(index_5, uint(col_1) + (groupId.x * 13))))
              .x;
      uVec[index_5] += (rmat[col_1][FEATURES_COUNT] * tmp_2);
      gchannel[index_5] += (rmat[col_1][FEATURES_COUNT + 1] * tmp_2);
      bchannel[index_5] += (rmat[col_1][FEATURES_COUNT + 2] * tmp_2);
    }
  }
  for (uint sub_vector_11 = 0; sub_vector_11 < BLOCK_PIXELS / LOCAL_SIZE;
       sub_vector_11++) {
    uint index_6 = (sub_vector_11 * LOCAL_SIZE) + groupThreadId;
    ivec2 uv_1 = ivec2(int(groupId.x % uint(uniforms.horizontal_blocks_count)),
                       int(groupId.x / uint(uniforms.horizontal_blocks_count)));
    uv_1 *= ivec2(32);
    uv_1 += ivec2(int(index_6 % 32), int(index_6 / 32));
    uv_1 += BLOCK_OFFSETS[uniforms.frame_number % 16];
    if ((((uv_1.x < 0) || (uv_1.y < 0)) || (uv_1.x >= uniforms.target_dim.x)) ||
        (uv_1.y >= uniforms.target_dim.y)) {
      continue;
    }
    vec4 storeTemp_20 =
        vec4((uVec[index_6] < 0.0) ? 0.0 : uVec[index_6],
             (gchannel[index_6] < 0.0) ? 0.0 : gchannel[index_6],
             (bchannel[index_6] < 0.0) ? 0.0 : bchannel[index_6],
             texelFetch(noisy_data, uv_1, 1).w);
    imageStore(render_texture, uv_1, storeTemp_20);
  }
}
