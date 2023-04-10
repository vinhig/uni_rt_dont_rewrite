#version 430

#define BLOCK_SIZE 16
#define W 16
#define S_W 4
#define M 10
// 1, NORM_X, NORM_Y, NORM_Z, POS_X, POS_Y, POS_Z, POS_X², POS_Y², POS_Z²
#define NOISE_AMOUNT 0.1

#define FEATURES_NOT_SCALED 4

#define BLOCK_OFFSETS_COUNT 16
const vec2 BLOCK_OFFSETS[BLOCK_OFFSETS_COUNT] = {
    vec2(0.44488132549030435, 0.31563986695199087),
    vec2(0.1148786159292483, 0.8548156864156085),
    vec2(-0.002148183176659213, -0.6542196608276174),
    vec2(-0.10773558199980471, -0.865014821847512),
    vec2(-0.1452097231468079, -0.361160178632681),
    vec2(-0.13079792159436576, 0.45205461311007844),
    vec2(0.49412769555245295, -0.025080750942926278),
    vec2(-0.8946359720290911, 0.3775707422982397),
    vec2(0.6185343503380443, 0.8431189038845106),
    vec2(-0.32354170620402933, -0.8860494747143273),
    vec2(0.16326567660620794, -0.55978452888754),
    vec2(0.3894441854425761, -0.42399912088548075),
    vec2(-0.6321152784081832, -0.664452722309357),
    vec2(0.3814582754823064, 0.8480884948253038),
    vec2(0.6478505901925895, -0.17737063699748123),
    vec2(0.08139039642641643, 0.9898172365586706),
};

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) uniform sampler2D tex_pos;
layout(binding = 1) uniform sampler2D tex_normal;
layout(binding = 2) uniform sampler2D tex_depth;
layout(binding = 3) uniform sampler2D tex_indirect;
layout(binding = 4) uniform sampler2D tex_albedo;

layout(binding = 5, rgba32f) uniform writeonly image2D tex_out;

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
  uint current_frame;
}
uniforms;

layout(std430, binding = 5) buffer debug_1 { float debug_tilde[W][M + 1]; };

layout(std430, binding = 6) buffer debug_2 { float debug_h[W][M + 1]; };

layout(std430, binding = 7) buffer debug_3 { float debug_alpha_red[W]; };
layout(std430, binding = 8) buffer debug_4 { float debug_alpha_green[W]; };
layout(std430, binding = 9) buffer debug_5 { float debug_alpha_blue[W]; };

float random(uint a) {
  a = (a + uint(2127912214u)) + (a << uint(12));
  a = (a ^ uint(3345072700u)) ^ (a >> uint(19));
  a = (a + uint(374761393u)) + (a << uint(5));
  a = (a + uint(3550635116u)) ^ (a << uint(9));
  a = (a + uint(4251993797u)) + (a << uint(3));
  a = (a ^ uint(3042594569u)) ^ (a >> uint(16));
  return float(a) / 4294967296.0;
}

float add_random(int x, int y, int feature_buffer) {
  return NOISE_AMOUNT * 2 *
         (random(uint(x + y + x * y + feature_buffer + W + M + x * W +
                      BLOCK_SIZE * M * uniforms.current_frame)) -
          0.5);
}

void add_vec(float vec_a[W], float vec_b[W], out float vec_r[W], int k) {
  for (int i = 0; i < W - k; i++) {
    vec_r[i] = vec_a[i] + vec_b[i];
  }
}

void mul_vec_s(float vec_a[W], float b, out float vec_r[W], int k) {
  for (int i = 0; i < W - k; i++) {
    vec_r[i] = vec_a[i] * b;
  }
}

void mul_vec(float vec_a[W], float vec_b[W], out float vec_r[W], int k) {
  for (int i = 0; i < W - k; i++) {
    vec_r[i] = vec_a[i] * vec_b[i];
  }
}

void sub_mat_w(float mat_a[W][W], float mat_b[W][W], int k,
               out float mat_r[W][W]) {
  for (int x = 0; x < W; x++) {
    for (int y = 0; y < W; y++) {
      mat_r[x][y] = mat_a[x][y];
    }
  }
  for (int x = k; x < W; x++) {
    for (int y = k; y < W; y++) {
      mat_r[x][y] = mat_a[x][y] - mat_b[x - k][y - k];
    }
  }
}

float norm_k(float vec[W], int k) {
  float sum = 0.0;
  for (int i = 0; i < W - k; i++) {
    sum += vec[i] * vec[i];
  }

  return sqrt(sum);
}

void identity(out float a[W][W]) {
  for (int x = 0; x < W; x++) {
    for (int y = 0; y < W; y++) {
      if (x == y) {
        a[x][y] = 1.0;
      } else {
        a[x][y] = 0.0;
      }
    }
  }
}

void householder_matrix(out float H[W][W], float v[W], int k) {
  float new_H[W][W];
  identity(new_H);

  float uut;
  for (int i = 0; i < W - k; i++) {
    uut += v[i] * v[i];
  }

  float utu[W][W];
  for (int x = 0; x < W - k; x++) {
    for (int y = 0; y < W - k; y++) {
      utu[y][x] = v[x] * v[y] / uut * 2;
    }
  }

  sub_mat_w(new_H, utu, k, H);
}

void householder_step(float T_tilde[W][M + 1], out float H[W][W], int k) {
  // Compute the reflection of the k column
  float alpha[W];

  for (int i = 0; i < W - k; i++) {
    alpha[i] = T_tilde[i + k][k];
  }

  float sign_a0 = sign(alpha[0]);
  float norm_a = norm_k(alpha, k);

  float e[W];
  e[0] = 1.0;
  for (int i = 1; i < W; i++) {
    e[i] = 0.0;
  }

  float v[W];
  mul_vec_s(e, sign_a0 * norm_a, v, k);
  add_vec(alpha, v, v, k);

  // Compute the corresponding householder matrix
  float h[W][W];

  householder_matrix(h, v, k);

  for (int w = 0; w < W; w++) {
    for (int m = 0; m < W; m++) {
      H[w][m] = h[w][m];
    }
  }
}

void mul_mat_H(float H[W][W], float A[W][M + 1], out float R[W][M + 1]) {
  for (int i = 0; i < W; i++) {
    for (int j = 0; j < M + 1; j++) {
      for (int l = 0; l < W; l++) {
        R[i][j] += H[i][l] * A[l][j];
      }
    }
  }
}

void householder_qr(float T_tilde[W][M + 1], out float R[W][M + 1]) {
  float H0[W][W];
  float A0[W][M + 1];

  householder_step(T_tilde, H0, 0);
  mul_mat_H(H0, T_tilde, A0);

  float H1[W][W];
  float A1[W][M + 1];

  householder_step(A0, H1, 1);
  mul_mat_H(H1, A0, A1);

  float H2[W][W];
  float A2[W][M + 1];

  householder_step(A1, H2, 2);
  mul_mat_H(H2, A1, A2);

  float H3[W][W];
  float A3[W][M + 1];

  householder_step(A2, H3, 3);
  mul_mat_H(H3, A2, A3);

  float H4[W][W];
  float A4[W][M + 1];

  householder_step(A3, H4, 4);
  mul_mat_H(H4, A3, A4);

  float H5[W][W];
  float A5[W][M + 1];

  householder_step(A4, H5, 5);
  mul_mat_H(H5, A4, A5);

  float H6[W][W];
  float A6[W][M + 1];

  householder_step(A5, H6, 6);
  mul_mat_H(H6, A5, A6);

  float H7[W][W];
  float A7[W][M + 1];

  householder_step(A6, H7, 7);
  mul_mat_H(H7, A6, A7);

  float H8[W][W];
  float A8[W][M + 1];

  householder_step(A7, H8, 8);
  mul_mat_H(H8, A7, A8);

  float H9[W][W];
  float A9[W][M + 1];

  householder_step(A8, H9, 9);
  mul_mat_H(H9, A8, A9);

  float H10[W][W];
  float A10[W][M + 1];

  householder_step(A9, H10, 10);
  mul_mat_H(H10, A9, A10);

  for (int w = 0; w < W; w++) {
    for (int m = 0; m < M + 1; m++) {
      R[w][m] = A10[w][m];
    }
  }

  return;
}

void resolve(float R[M][M], float r_c[M], out float a[M]) {
  for (int i = 0; i < M; i++) {
    a[i] = 0.0;
  }

  if (R[M - 1][M - 1] != 0) {
    a[M - 1] = r_c[M - 1] / R[M - 1][M - 1];
  } else {
    a[M - 1] = 0.0;
  }

  for (int i = M - 2; i >= 0; i--) {
    float sum = 0.0;
    for (int j = i + 1; j < M; j++) {
      sum += R[i][j] * a[j];
    }
    if (R[i][i] != 0) {
      a[i] = (r_c[i] - sum) / R[i][i];
    } else {
      a[i] = 0.0;
    }
  }
}

float dot_m(float a[M], float b[M]) {
  float sum = 0.0;
  for (int i = 0; i < M; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy) * BLOCK_SIZE;

  // vec2 offset =
  //     RELATIVE_OFFSETS[uniforms.current_frame % OFFSETS_COUNT] - vec2(0.5);
  // offset *= 16.0;
  // offset *= vec2(BLOCK_SIZE);
  coord += ivec2(BLOCK_OFFSETS[uniforms.current_frame % BLOCK_OFFSETS_COUNT] * BLOCK_SIZE);

  if (coord.x > uniforms.target_dim.x || coord.y > uniforms.target_dim.y) {
    return;
  }

  for (int offx = 0; offx < BLOCK_SIZE; offx++) {
    for (int offy = 0; offy < BLOCK_SIZE; offy++) {
      imageStore(tex_out, coord + ivec2(offx, offy), vec4(0.0));
    }
  }

  // coord *= ivec2(BLOCK_SIZE, BLOCK_SIZE);

  // Build T_tilde
  float T_tilde[W][M + 1];

  for (int i = 0; i < W; i++) {
    T_tilde[i][0] = 1.0;
  }

  float max_values[M];
  float min_values[M];
  float mag_values[M];

  for (int x = FEATURES_NOT_SCALED; x < M; x++) {
    max_values[x] = 0.0;
    min_values[x] = 0.0;
    mag_values[x] = 0.0;
  }

  for (int i = 0; i < W; i++) {
    int x = (i % S_W) * (BLOCK_SIZE / S_W);
    int y = (i / S_W) * (BLOCK_SIZE / S_W);
    ivec2 local_coord = coord + ivec2(x, y);
    vec4 norm = texelFetch(tex_normal, local_coord, 0);
    vec4 pos = texelFetch(tex_pos, local_coord, 0);
    vec4 alb = texelFetch(tex_albedo, local_coord, 0);
    T_tilde[i][1] = norm.x;
    T_tilde[i][2] = norm.y;
    T_tilde[i][3] = norm.z;
    T_tilde[i][4] = pos.x;
    T_tilde[i][5] = pos.y;
    T_tilde[i][6] = pos.z;
    T_tilde[i][7] = pos.x * pos.x;
    T_tilde[i][8] = pos.y * pos.y;
    T_tilde[i][9] = pos.z * pos.z;

    vec4 noisy = texelFetch(tex_indirect, local_coord, 0);
    T_tilde[i][10] = noisy.y;
  }

  for (int w = 0; w < W; w++) {
    for (int m = FEATURES_NOT_SCALED; m < M; m++) {
      min_values[m] = min(min_values[m], T_tilde[w][m]);
      max_values[m] = max(max_values[m], T_tilde[w][m]);
      mag_values[m] += T_tilde[w][m] * T_tilde[w][m];
    }
  }

  for (int m = FEATURES_NOT_SCALED; m < M; m++) {
    mag_values[m] = sqrt(mag_values[m]);
  }

  for (int w = 0; w < W; w++) {
    for (int m = 1; m < M; m++) {
      // if (max_values[m] - min_values[m] > 1.0) {
      //   T_tilde[w][m] =
      //       (T_tilde[w][m] - min_values[m]) / (max_values[m] -
      //       min_values[m]);
      // } else {
      //   T_tilde[w][m] = (T_tilde[w][m] - min_values[m]);
      // }
      if (m >= FEATURES_NOT_SCALED && mag_values[m] != 0.0) {
        T_tilde[w][m] /= mag_values[m];
      }

      T_tilde[w][m] += add_random(w, int(m + uniforms.current_frame), m);
    }
  }

  // Compute QR factorization for T_tilde
  float R_red_tilde[W][M + 1];
  householder_qr(T_tilde, R_red_tilde);

  for (int w = 0; w < W; w++) {
    int x = (w % S_W) * (BLOCK_SIZE / S_W);
    int y = (w / S_W) * (BLOCK_SIZE / S_W);
    ivec2 local_coord = coord + ivec2(x, y);
    T_tilde[w][M] = texelFetch(tex_indirect, local_coord, 0).y;
  }

  float R_green_tilde[W][M + 1];
  householder_qr(T_tilde, R_green_tilde);

  for (int w = 0; w < W; w++) {
    int x = (w % S_W) * (BLOCK_SIZE / S_W);
    int y = (w / S_W) * (BLOCK_SIZE / S_W);
    ivec2 local_coord = coord + ivec2(x, y);
    T_tilde[w][M] = texelFetch(tex_indirect, local_coord, 0).z;
  }

  float R_blue_tilde[W][M + 1];
  householder_qr(T_tilde, R_blue_tilde);

  // Extract R and a r for each channel
  float R[M][M];
  float r_red[M];
  float r_green[M];
  float r_blue[M];

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < M; j++) {
      R[i][j] = R_red_tilde[i][j];
    }
  }

  for (int j = 0; j < M; j++) {
    r_red[j] = R_red_tilde[j][M];
    r_green[j] = R_green_tilde[j][M];
    r_blue[j] = R_blue_tilde[j][M];
  }

  // Resolve Ra=r
  float alpha_red[M];
  resolve(R, r_red, alpha_red);
  float alpha_green[M];
  resolve(R, r_green, alpha_green);
  float alpha_blue[M];
  resolve(R, r_blue, alpha_blue);

  if (gl_GlobalInvocationID.x == 20 && gl_GlobalInvocationID.y == 10) {
    for (int w = 0; w < W; w++) {
      for (int m = 0; m < (M + 1); m++) {
        debug_tilde[w][m] = T_tilde[w][m];
      }
    }

    for (int w = 0; w < W; w++) {
      for (int m = 0; m < (M + 1); m++) {
        debug_h[w][m] = R_red_tilde[w][m];
      }
    }

    debug_alpha_red[0] = coord.x;
    debug_alpha_red[1] = coord.y;
  }

  // Output results
  for (int offx = 0; offx < BLOCK_SIZE; offx++) {
    for (int offy = 0; offy < BLOCK_SIZE; offy++) {
      // Fetch feature for this pixel
      vec4 pos = texelFetch(tex_pos, coord + ivec2(offx, offy), 0);
      vec4 norm = texelFetch(tex_normal, coord + ivec2(offx, offy), 0);
      vec4 depth = texelFetch(tex_depth, coord + ivec2(offx, offy), 0);
      vec4 alb = texelFetch(tex_albedo, coord + ivec2(offx, offy), 0);
      if (depth.x == 1) {
        continue;
      }
      float features[M];
      features[0] = 1.0;
      features[1] = norm.x;
      features[2] = norm.y;
      features[3] = norm.z;
      features[4] = pos.x;
      features[5] = pos.y;
      features[6] = pos.z;
      features[7] = pos.x * pos.x;
      features[8] = pos.y * pos.y;
      features[9] = pos.z * pos.z;
      for (int m = FEATURES_NOT_SCALED; m < M; m++) {
        // features[m] =
        //     (features[m] - min_values[m]) / (max_values[m] - min_values[m]);
        if (mag_values[m] != 0.0) {
          features[m] /= mag_values[m];
        }
      }
      float red = dot_m(alpha_red, features);
      if (red < 0.0) {
        red = 0.0;
      }
      float green = dot_m(alpha_green, features);
      if (green < 0.0) {
        green = 0.0;
      }
      float blue = dot_m(alpha_blue, features);
      if (blue < 0.0) {
        blue = 0.0;
      }
      // imageStore(tex_out, coord + ivec2(offx, offy),
      //            vec4(alpha_red[0], alpha_red[1], alpha_red[2],
      //            alpha_red[3]));
      imageStore(tex_out, coord + ivec2(offx, offy),
                 vec4(red, green, blue, 1.0));
    }
  }
}
