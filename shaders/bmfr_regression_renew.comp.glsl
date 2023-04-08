#version 430

#define BLOCK_SIZE 32
#define W 16
#define M 4
// 1, POS_X², POS_Y², POS_Z²
#define NOISE_AMOUNT 0.1

#define OFFSETS_COUNT 16
const vec2 RELATIVE_OFFSETS[OFFSETS_COUNT] = {
    /*
      vec2(0.5004785746285491, 0.7249900791844404),
      vec2(0.6616787922684506, 0.8431932401454286),
      vec2(0.013792616530536095, 0.07606840902247503),
      vec2(0.5506476459339343, 0.5526037993849915),
      vec2(0.19107659212621841, 0.6354946478787757),
      vec2(0.08635995805355967, 0.2472202396255624),
      vec2(0.1252250440433068, 0.08609885530246719),
      vec2(0.5904742704843103, 0.030490538789994526),
      vec2(0.10092023462217092, 0.28999319035930815),
      vec2(0.5111282665879786, 0.5210547623000872),
      vec2(0.41340214843009226, 0.4053946747614482),
      vec2(0.7918845643409962, 0.5976228115068767),
      vec2(0.581450087162451, 0.4589538001737091),
      vec2(0.842207304152363, 0.2930246707112014),
      vec2(0.8484321745810262, 0.6802420254928683),
      vec2(0.05634620713888261, 0.33998908042767395),
      vec2(0.4127325637806537, 0.03475081029893923),
      vec2(0.7351358812117753, 0.4609277171243483),
      vec2(0.7530362313643808, 0.07159967465392558),
      vec2(0.2537496924765742, 0.4081893890964827),
      vec2(0.8242599497999632, 0.3479496401290726),
      vec2(0.02472941535834594, 0.08531173028951677),
      vec2(0.14550073150566112, 0.00019923791476850194),
      vec2(0.8701282290178975, 0.10619976176689505),
      vec2(0.8301964700168487, 0.891685097229575),
      vec2(0.0965915286907838, 0.14401801210186915),
      vec2(0.739085379430238, 0.40751300048167005),
      vec2(0.8189104667888195, 0.07553790493802559),
      vec2(0.0819085928456772, 0.47462590600346855),
      vec2(0.27586799506009396, 0.23212979598942174),
      vec2(0.502863992002033, 0.07109038503647092),
      vec2(0.8964918160764741, 0.8827850755057025),
      */
    vec2(-30, -30) / 64.0 + 1.0, vec2(-12, -22) / 64.0 + 1.0,
    vec2(-24, -2) / 64.0 + 1.0,  vec2(-8, -16) / 64.0 + 1.0,
    vec2(-26, -24) / 64.0 + 1.0, vec2(-14, -4) / 64.0 + 1.0,
    vec2(-4, -28) / 64.0 + 1.0,  vec2(-26, -16) / 64.0 + 1.0,
    vec2(-4, -2) / 64.0 + 1.0,   vec2(-24, -32) / 64.0 + 1.0,
    vec2(-10, -10) / 64.0 + 1.0, vec2(-18, -18) / 64.0 + 1.0,
    vec2(-12, -30) / 64.0 + 1.0, vec2(-32, -4) / 64.0 + 1.0,
    vec2(-2, -20) / 64.0 + 1.0,  vec2(-22, -12) / 64.0 + 1.0,
};

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) uniform sampler2D tex_pos;
layout(binding = 1) uniform sampler2D tex_normal;
layout(binding = 2) uniform sampler2D tex_depth;
layout(binding = 3) uniform sampler2D tex_indirect;

layout(binding = 4, rgba32f) uniform writeonly image2D tex_out;

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

// layout(std430, binding = 5) buffer debug_1 { float debug_tilde[W][M + 3]; };

// layout(std430, binding = 6) buffer debug_2 { float debug_h[W][M + 3]; };

// layout(std430, binding = 7) buffer debug_3 { float debug_alpha[W]; };

// layout(std430, binding = 8) buffer debug_4 { float debug_other[W]; };

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

void householder_step(float T_tilde[W][M + 3], out float H[W][W], int k) {
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

void mul_mat_H(float H[W][W], float A[W][M + 3], out float R[W][M + 3]) {
  for (int i = 0; i < W; i++) {
    for (int j = 0; j < M + 3; j++) {
      for (int l = 0; l < W; l++) {
        R[i][j] += H[i][l] * A[l][j];
      }
    }
  }
}

void householder_qr(float T_tilde[W][M + 3], out float R[W][M + 3]) {
  float H0[W][W];
  float A0[W][M + 3];

  householder_step(T_tilde, H0, 0);
  mul_mat_H(H0, T_tilde, A0);

  float H1[W][W];
  float A1[W][M + 3];

  householder_step(A0, H1, 1);
  mul_mat_H(H1, A0, A1);

  float H2[W][W];
  float A2[W][M + 3];

  householder_step(A1, H2, 2);
  mul_mat_H(H2, A1, A2);

  float H3[W][W];
  float A3[W][M + 3];

  householder_step(A2, H3, 2);
  mul_mat_H(H3, A2, A3);

  for (int i = 0; i < W; i++) {
    for (int j = 0; j < M + 3; j++) {
      R[j][i] = A3[j][i];
    }
  }

  return;

  float H4[W][W];
  float A4[W][M + 3];

  householder_step(A3, H4, 2);
  mul_mat_H(H4, A3, A4);

  float H5[W][W];
  float A5[W][M + 3];

  householder_step(A4, H5, 2);
  mul_mat_H(H5, A4, A5);

  float H6[W][W];
  float A6[W][M + 3];

  householder_step(A5, H6, 2);
  mul_mat_H(H6, A5, A6);

  float H7[W][W];
  float A7[W][M + 3];

  householder_step(A6, H7, 2);
  mul_mat_H(H7, A6, A7);
}

void resolve(float R[M][M], float r_c[M], out float a[M]) {
  for (int i = 0; i < M; i++) {
    a[i] = 0.0;
  }

  a[M - 1] = r_c[M - 1] / R[M - 1][M - 1];

  for (int i = M - 2; i >= 0; i--) {
    float sum = 0.0;
    for (int j = i + 1; j < M; j++) {
      sum += R[i][j] * a[j];
    }
    a[i] = (r_c[i] - sum) / R[i][i];
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

  for (int offx = 0; offx < BLOCK_SIZE; offx++) {
    for (int offy = 0; offy < BLOCK_SIZE; offy++) {
      imageStore(tex_out, coord + ivec2(offx, offy),
                 vec4(coord / uniforms.target_dim, 1.0, 1.0));
    }
  }

  // coord *= ivec2(BLOCK_SIZE, BLOCK_SIZE);

  // Build T_tilde
  float T_tilde[W][M + 3];

  for (int i = 0; i < W; i++) {
    T_tilde[i][0] = 1.0;
  }

  for (int i = 0; i < W; i++) {
    ivec2 local_coord =
        coord +
        ivec2(RELATIVE_OFFSETS[(uniforms.current_frame + i) % OFFSETS_COUNT] *
              vec2(BLOCK_SIZE));

    vec4 pos = texelFetch(tex_pos, local_coord, 0);
    T_tilde[i][1] = pos.x * pos.x;
    T_tilde[i][2] = pos.y * pos.y;
    T_tilde[i][3] = pos.z * pos.z;

    T_tilde[i][1] += add_random(local_coord.x, local_coord.y, i + 1);
    T_tilde[i][2] += add_random(local_coord.x, local_coord.y, i + 2);
    T_tilde[i][3] += add_random(local_coord.x, local_coord.y, i + 3);

    // T_tilde[i][1] = float(i);

    // vec4 norm = texelFetch(tex_normal, local_coord, 0);
    // T_tilde[i][4] = norm.x + add_random(local_coord.x, local_coord.y, i + 4);
    // T_tilde[i][5] = norm.y + add_random(local_coord.x, local_coord.y, i + 5);
    // T_tilde[i][6] = norm.z + add_random(local_coord.x, local_coord.y, i + 6);

    // T_tilde[i][7] =
    //     pos.x * pos.x + add_random(local_coord.x, local_coord.y, i + 8);
    // T_tilde[i][8] =
    //     pos.y * pos.y + add_random(local_coord.x, local_coord.y, i + 8);
    // T_tilde[i][9] =
    //     pos.z * pos.z + add_random(local_coord.x, local_coord.y, i + 9);

    vec4 noisy = texelFetch(tex_indirect, local_coord, 0);
    T_tilde[i][4] = noisy.x;
    T_tilde[i][5] = noisy.y;
    T_tilde[i][6] = noisy.z;
  }

  // Compute QR factorization for T_tilde

  // Debug output for a single matrix
  // if (coord.x == 640 && coord.y == 360) {
  //   for (int w = 0; w < W; w++) {
  //     for (int m = 0; m < M + 3; m++) {
  //       debug_tilde[w][m] = T_tilde[w][m];
  //     }
  //   }
  // }

  float R_tilde[W][M + 3];
  householder_qr(T_tilde, R_tilde);

  // if (gl_GlobalInvocationID.x == 640 && gl_GlobalInvocationID.y == 360) {
  //   for (int i = 0; i < W; i++) {
  //     for (int j = 0; j < M + 3; j++) {
  //       debug_h[i][j] = R_tilde[i][j];
  //     }
  //   }
  // }

  // Extract R and a r for each channel
  float R[M][M];
  float r_red[M];
  float r_green[M];
  float r_blue[M];

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < M; j++) {
      R[i][j] = R_tilde[i][j];
    }
  }

  for (int j = 0; j < M; j++) {
    r_red[j] = R_tilde[j][4];
  }

  for (int j = 0; j < M; j++) {
    r_green[j] = R_tilde[j][5];
  }

  for (int j = 0; j < M; j++) {
    r_blue[j] = R_tilde[j][6];
  }

  // if (gl_GlobalInvocationID.x == 640 && gl_GlobalInvocationID.y == 360) {
  //   for (int i = 0; i < M; i++) {
  //     debug_alpha[i] = r_blue[i];
  //   }
  // }

  // Resolve Ra=r
  float alpha_red[M];
  resolve(R, r_red, alpha_red);
  float alpha_green[M];
  resolve(R, r_green, alpha_green);
  float alpha_blue[M];
  resolve(R, r_blue, alpha_blue);

  // Output results
  for (int offx = 0; offx < BLOCK_SIZE; offx++) {
    for (int offy = 0; offy < BLOCK_SIZE; offy++) {
      // Fetch feature for this pixel
      vec4 pos = texelFetch(tex_pos, coord + ivec2(offx, offy), 0);
      vec4 norm = texelFetch(tex_normal, coord + ivec2(offx, offy), 0);
      float features[M];
      features[0] = 1.0;
      features[1] = pos.x * pos.x;
      features[2] = pos.y * pos.y;
      features[3] = pos.z * pos.z;
      // features[4] = norm.x;
      // features[5] = norm.y;
      // features[6] = norm.z;
      // features[7] = pos.x * pos.x;
      // features[8] = pos.y * pos.y;
      // features[9] = pos.z * pos.z;
      float red = dot_m(alpha_red, features);
      float green = dot_m(alpha_green, features);
      float blue = dot_m(alpha_blue, features);
      // imageStore(tex_out, coord + ivec2(offx, offy),
      //            vec4(abs(alpha_red[0]), abs(alpha_red[1]), abs(alpha_red[2]),
      //                 abs(alpha_red[3])));
      imageStore(tex_out, coord + ivec2(offx, offy), vec4(red, green, blue, 1.0));
    }
  }
}
