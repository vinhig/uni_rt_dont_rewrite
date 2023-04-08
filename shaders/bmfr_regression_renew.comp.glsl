#version 430

#define BLOCK_SIZE 16
#define W 8
#define M 4 // 1, POS_X, POS_Y, POS_Z

layout(local_size_x = BLOCK_SIZE, local_size_y = BLOCK_SIZE,
       local_size_z = 1) in;

layout(binding = 0) uniform sampler2D tex_pos;
layout(binding = 1) uniform sampler2D tex_normal;
layout(binding = 2) uniform sampler2D tex_depth;
layout(binding = 3) uniform sampler2D tex_indirect;

layout(binding = 4, rgba32f) uniform writeonly image2D tex_out;

layout(std430, binding = 5) buffer debug_1 { float debug_tilde[W][M + 3]; };

layout(std430, binding = 6) buffer debug_2 { float debug_h[W][M + 3]; };

layout(std430, binding = 7) buffer debug_3 { float debug_alpha[W]; };

layout(std430, binding = 8) buffer debug_4 { float debug_other[W]; };

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
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

  for (int offx = 0; offx < BLOCK_SIZE; offx++) {
    for (int offy = 0; offy < BLOCK_SIZE; offy++) {
      imageStore(tex_out, coord + ivec2(offx, offy), vec4(0.0));
    }
  }

  // coord *= ivec2(BLOCK_SIZE, BLOCK_SIZE);

  // Build T_tilde
  float T_tilde[W][M + 3];

  for (int i = 0; i < W; i++) {
    T_tilde[i][0] = 1.0;
  }

  for (int i = 0; i < W; i++) {
    ivec2 local_coord = coord + ivec2(i, i);

    vec4 pos = texelFetch(tex_pos, local_coord, 0);
    T_tilde[i][1] = pos.x;
    T_tilde[i][2] = pos.y + W - float(i);
    T_tilde[i][3] = pos.z;

    // T_tilde[i][1] = float(i);

    vec4 norm = texelFetch(tex_normal, local_coord, 0);
    // T_tilde[i][4] = norm.x;
    // T_tilde[i][5] = norm.y;
    // T_tilde[i][1] = norm.z;

    vec4 noisy = texelFetch(tex_indirect, local_coord, 0);
    T_tilde[i][4] = noisy.x;
    T_tilde[i][5] = noisy.y;
    T_tilde[i][6] = noisy.z;
  }

  // Compute QR factorization for T_tilde

  // Debug output for a single matrix
  if (coord.x == 640 && coord.y == 360) {
    for (int w = 0; w < W; w++) {
      for (int m = 0; m < M + 3; m++) {
        debug_tilde[w][m] = T_tilde[w][m];
      }
    }
  }

  float R_tilde[W][M + 3];
  householder_qr(T_tilde, R_tilde);

  if (gl_GlobalInvocationID.x == 640 && gl_GlobalInvocationID.y == 360) {
    for (int i = 0; i < W; i++) {
      for (int j = 0; j < M + 3; j++) {
        debug_h[i][j] = R_tilde[i][j];
      }
    }
  }

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
    r_red[j] = R_tilde[j][M + 3 - 3];
  }

  for (int j = 0; j < M; j++) {
    r_green[j] = R_tilde[j][M + 3 - 2];
  }

  for (int j = 0; j < M; j++) {
    r_blue[j] = R_tilde[j][M + 3 - 1];
  }

  if (gl_GlobalInvocationID.x == 640 && gl_GlobalInvocationID.y == 360) {
    for (int i = 0; i < M; i++) {
      debug_alpha[i] = r_blue[i];
    }
  }

  // Resolve Ra=r
  float alpha_red[M];
  resolve(R, r_red, alpha_red);
  float alpha_green[M];
  resolve(R, r_green, alpha_green);
  float alpha_blue[M];
  resolve(R, r_blue, alpha_blue);

  if (gl_GlobalInvocationID.x == 640 && gl_GlobalInvocationID.y == 360) {
    for (int i = 0; i < M; i++) {
      debug_alpha[i] = alpha_red[i];
    }
  }

  // Output results
  for (int offx = 0; offx < BLOCK_SIZE; offx++) {
    for (int offy = 0; offy < BLOCK_SIZE; offy++) {
      // Fetch feature for this pixel
      vec4 pos = texelFetch(tex_pos, coord + ivec2(offx, offy), 0);
      float features[M];
      features[0] = 1.0;
      features[1] = pos.x;
      features[2] = pos.y;
      features[3] = pos.z;
      float red = dot_m(alpha_red, features);
      float green = dot_m(alpha_green, features);
      float blue = dot_m(alpha_blue, features);
      imageStore(tex_out, coord + ivec2(offx, offy),
                 vec4(red, green, blue, 1.0));
    }
  }
}