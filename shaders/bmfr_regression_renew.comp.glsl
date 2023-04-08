#version 430

#define BLOCK_SIZE 16
#define W 4
#define M 2 // 1, POS_X

layout(local_size_x = BLOCK_SIZE, local_size_y = BLOCK_SIZE,
       local_size_z = 1) in;

layout(binding = 0) uniform sampler2D tex_pos;
layout(binding = 1) uniform sampler2D tex_normal;
layout(binding = 2) uniform sampler2D tex_depth;
layout(binding = 3) uniform sampler2D tex_indirect;

layout(binding = 4, rgba32f) uniform writeonly image2D tex_out;

layout(std430, binding = 5) buffer debug_1 { float debug_tilde[W][M + 1]; };

layout(std430, binding = 6) buffer debug_2 { float debug_h[W][W]; };

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

void householder_step(float t_tilde[W][M + 1], out float H[W][W], int k) {
  // Compute the reflection of the k column
  float alpha[W];

  for (int i = 0; i < W - k; i++) {
    alpha[i] = t_tilde[i + k][k];
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

void mul_mat_H(float H[W][W], float A[W][M + 1], out float R[W][M + 1], int k) {
  for (int i = 0; i < W; i++) {
    for (int j = 0; j < M + 1; j++) {
      for (int l = 0; l < W; l++) {
        R[i][j] += H[i][l] * A[l][j];
      }
    }
  }
}

void householder_qr(float t_tilde[W][M + 1], out float t_v[W][M + 1]) {

  float H0[W][W];
  float A0[W][M + 1];

  householder_step(t_tilde, H0, 0);
  mul_mat_H(H0, t_tilde, A0, 0);

  float H1[W][W];
  float A1[W][M + 1];

  householder_step(A0, H1, 1);
  mul_mat_H(H1, A0, A1, 1);

  float H2[W][W];
  float A2[W][M + 1];

  householder_step(A1, H2, 2);
  mul_mat_H(H2, A1, A2, 2);

  if (gl_GlobalInvocationID.x == 640 && gl_GlobalInvocationID.y == 360) {
    for (int i = 0; i < W; i++) {
      for (int j = 0; j < W; j++) {
        debug_h[i][j] = A2[i][j];
      }
    }
  }

  // if (gl_GlobalInvocationID.x == 640 && gl_GlobalInvocationID.y == 360) {
  //   for (int i = 0; i < W; i++) {
  //     for (int j = 0; j < W; j++) {
  //       debug_h[i][j] = H1[i][j];
  //     }
  //   }
  // }
}

void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

  for (int offx = 0; offx < BLOCK_SIZE; offx++) {
    for (int offy = 0; offy < BLOCK_SIZE; offy++) {
      imageStore(tex_out, coord + ivec2(offx, offy), vec4(0.0));
    }
  }

  // coord *= ivec2(BLOCK_SIZE, BLOCK_SIZE);

  // Build t_tilde
  float t_tilde[W][M + 1];

  for (int i = 0; i < W; i++) {
    t_tilde[i][0] = 1.0;
  }

  for (int i = 0; i < W; i++) {
    ivec2 local_coord = coord + ivec2(i, i);

    vec4 pos = texelFetch(tex_pos, local_coord, 0);
    // t_tilde[i][1] = pos.x;
    t_tilde[i][2] = pos.y + W - float(i);
    // t_tilde[i][3] = pos.z;

    // t_tilde[i][1] = float(i);

    vec4 norm = texelFetch(tex_normal, local_coord, 0);
    // t_tilde[i][4] = norm.x;
    // t_tilde[i][5] = norm.y;
    t_tilde[i][1] = norm.z;

    // vec4 noisy = texelFetch(tex_indirect, local_coord, 0);
    // t_tilde[i][7] = noisy.x;
    // t_tilde[i][8] = noisy.y;
    // t_tilde[i][9] = noisy.z;
  }

  // Compute QR factorization for t_tilde

  // Debug output for a single matrix
  if (coord.x == 640 && coord.y == 360) {
    for (int w = 0; w < W; w++) {
      for (int m = 0; m < M + 1; m++) {
        debug_tilde[w][m] = t_tilde[w][m];
      }
    }
  }

  float t_v[W][M + 1];
  householder_qr(t_tilde, t_v);

  // Output results
  for (int offx = 0; offx < BLOCK_SIZE; offx++) {
    for (int offy = 0; offy < BLOCK_SIZE; offy++) {
      imageStore(tex_out, coord + ivec2(offx, offy),
                 texelFetch(tex_indirect, coord + ivec2(offx, offy), 0));
    }
  }
}