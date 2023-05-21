#version 430

#define BLOCK_SIZE 64
#define W 36
#define S_W 5
#define M 10
// 1, NORM_X, NORM_Y, NORM_Z, POS_X, POS_Y, POS_Z, POS_X², POS_Y², POS_Z²
// 2
#define NOISE_AMOUNT 0.01

#define FEATURES_NOT_SCALED 4

#define BLOCK_OFFSETS_COUNT 16
const ivec2 BLOCK_OFFSETS[BLOCK_OFFSETS_COUNT] = {
    ivec2(-30, -30), ivec2(-12, -22), ivec2(-24, -2),  ivec2(-8, -16),
    ivec2(-26, -24), ivec2(-14, -4),  ivec2(-4, -28),  ivec2(-26, -16),
    ivec2(-4, -2),   ivec2(-24, -32), ivec2(-10, -10), ivec2(-18, -18),
    ivec2(-12, -30), ivec2(-32, -4),  ivec2(-2, -20),  ivec2(-22, -12),
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
  float gradient_cap;
  uint current_frame;
}
uniforms;

int my_W = W;
int my_M = M;

layout(std430, binding = 1) buffer red_tilde {
  //
  float R_red_tilde[][W][M + 1];
};

layout(std430, binding = 2) buffer green_tilde {
  //
  float R_green_tilde[][W][M + 1];
};

layout(std430, binding = 3) buffer blue_tilde {
  //
  float R_blue_tilde[][W][M + 1];
};

layout(std430, binding = 4) buffer tilde {
  //
  float T_tilde[][W][M + 1];
};

layout(std430, binding = 5) buffer r {
  //
  float R[][W][M + 1];
};

layout(std430, binding = 6) buffer tmp_in_tilde {
  //
  float T_tmp_in_tilde[][W][M + 1];
};

layout(std430, binding = 7) buffer tmp_out_tilde {
  //
  float T_tmp_out_tilde[][W][M + 1];
};

layout(std430, binding = 8) buffer tmp_h {
  //
  float H_tmp[][W][W];
};

layout(std430, binding = 9) buffer tmp_alpha {
  //
  float alpha[][W];
};

layout(std430, binding = 10) buffer tmp_v {
  //
  float v[][W];
};

shared float alpha_red[M];
shared float alpha_green[M];
shared float alpha_blue[M];

shared float max_values[M];
shared float min_values[M];
shared float mag_values[M];

float luminance(vec3 color) { return dot(color, vec3(0.2126, 0.7152, 0.0722)); }

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

void householder_step(int index_buff, int k) {
  // Compute the reflection of the k column

  for (int i = 0; i < my_W - k; i++) {
    alpha[index_buff][i] = T_tmp_in_tilde[index_buff][i + k][k];
  }

  float sign_a0 = sign(alpha[index_buff][0]);
  float norm_a = 0.0;
  for (int i = 0; i < my_W - k; i++) {
    norm_a += alpha[index_buff][i] * alpha[index_buff][i];
  }
  norm_a = sqrt(norm_a);

  for (int i = 0; i < my_W; i++) {
    v[index_buff][i] = 0.0;
  }

  v[index_buff][0] = sign_a0 * norm_a + alpha[index_buff][0];
  for (int i = 1; i < my_W; i++) {
    v[index_buff][i] = v[index_buff][i] + alpha[index_buff][i];
  }

  // Compute the corresponding householder matrix
  float uut;
  for (int i = 0; i < my_W - k; i++) {
    uut += v[index_buff][i] * v[index_buff][i];
  }

  for (int x = 0; x < my_W; x++) {
    for (int y = 0; y < my_W; y++) {
      if (x < k || y < k) {
        if (x == y) {
          H_tmp[index_buff][x][y] = 1.0;
        } else {
          H_tmp[index_buff][x][y] = 0.0;
        }
      } else {
        if (x == y) {
          H_tmp[index_buff][x][y] =
              1.0 - v[index_buff][x - k] * v[index_buff][y - k] / uut * 2;
        } else {
          H_tmp[index_buff][x][y] =
              0.0 - v[index_buff][x - k] * v[index_buff][y - k] / uut * 2;
        }
      }
    }
  }
}

void mul_mat_H(int index_buff) {
  for (int i = 0; i < my_W; i++) {
    for (int j = 0; j < my_M + 1; j++) {
      T_tmp_out_tilde[index_buff][i][j] = 0.0;
    }
  }
  for (int i = 0; i < my_W; i++) {
    for (int j = 0; j < my_M + 1; j++) {
      for (int l = 0; l < my_W; l++) {
        T_tmp_out_tilde[index_buff][i][j] +=
            H_tmp[index_buff][i][l] * T_tmp_in_tilde[index_buff][l][j];
      }
    }
  }
}

void householder_qr(int index_buff, int channel) {
  T_tmp_in_tilde[index_buff] = T_tilde[index_buff];

  householder_step(index_buff, 0);
  mul_mat_H(index_buff);

  T_tmp_in_tilde[index_buff] = T_tmp_out_tilde[index_buff];

  householder_step(index_buff, 1);
  mul_mat_H(index_buff);

  T_tmp_in_tilde[index_buff] = T_tmp_out_tilde[index_buff];

  householder_step(index_buff, 2);
  mul_mat_H(index_buff);

  T_tmp_in_tilde[index_buff] = T_tmp_out_tilde[index_buff];

  householder_step(index_buff, 3);
  mul_mat_H(index_buff);

  T_tmp_in_tilde[index_buff] = T_tmp_out_tilde[index_buff];

  householder_step(index_buff, 4);
  mul_mat_H(index_buff);

  T_tmp_in_tilde[index_buff] = T_tmp_out_tilde[index_buff];

  householder_step(index_buff, 5);
  mul_mat_H(index_buff);

  T_tmp_in_tilde[index_buff] = T_tmp_out_tilde[index_buff];

  householder_step(index_buff, 6);
  mul_mat_H(index_buff);

  T_tmp_in_tilde[index_buff] = T_tmp_out_tilde[index_buff];

  householder_step(index_buff, 7);
  mul_mat_H(index_buff);

  T_tmp_in_tilde[index_buff] = T_tmp_out_tilde[index_buff];

  householder_step(index_buff, 8);
  mul_mat_H(index_buff);

  T_tmp_in_tilde[index_buff] = T_tmp_out_tilde[index_buff];

  householder_step(index_buff, 9);
  mul_mat_H(index_buff);

  T_tmp_in_tilde[index_buff] = T_tmp_out_tilde[index_buff];

  householder_step(index_buff, 10);
  mul_mat_H(index_buff);

  if (channel == 0) {
    for (int w = 0; w < my_W; w++) {
      for (int m = 0; m < my_M + 1; m++) {
        R_red_tilde[index_buff][w][m] = T_tmp_out_tilde[index_buff][w][m];
      }
    }
  } else if (channel == 1) {
    for (int w = 0; w < my_W; w++) {
      for (int m = 0; m < my_M + 1; m++) {
        R_green_tilde[index_buff][w][m] = T_tmp_out_tilde[index_buff][w][m];
      }
    }
  } else if (channel == 2) {
    for (int w = 0; w < my_W; w++) {
      for (int m = 0; m < my_M + 1; m++) {
        R_blue_tilde[index_buff][w][m] = T_tmp_out_tilde[index_buff][w][m];
      }
    }
  }

  return;
}

void resolve(int index_buff, float r_c[M], out float a[M]) {
  for (int i = 0; i < my_M; i++) {
    a[i] = 0.0;
  }

  if (R[index_buff][M - 1][M - 1] != 0) {
    a[M - 1] = r_c[M - 1] / R[index_buff][M - 1][M - 1];
  } else {
    a[M - 1] = 0.0;
  }

  for (int i = M - 2; i >= 0; i--) {
    float sum = 0.0;
    for (int j = i + 1; j < my_M; j++) {
      sum += R[index_buff][i][j] * a[j];
    }
    if (R[index_buff][i][i] != 0) {
      a[i] = (r_c[i] - sum) / R[index_buff][i][i];
    } else {
      a[i] = 0.0;
    }
  }
}

float dot_m(float a[M], float b[M]) {
  float sum = 0.0;
  for (int i = 0; i < my_M; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

void main() {
  if (my_M != M) {
    return;
  }
  if (my_W != W) {
    return;
  }

  int line_width = 1280 / BLOCK_SIZE;
  int index_buff =
      line_width * int(gl_GlobalInvocationID.x) + int(gl_GlobalInvocationID.y);

  ivec2 coord = ivec2(gl_GlobalInvocationID.xy) * BLOCK_SIZE;

  coord += BLOCK_OFFSETS[uniforms.current_frame % BLOCK_OFFSETS_COUNT] * int((float(BLOCK_SIZE) / 32.0) * 2.0);

  if (coord.x > uniforms.target_dim.x || coord.y > uniforms.target_dim.y) {
    return;
  }

  for (int offx = 0; offx < BLOCK_SIZE; offx++) {
    for (int offy = 0; offy < BLOCK_SIZE; offy++) {
      imageStore(tex_out, coord + ivec2(offx, offy), vec4(0.0));
    }
  }

  // Build T_tilde
  for (int i = 0; i < my_W; i++) {
    T_tilde[index_buff][i][0] = 1.0;
  }

  for (int x = FEATURES_NOT_SCALED; x < my_M; x++) {
    max_values[x] = 0.0;
    min_values[x] = 0.0;
    mag_values[x] = 0.0;
  }

  for (int i = 0; i < my_W; i++) {
    int x = (i % S_W) * (BLOCK_SIZE / S_W);
    int y = (i / S_W) * (BLOCK_SIZE / S_W);
    ivec2 local_coord = coord + ivec2(x, y);

    vec4 norm = texelFetch(tex_normal, local_coord, 0);
    vec4 pos = texelFetch(tex_pos, local_coord, 0);
    vec4 alb = texelFetch(tex_albedo, local_coord, 0);
    T_tilde[index_buff][i][1] = norm.x;
    T_tilde[index_buff][i][2] = norm.y;
    T_tilde[index_buff][i][3] = norm.z;
    T_tilde[index_buff][i][4] = pos.x;
    T_tilde[index_buff][i][5] = pos.y;
    T_tilde[index_buff][i][6] = pos.z;
    T_tilde[index_buff][i][7] = pos.x * pos.x;
    T_tilde[index_buff][i][8] = pos.y * pos.y;
    T_tilde[index_buff][i][9] = pos.z * pos.z;

    vec4 noisy = texelFetch(tex_indirect, local_coord, 0);
    T_tilde[index_buff][i][M] = noisy.y;
  }

  for (int w = 0; w < my_W; w++) {
    for (int m = FEATURES_NOT_SCALED; m < my_M; m++) {
      min_values[m] = min(min_values[m], T_tilde[index_buff][w][m]);
      max_values[m] = max(max_values[m], T_tilde[index_buff][w][m]);
      mag_values[m] += T_tilde[index_buff][w][m] * T_tilde[index_buff][w][m];
    }
  }

  for (int m = FEATURES_NOT_SCALED; m < my_M; m++) {
    mag_values[m] = sqrt(mag_values[m]);
  }

  for (int w = 0; w < my_W; w++) {
    for (int m = 1; m < my_M; m++) {

      if (m >= FEATURES_NOT_SCALED && mag_values[m] != 0.0) {
        T_tilde[index_buff][w][m] /= mag_values[m];
      }

      T_tilde[index_buff][w][m] +=
          add_random(w, int(m + uniforms.current_frame), m);
    }
  }

  // Compute QR factorization for T_tilde
  householder_qr(index_buff, 0);

  for (int w = 0; w < my_W; w++) {
    int x = (w % S_W) * (BLOCK_SIZE / S_W);
    int y = (w / S_W) * (BLOCK_SIZE / S_W);
    ivec2 local_coord = coord + ivec2(x, y);
    T_tilde[index_buff][w][M] = texelFetch(tex_indirect, local_coord, 0).y;
  }

  householder_qr(index_buff, 1);

  for (int w = 0; w < my_W; w++) {
    int x = (w % S_W) * (BLOCK_SIZE / S_W);
    int y = (w / S_W) * (BLOCK_SIZE / S_W);
    ivec2 local_coord = coord + ivec2(x, y);
    T_tilde[index_buff][w][M] = texelFetch(tex_indirect, local_coord, 0).z;
  }

  householder_qr(index_buff, 2);

  // Extract R and a r for each channel
  float r_red[M];
  float r_green[M];
  float r_blue[M];

  for (int i = 0; i < my_M; i++) {
    for (int j = 0; j < my_M; j++) {
      R[index_buff][i][j] = R_red_tilde[index_buff][i][j];
    }
  }

  for (int j = 0; j < my_M; j++) {
    r_red[j] = R_red_tilde[index_buff][j][M];
    r_green[j] = R_green_tilde[index_buff][j][M];
    r_blue[j] = R_blue_tilde[index_buff][j][M];
  }

  // Resolve Ra=r
  resolve(index_buff, r_red, alpha_red);
  resolve(index_buff, r_green, alpha_green);
  resolve(index_buff, r_blue, alpha_blue);

  // Output results
  for (int offx = 0; offx < BLOCK_SIZE; offx++) {
    for (int offy = 0; offy < BLOCK_SIZE; offy++) {
      // Fetch feature for this pixel
      ivec2 local_coord = coord + ivec2(offx, offy);
      if (local_coord.x < 0 || local_coord.y < 0) {
        continue;
      }
      if (local_coord.y > uniforms.target_dim.y ||
          local_coord.x > uniforms.target_dim.x) {
        continue;
      }
      vec4 pos = texelFetch(tex_pos, local_coord, 0);
      vec4 norm = texelFetch(tex_normal, local_coord, 0);
      vec4 depth = texelFetch(tex_depth, local_coord, 0);
      vec4 alb = texelFetch(tex_albedo, local_coord, 0);
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
      for (int m = FEATURES_NOT_SCALED; m < my_M; m++) {
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

      vec3 color = vec3(red, green, blue);
      if (luminance(color) > 2.0) {
        color = normalize(color) * 2.0;
      }

      imageStore(tex_out, coord + ivec2(offx, offy), vec4(color, 1.0));
    }
  }
}
