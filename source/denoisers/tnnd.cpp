#include "tnnd.h"

#include <stdio.h>

#include <fstream>
#include <iostream>
#include <sstream>

namespace UniRt {

GLuint CompileShader(const char *path, const char *source, GLenum shaderType);

namespace Denoiser {
const float conv_weights[] = {
    -0.549815, -0.450765, -0.387177, -0.132785, 0.042528,  -0.066847, 0.017933,
    -0.013385, -0.080221, 0.351807,  0.446110,  0.509326,  0.002613,  -0.066085,
    -0.051829, 0.015057,  -0.017953, 0.005892,  -0.048645, 0.016222,  0.046158,
    0.015870,  -0.006542, -0.013092, -0.004355, 0.006474,  -0.004166, 0.054990,
    -0.080566, -0.045732, 0.016365,  -0.008066, -0.014056, -0.004598, 0.009611,
    0.002459,  -0.003227, 0.011505,  -0.022025, -0.549815, -0.450765, -0.387177,
    0.179626,  0.038084,  0.079379,  0.017933,  -0.013385, -0.080221, 0.351807,
    0.446110,  0.509326,  -0.018442, 0.017195,  0.003902,  -0.015241, 0.018173,
    -0.002484, -0.000974, 0.034893,  0.022054,  -0.035360, 0.021644,  0.022285,
    0.007473,  -0.024345, -0.000360, -0.003462, 0.004795,  0.157690,  -0.037640,
    0.022846,  0.022691,  0.006989,  -0.013198, 0.004625,  -0.030368, 0.025269,
    0.013464,  -0.442808, -0.489123, -0.440668, -0.133731, 0.048879,  -0.068266,
    0.007131,  -0.022635, -0.067429, 0.424593,  0.409718,  0.455244,  -0.061488,
    0.011191,  -0.063969, 0.012595,  -0.018203, 0.004040,  -0.051185, 0.025610,
    0.039462,  0.001383,  0.007538,  -0.013685, -0.003113, 0.006395,  -0.004889,
    0.050393,  -0.072457, -0.053422, 0.001689,  0.008648,  -0.015433, -0.003778,
    0.010003,  0.002529,  -0.001266, 0.002949,  -0.016167, -0.442808, -0.489123,
    -0.440668, 0.168415,  0.046460,  0.074702,  0.007131,  -0.022635, -0.067429,
    0.424593,  0.409718,  0.455244,  0.006472,  -0.017537, 0.013287,  -0.013414,
    0.016344,  -0.002930, -0.007532, 0.046415,  0.029496,  -0.002607, -0.017695,
    0.027890,  0.003682,  -0.021993, 0.000650,  -0.014597, 0.021102,  0.171368,
    -0.002788, -0.020131, 0.030818,  0.006392,  -0.012124, 0.006755,  -0.030990,
    0.028773,  0.004026,  -0.367037, -0.445783, -0.546304, -0.130397, 0.055508,
    -0.067309, -0.002318, -0.029222, -0.057093, 0.476640,  0.427351,  0.372271,
    -0.050945, -0.060075, -0.001959, 0.011691,  -0.018826, 0.002107,  -0.050792,
    0.028041,  0.037512,  -0.003243, 0.002190,  -0.003254, -0.001990, 0.006903,
    -0.005514, 0.047447,  -0.066568, -0.060609, -0.002014, 0.002066,  -0.004037,
    -0.003632, 0.009958,  0.002450,  -0.001786, -0.000410, -0.008471, -0.367037,
    -0.445783, -0.546304, 0.154776,  0.054423,  0.067855,  -0.002318, -0.029222,
    -0.057093, 0.476640,  0.427351,  0.372271,  0.006389,  -0.003979, -0.000672,
    -0.012956, 0.014691,  -0.003675, -0.013892, 0.053678,  0.048653,  0.002859,
    0.000250,  0.004021,  0.002389,  -0.020841, 0.000864,  -0.025808, 0.037054,
    0.197570,  0.002185,  -0.000309, 0.006139,  0.006894,  -0.010333, 0.009654,
    -0.029927, 0.025684,  -0.011881, 0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    0.000000,  0.000000,
};
const float nn_weights[] = {
    -0.002660, -0.002763, -0.003612, 0.000000, 0.000000,
    0.000000,  0.000000,  0.000000,  0.000000,
};

TnndDenoiser::TnndDenoiser() {
  {
    std::ifstream features_ta_file("../shaders/bmfr_reprojection.comp.glsl");
    std::ostringstream ta_ss;
    ta_ss << features_ta_file.rdbuf();
    std::string features_ta_source = ta_ss.str();

    GLuint ta_compute =
        CompileShader("../shader/bmfr_reprojection.comp.glsl",
                      features_ta_source.c_str(), GL_COMPUTE_SHADER);

    reprojection_program = glCreateProgram();
    glAttachShader(reprojection_program, ta_compute);
    glLinkProgram(reprojection_program);

    glDeleteShader(ta_compute);
  }
  {
    std::ifstream features_ta_file("../../../Projects/tnnd/shaders/"
                                   "tnnd_growing_cellular_growing.comp.glsl");
    std::ostringstream ta_ss;
    ta_ss << features_ta_file.rdbuf();
    std::string features_ta_source = ta_ss.str();

    GLuint ta_compute =
        CompileShader("../../../Projects/tnnd/shaders/"
                      "tnnd_growing_cellular_growing.comp.glsl",
                      features_ta_source.c_str(), GL_COMPUTE_SHADER);

    tnnd_growing_program = glCreateProgram();
    glAttachShader(tnnd_growing_program, ta_compute);
    glLinkProgram(tnnd_growing_program);

    glDeleteShader(ta_compute);
  }

  {
    std::ifstream features_ta_file(
        "../../../Projects/tnnd/shaders/"
        "tnnd_growing_cellular_deep_image.comp.glsl");
    std::ostringstream ta_ss;
    ta_ss << features_ta_file.rdbuf();
    std::string features_ta_source = ta_ss.str();

    GLuint ta_compute =
        CompileShader("../../../Projects/tnnd/shaders/"
                      "tnnd_growing_cellular_deep_image.comp.glsl",
                      features_ta_source.c_str(), GL_COMPUTE_SHADER);

    tnnd_deep_image_program = glCreateProgram();
    glAttachShader(tnnd_deep_image_program, ta_compute);
    glLinkProgram(tnnd_deep_image_program);

    glDeleteShader(ta_compute);
  }

  glGenTextures(1, &ping_texture);
  glGenTextures(1, &pong_texture);
  glGenTextures(1, &denoised_texture);

  glBindTexture(GL_TEXTURE_2D, denoised_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
               nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glBindTexture(GL_TEXTURE_3D, ping_texture);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, 1280, 720, 39, 0, GL_RED, GL_FLOAT,
               nullptr);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glBindTexture(GL_TEXTURE_3D, pong_texture);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, 1280, 720, 39, 0, GL_RED, GL_FLOAT,
               nullptr);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glGenBuffers(1, &conv_weights_buffer);
  glGenBuffers(1, &nn_weights_buffer);

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, conv_weights_buffer);
  glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(conv_weights), &conv_weights[0],
               GL_DYNAMIC_DRAW);

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, nn_weights_buffer);
  glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(nn_weights), &nn_weights[0],
               GL_DYNAMIC_DRAW);

  growing_unroll_3_uniform =
      glGetUniformLocation(tnnd_growing_program, "pls_dont_unroll_3");
  growing_unroll_5_uniform =
      glGetUniformLocation(tnnd_growing_program, "pls_dont_unroll_5");
  growing_channel_in_uniform =
      glGetUniformLocation(tnnd_growing_program, "deep_image_depth");
  growing_depth_uniform =
      glGetUniformLocation(tnnd_growing_program, "channel_in");

  deep_image_unroll_3_uniform =
      glGetUniformLocation(tnnd_deep_image_program, "pls_dont_unroll_3");
  deep_image_unroll_5_uniform =
      glGetUniformLocation(tnnd_deep_image_program, "pls_dont_unroll_5");
  deep_image_channel_in_uniform =
      glGetUniformLocation(tnnd_deep_image_program, "deep_image_depth");
  deep_image_depth_uniform =
      glGetUniformLocation(tnnd_deep_image_program, "channel_in");
}

void TnndDenoiser::ReprojectSeed(BunchOfTexture &textures, int current_frame) {
  glUseProgram(reprojection_program);

  glBindImageTexture(0, textures.rng_seed_texture[current_frame % 2], 0, 0, 0,
                     GL_READ_WRITE, GL_RGBA32UI);

  glBindBufferBase(GL_UNIFORM_BUFFER, 0, textures.reprojection_buffer);

  glDispatchCompute((1280 + 15) / 16, (720 + 15) / 16, 1);

  glMemoryBarrier(GL_ALL_BARRIER_BITS);
}

GLuint TnndDenoiser::Denoise(BunchOfTexture &textures, int current_frame) {
  glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "TNND DENOISING");

  glUseProgram(tnnd_deep_image_program);

  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, conv_weights_buffer);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, nn_weights_buffer);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, textures.normal_texture[current_frame % 2]);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, textures.albedo_texture[current_frame % 2]);

  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, textures.noisy_texture[current_frame % 2]);

  glBindImageTexture(3, ping_texture, 0, true, 0, GL_READ_WRITE, GL_R32F);

  glUniform1i(deep_image_unroll_3_uniform, 3);
  glUniform1i(deep_image_unroll_5_uniform, 5);
  glUniform1i(deep_image_channel_in_uniform, 78);
  glUniform1i(deep_image_depth_uniform, 9);

  glDispatchCompute((1280 + 16) / 16, (720 + 16) / 16, 1);

  glMemoryBarrier(GL_ALL_BARRIER_BITS);

  glUseProgram(tnnd_growing_program);

  glUniform1i(growing_unroll_3_uniform, 3);
  glUniform1i(growing_unroll_5_uniform, 5);
  glUniform1i(growing_channel_in_uniform, 78);
  glUniform1i(growing_depth_uniform, 9);

  glBindImageTexture(5, denoised_texture, 0, false, 0, GL_READ_WRITE, GL_RGBA32F);

  for (unsigned i = 0; i < iteration; i++) {
    if (i % 2 == 0) {
      glBindImageTexture(3, ping_texture, 0, true, 0, GL_READ_WRITE, GL_R32F);
      glBindImageTexture(4, pong_texture, 0, true, 0, GL_READ_WRITE, GL_R32F);
    } else {
      glBindImageTexture(4, ping_texture, 0, true, 0, GL_READ_WRITE, GL_R32F);
      glBindImageTexture(3, pong_texture, 0, true, 0, GL_READ_WRITE, GL_R32F);
    }

    glDispatchCompute((1280 + 16) / 16, (720 + 16) / 16, 1);
  }

  glPopDebugGroup();

  return denoised_texture;
}
} // namespace Denoiser
} // namespace UniRt