#include "bmfr.h"

#include <stdio.h>

#include <fstream>
#include <iostream>
#include <sstream>

namespace UniRt {

GLuint CompileShader(const char *path, const char *source, GLenum shaderType);

namespace Denoiser {
BmfrDenoiser::BmfrDenoiser() {
  {
    std::ifstream features_ta_file("../shaders/bmfr_regression.comp.glsl");
    std::ostringstream ta_ss;
    ta_ss << features_ta_file.rdbuf();
    std::string features_ta_source = ta_ss.str();

    GLuint ta_compute =
        CompileShader("../shader/bmfr_regression.comp.glsl",
                      features_ta_source.c_str(), GL_COMPUTE_SHADER);

    bmfr_program = glCreateProgram();
    glAttachShader(bmfr_program, ta_compute);
    glLinkProgram(bmfr_program);

    glDeleteShader(ta_compute);
  }

  int width, height;
  width = (1280 + (BLOCK_EDGE_LENGTH - 1)) / BLOCK_EDGE_LENGTH;
  height = (720 + (BLOCK_EDGE_LENGTH - 1)) / BLOCK_EDGE_LENGTH;
  int w = width + 1;
  int h = height + 1;

  glGenTextures(1, &tmp_fitting_texture);
  glGenTextures(1, &out_fitting_texture);

  glGenTextures(2, denoised_texture);

  glBindTexture(GL_TEXTURE_2D, tmp_fitting_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 1024, w * h * 13, 0, GL_RGBA,
               GL_FLOAT, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glBindTexture(GL_TEXTURE_2D, out_fitting_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 1024, w * h * 13, 0, GL_RGBA,
               GL_FLOAT, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glBindTexture(GL_TEXTURE_2D, denoised_texture[0]);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
               nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glBindTexture(GL_TEXTURE_2D, denoised_texture[1]);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
               nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glGenBuffers(1, &per_frame_buffer);
  glBindBuffer(GL_UNIFORM_BUFFER, per_frame_buffer);
  glBufferData(GL_UNIFORM_BUFFER, sizeof(PerFrameCB), &per_frame,
               GL_DYNAMIC_DRAW);
}

GLuint BmfrDenoiser::Denoise(unsigned current_frame, GLuint noisy_texture,
                             GLuint position_texture, GLuint normal_texture,
                             GLuint depth_texture, GLuint albedo_texture) {
  glUseProgram(bmfr_program);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, position_texture);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, normal_texture);

  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, depth_texture);

  glActiveTexture(GL_TEXTURE3);
  glBindTexture(GL_TEXTURE_2D, noisy_texture);

  glActiveTexture(GL_TEXTURE8);
  glBindTexture(GL_TEXTURE_2D, albedo_texture);

  glBindImageTexture(4, out_fitting_texture, 0, 0, 0, GL_READ_WRITE, GL_R32F);

  glBindImageTexture(5, tmp_fitting_texture, 0, 0, 0, GL_READ_WRITE, GL_R32F);

  glBindImageTexture(6, denoised_texture[current_frame % 2], 0, 0, 0,
                     GL_READ_WRITE, GL_RGBA32F);

  int width, height;
  width = (1280 + (BLOCK_EDGE_LENGTH - 1)) / BLOCK_EDGE_LENGTH;
  height = (720 + (BLOCK_EDGE_LENGTH - 1)) / BLOCK_EDGE_LENGTH;
  int w = width + 1;
  int h = height + 1;
  // w /= 2;

  per_frame.horizontal_blocks_count = w;
  per_frame.frame_number = current_frame;

  glBindBuffer(GL_UNIFORM_BUFFER, per_frame_buffer);
  glBufferData(GL_UNIFORM_BUFFER, sizeof(PerFrameCB), &per_frame,
               GL_DYNAMIC_DRAW);
  glBindBufferBase(GL_UNIFORM_BUFFER, 7, per_frame_buffer);

  glDispatchCompute(w * h, 1, 1);

  glMemoryBarrier(GL_ALL_BARRIER_BITS);

  return denoised_texture[current_frame % 2];
}

BmfrDenoiser::~BmfrDenoiser() {
  glDeleteProgram(bmfr_program);
  glDeleteTextures(1, &tmp_fitting_texture);
  glDeleteTextures(1, &out_fitting_texture);
  glDeleteTextures(2, denoised_texture);
};
} // namespace Denoiser
} // namespace UniRt