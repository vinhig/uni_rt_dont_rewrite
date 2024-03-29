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

  {
    std::ifstream features_ta_file("../shaders/bmfr_reprojection.comp.glsl");
    std::ostringstream ta_ss;
    ta_ss << features_ta_file.rdbuf();
    std::string features_ta_source = ta_ss.str();

    GLuint ta_compute =
        CompileShader("../shader/bmfr_reprojection.comp.glsl",
                      features_ta_source.c_str(), GL_COMPUTE_SHADER);

    bmfr_reprojection_program = glCreateProgram();
    glAttachShader(bmfr_reprojection_program, ta_compute);
    glLinkProgram(bmfr_reprojection_program);

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

  printf("hello from bmfr denoiser\n");
}

GLuint BmfrDenoiser::Denoise(BunchOfTexture &textures, int current_frame) {
  glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "BMFR REGRESSION");
  glUseProgram(bmfr_program);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, textures.position_texture[current_frame % 2]);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, textures.normal_texture[current_frame % 2]);

  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, textures.depth_texture[current_frame % 2]);

  glActiveTexture(GL_TEXTURE3);
  glBindTexture(GL_TEXTURE_2D, textures.noisy_texture[current_frame % 2]);

  glActiveTexture(GL_TEXTURE8);
  glBindTexture(GL_TEXTURE_2D, textures.albedo_texture[current_frame % 2]);

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

  glBindBufferBase(GL_UNIFORM_BUFFER, 0, textures.reprojection_buffer);

  glDispatchCompute(w * h, 1, 1);

  glMemoryBarrier(GL_ALL_BARRIER_BITS);

  glPopDebugGroup();

  return denoised_texture[current_frame % 2];
}

BmfrDenoiser::~BmfrDenoiser() {
  glDeleteProgram(bmfr_program);
  glDeleteTextures(1, &tmp_fitting_texture);
  glDeleteTextures(1, &out_fitting_texture);
  glDeleteTextures(2, denoised_texture);
};

void BmfrDenoiser::ReprojectSeed(BunchOfTexture &textures, int current_frame) {
  glUseProgram(bmfr_reprojection_program);

  glBindImageTexture(0, textures.rng_seed_texture[current_frame % 2], 0, 0, 0,
                     GL_READ_WRITE, GL_RGBA32UI);

  glBindBufferBase(GL_UNIFORM_BUFFER, 0, textures.reprojection_buffer);

  glDispatchCompute((1280 + 15) / 16, (720 + 15) / 16, 1);
}
} // namespace Denoiser
} // namespace UniRt
