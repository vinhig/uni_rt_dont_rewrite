#include "accumulator.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace UniRt {

GLuint CompileShader(const char *path, const char *source, GLenum shaderType);

namespace Denoiser {
AccumulatorDenoiser::AccumulatorDenoiser() {
  {
    std::ifstream features_ta_file("../shaders/accumulator.comp.glsl");
    std::ostringstream ta_ss;
    ta_ss << features_ta_file.rdbuf();
    std::string features_ta_source = ta_ss.str();

    GLuint compute =
        CompileShader("../shader/bmfr_regression.comp.glsl",
                      features_ta_source.c_str(), GL_COMPUTE_SHADER);

    accum_program = glCreateProgram();
    glAttachShader(accum_program, compute);
    glLinkProgram(accum_program);

    glDeleteShader(compute);
  }

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

  glGenTextures(1, &accum_texture);
  glBindTexture(GL_TEXTURE_2D, accum_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
               nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

GLuint AccumulatorDenoiser::Denoise(BunchOfTexture &textures,
                                    int current_frame) {
  glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1,
                   "ACCUMULATOR REGRESSION");

  glUseProgram(accum_program);
  glBindBufferBase(GL_UNIFORM_BUFFER, 1, textures.reprojection_buffer);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, textures.noisy_texture[current_frame % 2]);

  glBindImageTexture(1, accum_texture, 0, 0, 0, GL_READ_WRITE, GL_RGBA32F);

  glDispatchCompute((1280 + 16) / 16, (720 + 16) / 16, 1);
  glPopDebugGroup();

  return accum_texture;
}

void AccumulatorDenoiser::ReprojectSeed(BunchOfTexture &textures, int current_frame) {
  glUseProgram(reprojection_program);

  glBindImageTexture(0, textures.rng_seed_texture[current_frame % 2], 0, 0, 0,
                     GL_READ_WRITE, GL_RGBA32UI);

  glBindBufferBase(GL_UNIFORM_BUFFER, 0, textures.reprojection_buffer);

  glDispatchCompute((1280 + 16) / 16, (720 + 16) / 16, 1);
}

} // namespace Denoiser
} // namespace UniRt