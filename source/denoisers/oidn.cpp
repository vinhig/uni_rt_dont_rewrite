#include <stdio.h>

#include <OpenImageDenoise/oidn.hpp>

#include "oidn.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace UniRt {

GLuint CompileShader(const char *path, const char *source, GLenum shaderType);

namespace Denoiser {
OidnDenoiser::OidnDenoiser() {
  printf("Loading Oidn denoiser...\n");
  {
    std::ifstream features_ta_file("../shaders/bmfr_reprojection.comp.glsl");
    std::ostringstream ta_ss;
    ta_ss << features_ta_file.rdbuf();
    std::string features_ta_source = ta_ss.str();

    GLuint ta_compute =
        CompileShader("../shader/bmfr_reprojection.comp.glsl",
                      features_ta_source.c_str(), GL_COMPUTE_SHADER);

    seed_program = glCreateProgram();
    glAttachShader(seed_program, ta_compute);
    glLinkProgram(seed_program);

    glDeleteShader(ta_compute);
  }

  device = oidn::newDevice();
  device.commit();

  filter = device.newFilter("RT");

  color_buffer = new float[1280 * 720 * 4];
  normal_buffer = new float[1280 * 720 * 4];
  albedo_buffer = new float[1280 * 720 * 4];
  output_buffer = new float[1280 * 720 * 4];

  glGenTextures(1, &denoised_texture);

  glBindTexture(GL_TEXTURE_2D, denoised_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
               nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  printf("Finished loading Oidn denoiser...\n");
}

void OidnDenoiser::ReprojectSeed(BunchOfTexture &textures, int current_frame) {
  glUseProgram(seed_program);

  glBindImageTexture(0, textures.rng_seed_texture[current_frame % 2], 0, 0, 0,
                     GL_READ_WRITE, GL_RGBA32UI);

  glBindBufferBase(GL_UNIFORM_BUFFER, 0, textures.reprojection_buffer);

  glDispatchCompute((1280 + 15) / 16, (720 + 15) / 16, 1);
}

GLuint OidnDenoiser::Denoise(BunchOfTexture &textures, int current_frame) {
  printf("Beginning denoising with oidn...\n");
  // Fetch data from GPU
  glBindTexture(GL_TEXTURE_2D, textures.noisy_texture[current_frame % 2]);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, color_buffer);

  glBindTexture(GL_TEXTURE_2D, textures.normal_texture[current_frame % 2]);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, normal_buffer);

  glBindTexture(GL_TEXTURE_2D, textures.albedo_texture[current_frame % 2]);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, albedo_buffer);

  glMemoryBarrier(GL_ALL_BARRIER_BITS);

  filter.setImage("color", color_buffer, oidn::Format::Float3, 1280, 720);
  // filter.setImage("normal", normal_buffer, oidn::Format::Float3, 1280, 720);
  // filter.setImage("albedo", albedo_buffer, oidn::Format::Float3, 1280, 720);
  filter.setImage("output", output_buffer, oidn::Format::Float3, 1280, 720);
  filter.set("hdr", true);

  filter.commit();
  filter.execute();

  const char *error_message;
  if (device.getError(error_message) != oidn::Error::None) {
    printf("Error with oidn: %s\n", error_message);

  }

  glBindTexture(GL_TEXTURE_2D, denoised_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGB, GL_FLOAT,
               output_buffer);

  printf("Finished denoising with oidn...\n");

  return denoised_texture;
}
} // namespace Denoiser
} // namespace UniRt