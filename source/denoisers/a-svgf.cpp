#include "a-svgf.h"

#include <stdio.h>

#include <fstream>
#include <iostream>
#include <sstream>

namespace UniRt {

GLuint CompileShader(const char *path, const char *source, GLenum shaderType);

namespace Denoiser {
ASvgfDenoiser::ASvgfDenoiser() {
  {
    std::ifstream comp_file("../shaders/asvgf_gradient_img.comp.glsl");
    std::ostringstream comp_ss;
    comp_ss << comp_file.rdbuf();
    std::string comp_source = comp_ss.str();

    GLuint comp_shader = CompileShader("../shader/asvgf_gradient_img.comp.glsl",
                                       comp_source.c_str(), GL_COMPUTE_SHADER);

    gradient_image_program = glCreateProgram();
    glAttachShader(gradient_image_program, comp_shader);
    glLinkProgram(gradient_image_program);

    glDeleteShader(comp_shader);
  }

  glGenTextures(2, gradient_texture);

  for (int i = 0; i < 2; i++) {
    glBindTexture(GL_TEXTURE_2D, gradient_texture[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
                 nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  }

  printf("hello from a-svgf denoiser\n");
}

ASvgfDenoiser::~ASvgfDenoiser() {}

GLuint ASvgfDenoiser::Denoise(BunchOfTexture &textures, int current_frame) {

  glUseProgram(gradient_image_program);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, textures.position_texture[current_frame % 2]);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D,
                textures.position_texture[1 - current_frame % 2]);

  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, textures.normal_texture[current_frame % 2]);
  glActiveTexture(GL_TEXTURE3);
  glBindTexture(GL_TEXTURE_2D, textures.normal_texture[1 - current_frame % 2]);

  glActiveTexture(GL_TEXTURE4);
  glBindTexture(GL_TEXTURE_2D, textures.depth_texture[current_frame % 2]);
  glActiveTexture(GL_TEXTURE5);
  glBindTexture(GL_TEXTURE_2D, textures.depth_texture[1 - current_frame % 2]);

  glActiveTexture(GL_TEXTURE6);
  glBindTexture(GL_TEXTURE_2D, textures.visibility_texture[current_frame % 2]);
  glActiveTexture(GL_TEXTURE7);
  glBindTexture(GL_TEXTURE_2D,
                textures.visibility_texture[1 - current_frame % 2]);

  glActiveTexture(GL_TEXTURE8);
  glBindTexture(GL_TEXTURE_2D, textures.noisy_texture[current_frame % 2]);
  glActiveTexture(GL_TEXTURE9);
  glBindTexture(GL_TEXTURE_2D, textures.noisy_texture[1 - current_frame % 2]);

  glBindImageTexture(10, gradient_texture[current_frame % 2], 0, 0, 0,
                     GL_READ_WRITE, GL_RGBA32F);

  glBindBufferBase(GL_UNIFORM_BUFFER, 0, textures.reprojection_buffer);

  glDispatchCompute(1280 / 16, 720 / 16, 1);

  glMemoryBarrier(GL_ALL_BARRIER_BITS);

  // printf("gradient_texture[current_frame % 2] => %d\n",
  //        gradient_texture[current_frame % 2]);

  return gradient_texture[current_frame % 2];
}
} // namespace Denoiser
} // namespace UniRt