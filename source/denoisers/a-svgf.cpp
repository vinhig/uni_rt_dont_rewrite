#include "a-svgf.h"

#include <stdio.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string.h>

#include "../stb_image.h"

namespace UniRt {

GLuint CompileShader(const char *path, const char *source, GLenum shaderType);

namespace Denoiser {
ASvgfDenoiser::ASvgfDenoiser() {
  // {
  //   std::ifstream comp_file("../shaders/asvgf_gradient_reproject.comp.glsl");
  //   std::ostringstream comp_ss;
  //   comp_ss << comp_file.rdbuf();
  //   std::string comp_source = comp_ss.str();

  //   GLuint comp_shader =
  //       CompileShader("../shader/asvgf_gradient_reproject.comp.glsl",
  //                     comp_source.c_str(), GL_COMPUTE_SHADER);

  //   gradient_reproject_program = glCreateProgram();
  //   glAttachShader(gradient_reproject_program, comp_shader);
  //   glLinkProgram(gradient_reproject_program);

  //   glDeleteShader(comp_shader);
  // }
  {
    std::ifstream comp_file("../shaders/asvgf_gradient_reprojection.comp.glsl");
    std::ostringstream comp_ss;
    comp_ss << comp_file.rdbuf();
    std::string comp_source = comp_ss.str();

    GLuint comp_shader =
        CompileShader("../shader/asvgf_just_gradient.comp.glsl",
                      comp_source.c_str(), GL_COMPUTE_SHADER);

    gradient_just_program = glCreateProgram();
    glAttachShader(gradient_just_program, comp_shader);
    glLinkProgram(gradient_just_program);

    glDeleteShader(comp_shader);
  }
  {
    std::ifstream comp_file("../shaders/asvgf_gradient_atrous.comp.glsl");
    std::ostringstream comp_ss;
    comp_ss << comp_file.rdbuf();
    std::string comp_source = comp_ss.str();

    GLuint comp_shader =
        CompileShader("../shader/asvgf_gradient_atrous.comp.glsl",
                      comp_source.c_str(), GL_COMPUTE_SHADER);

    gradient_atrous_program = glCreateProgram();
    glAttachShader(gradient_atrous_program, comp_shader);
    glLinkProgram(gradient_atrous_program);

    glDeleteShader(comp_shader);
  }
  {
    std::ifstream comp_file("../shaders/asvgf_temporal.comp.glsl");
    std::ostringstream comp_ss;
    comp_ss << comp_file.rdbuf();
    std::string comp_source = comp_ss.str();

    GLuint comp_shader = CompileShader("../shader/asvgf_temporal.comp.glsl",
                                       comp_source.c_str(), GL_COMPUTE_SHADER);

    temporal_program = glCreateProgram();
    glAttachShader(temporal_program, comp_shader);
    glLinkProgram(temporal_program);

    glDeleteShader(comp_shader);
  }
  {
    std::ifstream comp_file("../shaders/asvgf_color_atrous.comp.glsl");
    std::ostringstream comp_ss;
    comp_ss << comp_file.rdbuf();
    std::string comp_source = comp_ss.str();

    GLuint comp_shader = CompileShader("../shader/asvgf_color_atrous.comp.glsl",
                                       comp_source.c_str(), GL_COMPUTE_SHADER);

    color_atrous_program = glCreateProgram();
    glAttachShader(color_atrous_program, comp_shader);
    glLinkProgram(color_atrous_program);

    glDeleteShader(comp_shader);
  }

  uniform_grad_push_iteration_location =
      glGetUniformLocation(gradient_atrous_program, "push_iteration");
  uniform_color_push_iteration_location =
      glGetUniformLocation(color_atrous_program, "push_iteration");
  // Load blue noise texture hehehe
  {
    glGenTextures(1, &blue_noise_texture);
    glBindTexture(GL_TEXTURE_3D, blue_noise_texture);

    unsigned char *data = new unsigned char[256 * 256 * 4 * 128];

    for (int i = 0; i < 128; i++) {
      int w, h, n;
      char buf[1024];

      snprintf(buf, sizeof buf,
               "../textures/blue_noise/256_256/HDR_RGBA_%04d.png", i);

      stbi_uc *img_data = stbi_load(buf, &w, &h, &n, 4);

      if (img_data == NULL) {
        printf("Couldn't load blue noise texture '%s'\n", buf);
      }

      memcpy(&data[256 * 256 * 4 * i], img_data,
             256 * 256 * 4 * sizeof(unsigned char));

      stbi_image_free(img_data);
    }

    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, 256, 256, 128, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    delete data;
  }

  glGenTextures(2, reprojected_luminance_texture);

  glGenTextures(2, gradient_texture);

  glGenTextures(2, moments_texture);
  glGenTextures(1, &hist_len_texture);

  glGenTextures(1, &gradient_ping_texture);
  glGenTextures(1, &gradient_pong_texture);

  glGenTextures(1, &color_ping_texture);
  glGenTextures(1, &color_pong_texture);
  glGenTextures(1, &moment_ping_texture);
  glGenTextures(1, &moment_pong_texture);
  glGenTextures(1, &atrous_ping_texture);
  glGenTextures(1, &atrous_pong_texture);

  glGenTextures(1, &debug_texture);

  glBindTexture(GL_TEXTURE_2D, debug_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
               nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glBindTexture(GL_TEXTURE_2D, gradient_ping_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280 / 3, 720 / 3, 0, GL_RGBA,
               GL_FLOAT, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glBindTexture(GL_TEXTURE_2D, gradient_pong_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280 / 3, 720 / 3, 0, GL_RGBA,
               GL_FLOAT, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glBindTexture(GL_TEXTURE_2D, color_ping_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
               nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glBindTexture(GL_TEXTURE_2D, color_pong_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
               nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glBindTexture(GL_TEXTURE_2D, moment_ping_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
               nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glBindTexture(GL_TEXTURE_2D, moment_pong_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
               nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glBindTexture(GL_TEXTURE_2D, atrous_ping_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
               nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glBindTexture(GL_TEXTURE_2D, atrous_pong_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
               nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glBindTexture(GL_TEXTURE_2D, hist_len_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R8UI, 1280, 720, 0, GL_RED_INTEGER,
               GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  for (int i = 0; i < 2; i++) {
    // glBindTexture(GL_TEXTURE_2D, gradient_reproject_texture[i]);
    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280 / 3, 720 / 3, 0, GL_RGBA,
    //              GL_FLOAT, nullptr);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, reprojected_luminance_texture[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280 / 3, 720 / 3, 0, GL_RGBA,
                 GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, gradient_texture[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280 / 3, 720 / 3, 0, GL_RGBA,
                 GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, moments_texture[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
                 nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  }

  glGenBuffers(1, &denoising_buffer);
  glBindBuffer(GL_UNIFORM_BUFFER, denoising_buffer);
  glBufferData(GL_UNIFORM_BUFFER, sizeof(DenoisingCB), &denoising,
               GL_DYNAMIC_DRAW);

  printf("hello from a-svgf denoiser\n");

  glObjectLabel(GL_TEXTURE, gradient_texture[0], -1, "gradient_texture[0]");
  glObjectLabel(GL_TEXTURE, gradient_texture[1], -1, "gradient_texture[1]");

  glObjectLabel(GL_TEXTURE, reprojected_luminance_texture[0], -1,
                "reprojected_luminance_texture[0]");
  glObjectLabel(GL_TEXTURE, reprojected_luminance_texture[1], -1,
                "reprojected_luminance_texture[1]");

  glObjectLabel(GL_TEXTURE, gradient_ping_texture, -1, "gradient_ping_texture");
  glObjectLabel(GL_TEXTURE, gradient_ping_texture, -1, "gradient_pong_texture");
}

ASvgfDenoiser::~ASvgfDenoiser() {}

void ASvgfDenoiser::ReprojectSeed(BunchOfTexture &textures, int current_frame) {
  {
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "ASVGF JUST GRADIENT");
    glUseProgram(gradient_just_program);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D,
                  textures.geo_normal_texture[current_frame % 2]);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D,
                  textures.geo_normal_texture[1 - current_frame % 2]);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, textures.depth_texture[current_frame % 2]);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, textures.depth_texture[1 - current_frame % 2]);

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, textures.noisy_texture[1 - current_frame % 2]);

    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, textures.rng_seed_texture[current_frame % 2]);
    glActiveTexture(GL_TEXTURE6);
    glBindTexture(GL_TEXTURE_2D,
                  textures.rng_seed_texture[1 - current_frame % 2]);

    glBindImageTexture(7, textures.rng_seed_texture[current_frame % 2], 0, 0, 0,
                       GL_READ_WRITE, GL_RGBA32I);

    glBindImageTexture(8, reprojected_luminance_texture[current_frame % 2], 0,
                       0, 0, GL_READ_WRITE, GL_RGBA32F);

    glBindBufferBase(GL_UNIFORM_BUFFER, 0, textures.reprojection_buffer);

    unsigned group_size_pixels = 24;
    glDispatchCompute((1280 + group_size_pixels - 1) / group_size_pixels,
                      (720 + group_size_pixels - 1) / group_size_pixels, 1);

    glMemoryBarrier(GL_ALL_BARRIER_BITS);

    glPopDebugGroup();
  }
}

GLuint ASvgfDenoiser::Denoise(BunchOfTexture &textures, int current_frame) {
  // Gradient reprojection
  // {
  //   glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1,
  //                    "ASVGF GRADIENT REPROJECTION");
  //   glUseProgram(gradient_reproject_program);

  //   glActiveTexture(GL_TEXTURE0);
  //   glBindTexture(GL_TEXTURE_2D, textures.normal_texture[current_frame % 2]);
  //   glActiveTexture(GL_TEXTURE1);
  //   glBindTexture(GL_TEXTURE_2D,
  //                 textures.normal_texture[1 - current_frame % 2]);

  //   glActiveTexture(GL_TEXTURE2);
  //   glBindTexture(GL_TEXTURE_2D, textures.depth_texture[current_frame % 2]);
  //   glActiveTexture(GL_TEXTURE3);
  //   glBindTexture(GL_TEXTURE_2D, textures.depth_texture[1 - current_frame %
  //   2]);

  //   glActiveTexture(GL_TEXTURE4);
  //   glBindTexture(GL_TEXTURE_2D, textures.noisy_texture[current_frame % 2]);
  //   glActiveTexture(GL_TEXTURE5);
  //   glBindTexture(GL_TEXTURE_2D, textures.noisy_texture[1 - current_frame %
  //   2]);

  //   glActiveTexture(GL_TEXTURE6);
  //   glBindTexture(GL_TEXTURE_2D, textures.rng_seed_texture[current_frame %
  //   2]); glActiveTexture(GL_TEXTURE7); glBindTexture(GL_TEXTURE_2D,
  //                 textures.rng_seed_texture[1 - current_frame % 2]);

  //   glBindImageTexture(8, gradient_reproject_texture[current_frame % 2], 0,
  //   0,
  //                      0, GL_READ_WRITE, GL_RGBA32F);

  //   glBindImageTexture(9, textures.rng_seed_texture[current_frame % 2], 0, 0,
  //   0,
  //                      GL_READ_WRITE, GL_RGBA32I);

  //   glBindBufferBase(GL_UNIFORM_BUFFER, 0, textures.reprojection_buffer);

  //   unsigned group_size_pixels = 24;
  //   glDispatchCompute((1280 + group_size_pixels - 1) / group_size_pixels,
  //                     (720 + group_size_pixels - 1) / group_size_pixels, 1);

  //   glMemoryBarrier(GL_ALL_BARRIER_BITS);

  //   // printf("gradient_texture[current_frame % 2] => %d\n",
  //   //        gradient_texture[current_frame % 2]);

  //   glPopDebugGroup();
  // }

  // Gradient computation

  {}

  return reprojected_luminance_texture[current_frame % 2];

  // Gradient à-trous
  {
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1,
                     "ASVGF A-TROUS GRADIENT");
    glUseProgram(gradient_atrous_program);

    const int num_atrous_iterations_gradient = 3;
    for (int i = 0; i < num_atrous_iterations_gradient; i++) {
      glUniform1i(uniform_grad_push_iteration_location, i);

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, gradient_ping_texture);

      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, gradient_pong_texture);

      glBindImageTexture(2, gradient_ping_texture, 0, 0, 0, GL_READ_WRITE,
                         GL_RGBA32F);

      glBindImageTexture(3, gradient_pong_texture, 0, 0, 0, GL_READ_WRITE,
                         GL_RGBA32F);

      glDispatchCompute((1280 / 3 + 15) / 16, (720 / 3 + 15) / 16, 1);
      glMemoryBarrier(GL_ALL_BARRIER_BITS);
    }

    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    glPopDebugGroup();
  }

  // Temporal accumulation with the famous anti-lag stuff
  {
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1,
                     "ASVGF TEMPORAL ACCUMULATION");
    glUseProgram(temporal_program);

    glBindBufferBase(GL_UNIFORM_BUFFER, 0, textures.reprojection_buffer);
    glBindBufferBase(GL_UNIFORM_BUFFER, 1, denoising_buffer);

    // TODO: should REALLY bind every fucking textures in one call instead of
    // repeating this pile of shit every time

    // t_curr_normal
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textures.normal_texture[current_frame % 2]);
    // t_prev_normal
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D,
                  textures.normal_texture[1 - current_frame % 2]);

    // t_prev_moments
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, moments_texture[1 - current_frame % 2]);

    // t_curr_indirect
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, textures.noisy_texture[current_frame % 2]);

    // t_prev_accumulated
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, atrous_pong_texture);

    // t_out_accumulated
    glBindImageTexture(5, atrous_ping_texture, 0, 0, 0, GL_READ_WRITE,
                       GL_RGBA32F);

    // t_out_moments
    glBindImageTexture(6, moments_texture[current_frame % 2], 0, 0, 0,
                       GL_READ_WRITE, GL_RGBA32F);

    // t_out_history_length
    glBindImageTexture(7, hist_len_texture, 0, 0, 0, GL_READ_WRITE, GL_R8UI);

    // t_curr_depth
    glActiveTexture(GL_TEXTURE8);
    glBindTexture(GL_TEXTURE_2D, textures.depth_texture[current_frame % 2]);
    // t_prev_depth
    glActiveTexture(GL_TEXTURE9);
    glBindTexture(GL_TEXTURE_2D, textures.depth_texture[1 - current_frame % 2]);

    glActiveTexture(GL_TEXTURE10);
    glBindTexture(GL_TEXTURE_2D, gradient_pong_texture);

    glBindImageTexture(11, debug_texture, 0, 0, 0, GL_READ_WRITE, GL_RGBA32F);

    glDispatchCompute((1280 + 14) / 15, (720 + 14) / 15, 1);

    glMemoryBarrier(GL_ALL_BARRIER_BITS);

    glPopDebugGroup();
  }

  // Color à-trous
  {
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "ASVGF A-TROUS COLOR");
    glUseProgram(color_atrous_program);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textures.position_texture[current_frame % 2]);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D,
                  textures.position_texture[1 - current_frame % 2]);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, textures.normal_texture[current_frame % 2]);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D,
                  textures.normal_texture[1 - current_frame % 2]);

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D,
                  textures.geo_normal_texture[current_frame % 2]);
    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D,
                  textures.geo_normal_texture[1 - current_frame % 2]);

    glActiveTexture(GL_TEXTURE6);
    glBindTexture(GL_TEXTURE_2D, textures.depth_texture[current_frame % 2]);
    glActiveTexture(GL_TEXTURE7);
    glBindTexture(GL_TEXTURE_2D, textures.depth_texture[1 - current_frame % 2]);

    glActiveTexture(GL_TEXTURE8);
    glBindTexture(GL_TEXTURE_2D, textures.motion_texture[current_frame % 2]);
    glActiveTexture(GL_TEXTURE9);
    glBindTexture(GL_TEXTURE_2D,
                  textures.motion_texture[1 - current_frame % 2]);

    glActiveTexture(GL_TEXTURE10);
    glBindTexture(GL_TEXTURE_2D, moments_texture[current_frame % 2]);
    glActiveTexture(GL_TEXTURE11);
    glBindTexture(GL_TEXTURE_2D, moments_texture[1 - current_frame % 2]);

    glActiveTexture(GL_TEXTURE21);
    glBindTexture(GL_TEXTURE_2D, hist_len_texture);

    glActiveTexture(GL_TEXTURE12);
    glBindTexture(GL_TEXTURE_3D, blue_noise_texture);

    glBindImageTexture(13, moment_ping_texture, 0, 0, 0, GL_READ_WRITE,
                       GL_RGBA32F);
    glBindImageTexture(14, moment_pong_texture, 0, 0, 0, GL_READ_WRITE,
                       GL_RGBA32F);

    glBindImageTexture(15, atrous_ping_texture, 0, 0, 0, GL_READ_WRITE,
                       GL_RGBA32F);
    glBindImageTexture(16, atrous_pong_texture, 0, 0, 0, GL_READ_WRITE,
                       GL_RGBA32F);

    glActiveTexture(GL_TEXTURE17);
    glBindTexture(GL_TEXTURE_2D, moment_ping_texture);
    glActiveTexture(GL_TEXTURE18);
    glBindTexture(GL_TEXTURE_2D, moment_pong_texture);

    glActiveTexture(GL_TEXTURE19);
    glBindTexture(GL_TEXTURE_2D, atrous_ping_texture);
    glActiveTexture(GL_TEXTURE20);
    glBindTexture(GL_TEXTURE_2D, atrous_pong_texture);

    const int num_atrous_iterations = 4;
    for (int i = 0; i < 4; i++) {
      glUniform1i(uniform_color_push_iteration_location, i);
      glDispatchCompute((1280 + 15) / 16, (720 + 15) / 16, 1);
    }

    glPopDebugGroup();
  }

  return gradient_pong_texture;
}
} // namespace Denoiser
} // namespace UniRt