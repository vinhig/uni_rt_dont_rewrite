#include "none.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace UniRt {

GLuint CompileShader(const char *path, const char *source, GLenum shaderType);

namespace Denoiser {
NoneDenoiser::NoneDenoiser() {
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
    printf("hello from none denoiser\n");
  };
}

void NoneDenoiser::ReprojectSeed(BunchOfTexture &textures, int current_frame) {
  glUseProgram(reprojection_program);

  glBindImageTexture(0, textures.rng_seed_texture[current_frame % 2], 0, 0, 0,
                     GL_READ_WRITE, GL_RGBA32UI);

  glBindBufferBase(GL_UNIFORM_BUFFER, 0, textures.reprojection_buffer);

  glDispatchCompute((1280 + 15) / 16, (720 + 15) / 16, 1);

  glMemoryBarrier(GL_ALL_BARRIER_BITS);
}
} // namespace Denoiser
} // namespace UniRt