#pragma once

#include "../glad/glad.h"
#include "denoiser.h"

namespace UniRt::Denoiser {
struct AccumulatorDenoiser : Denoiser {
public:
  AccumulatorDenoiser();

  ~AccumulatorDenoiser() = default;

  bool NeedPreTemporalAccumulation() override { return false; }

  GLuint Denoise(BunchOfTexture &textures, int current_frame);

  bool NeedPostTemporalAccumulation() override { return false; }

  void ReprojectSeed(BunchOfTexture &textures, int current_frame) override;

private:
    GLuint accum_program;
    GLuint reprojection_program;

    GLuint accum_texture;

};
} // namespace UniRt::Denoiser
