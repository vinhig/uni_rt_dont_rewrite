#pragma once

#include "../glad/glad.h"
#include "denoiser.h"

namespace UniRt::Denoiser {
struct NoneDenoiser : Denoiser {
public:
  NoneDenoiser() { printf("hello from none denoiser\n"); };

  ~NoneDenoiser() = default;

  bool NeedPreTemporalAccumulation() override { return false; }

  GLuint Denoise(BunchOfTexture &textures, int current_frame) override {
    return textures.noisy_texture[current_frame % 2];
  };

  bool NeedPostTemporalAccumulation() override { return false; }

  void ReprojectSeed(BunchOfTexture &textures, int current_frame) override{};
};
} // namespace UniRt::Denoiser
