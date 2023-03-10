#pragma once

#include "../glad/glad.h"
#include "denoiser.h"

namespace UniRt::Denoiser {
struct NoneDenoiser : Denoiser {
public:
  NoneDenoiser() {
    printf("hello from none denoiser\n");
  };
  ~NoneDenoiser() = default;
  GLuint Denoise(unsigned current_frame, GLuint noisy_texture, GLuint position_texture,
                 GLuint normal_texture, GLuint depth_texture,
                 GLuint albedo_texture) override {
    return noisy_texture;
  };

  bool DidSomething() override { return false; }
};
} // namespace UniRt::Denoiser
