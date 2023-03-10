#pragma once

#include "../glad/glad.h"

namespace UniRt::Denoiser {
struct Denoiser {

public:
  virtual GLuint Denoise(unsigned current_frame, GLuint noisy_texture,
                         GLuint position_texture, GLuint normal_texture,
                         GLuint depth_texture, GLuint albedo_texture) = 0;

  virtual bool DidSomething() = 0;
};
} // namespace UniRt::Denoiser
