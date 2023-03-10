#pragma once

#include "../glad/glad.h"
#include "denoiser.h"

namespace UniRt::Denoiser {
struct BmfrDenoiser : Denoiser {
private:
  GLuint tmp_fitting_texture;
  GLuint out_fitting_texture;

  GLuint denoised_texture[2];

  GLuint bmfr_program;

  GLuint per_frame_buffer;

  const int BLOCK_EDGE_LENGTH = 32;

  struct PerFrameCB {
    float target_dim[2]{1280, 720};
    int frame_number{00};
    int horizontal_blocks_count{0};
  };

  PerFrameCB per_frame;

public:
  BmfrDenoiser();
  ~BmfrDenoiser();
  GLuint Denoise(unsigned current_frame, GLuint noisy_texture,
                 GLuint position_texture, GLuint normal_texture,
                 GLuint depth_texture, GLuint albedo_texture) override;

  bool DidSomething() override { return true; }
};

} // namespace UniRt::Denoiser
