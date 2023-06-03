#pragma once

#include "../glad/glad.h"
#include "denoiser.h"

namespace UniRt::Denoiser {
struct TnndDenoiser : Denoiser {
private:
  GLuint reprojection_program;
  GLuint tnnd_growing_program;
  GLuint tnnd_deep_image_program;

  GLuint ping_texture;
  GLuint pong_texture;
  GLuint denoised_texture;

  GLuint conv_weights_buffer;
  GLuint nn_weights_buffer;

  GLuint growing_unroll_3_uniform;
  GLuint growing_unroll_5_uniform;
  GLuint growing_channel_in_uniform;
  GLuint growing_depth_uniform;

  GLuint deep_image_unroll_3_uniform;
  GLuint deep_image_unroll_5_uniform;
  GLuint deep_image_channel_in_uniform;
  GLuint deep_image_depth_uniform;

  int iteration = 3;

public:
  TnndDenoiser();

  ~TnndDenoiser() = default;

  bool NeedPreTemporalAccumulation() override { return true; }

  GLuint Denoise(BunchOfTexture &textures, int current_frame) override;

  bool NeedPostTemporalAccumulation() override { return false; }

  void ReprojectSeed(BunchOfTexture &textures, int current_frame) override;
};
} // namespace UniRt::Denoiser
