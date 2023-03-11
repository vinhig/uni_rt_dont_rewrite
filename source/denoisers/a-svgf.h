#pragma once

#include "../glad/glad.h"
#include "denoiser.h"

namespace UniRt::Denoiser {
struct ASvgfDenoiser : Denoiser {
public:
  ASvgfDenoiser();
  ~ASvgfDenoiser();

  bool NeedPreTemporalAccumulation() override { return false; }

  GLuint Denoise(BunchOfTexture &textures, int current_frame) override;

  bool NeedPostTemporalAccumulation() override { return false; }

private:
  GLuint gradient_image_program{0};

  GLuint gradient_texture[2];

  BunchOfTexture textures;
};

} // namespace UniRt::Denoiser
