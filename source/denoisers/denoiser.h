#pragma once

#include "../glad/glad.h"

namespace UniRt::Denoiser {
struct BunchOfTexture {
  // the thing to denoise
  GLuint noisy_texture[2];

  GLuint position_texture[2];
  GLuint normal_texture[2];
  GLuint visibility_texture[2];
  GLuint albedo_texture[2];
  GLuint depth_texture[2];
  GLuint geo_normal_texture[2];
  GLuint motion_texture[2];

  // reprojection uniform buffer
  GLuint reprojection_buffer;
};
struct Denoiser {

public:
  // Given by Renderer
  // Not create by the denoiser

  virtual bool NeedPreTemporalAccumulation() = 0;

  virtual GLuint Denoise(BunchOfTexture &textures, int current_frame) = 0;

  virtual bool NeedPostTemporalAccumulation() = 0;
};
} // namespace UniRt::Denoiser
