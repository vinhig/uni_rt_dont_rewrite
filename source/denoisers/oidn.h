#pragma once

#include "../glad/glad.h"
#include "denoiser.h"

#include <OpenImageDenoise/oidn.hpp>

namespace UniRt::Denoiser {
struct OidnDenoiser : Denoiser {
public:
  OidnDenoiser();

  ~OidnDenoiser() = default;

  bool NeedPreTemporalAccumulation() override { return true; }

  GLuint Denoise(BunchOfTexture &textures, int current_frame) override;

  bool NeedPostTemporalAccumulation() override { return false; }

  void ReprojectSeed(BunchOfTexture &textures, int current_frame) override;

private:
  float *color_buffer;
  float *normal_buffer;
  float *albedo_buffer;
  float *output_buffer;

  oidn::DeviceRef device;
  oidn::FilterRef filter;

  GLuint denoised_texture;
  GLuint seed_program;
};
} // namespace UniRt::Denoiser
