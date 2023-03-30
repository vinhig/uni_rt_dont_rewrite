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

  void ReprojectSeed(BunchOfTexture &textures, int current_frame);

private:
  struct DenoisingCB {
    float flt_antilag{1.0f};
    float flt_temporal{1.0f};
    float flt_min_alpha_color{0.02f};
    float flt_min_alpha_moments{0.1f};
    float flt_atrous{4.0f};
    float flt_atrous_lum{16.0f};
    float flt_atrous_normal{64.0f};
    float flt_atrous_depth{0.5f};
  };
  DenoisingCB denoising;

  GLuint denoising_buffer{0};

  GLuint gradient_reproject_program{0};
  GLuint gradient_just_program{0};
  GLuint gradient_atrous_program{0};
  GLuint temporal_program{0};
  GLuint color_atrous_program{0};

  GLuint uniform_grad_push_iteration_location{0};
  GLuint uniform_color_push_iteration_location{0};

  // GLuint gradient_reproject_texture[2];
  GLuint gradient_texture[2];

  GLuint gradient_ping_texture;
  GLuint gradient_pong_texture;
  GLuint color_ping_texture;
  GLuint color_pong_texture;
  GLuint moment_ping_texture;
  GLuint moment_pong_texture;
  GLuint atrous_ping_texture;
  GLuint atrous_pong_texture;

  GLuint reprojected_luminance_texture[2];

  GLuint moments_texture[2];
  GLuint hist_len_texture;
  // GLuint gradient_a_trous_texture[2];

  GLuint debug_texture;

  GLuint blue_noise_texture;

  BunchOfTexture textures;
};

} // namespace UniRt::Denoiser
