#pragma once

#include "../glad/glad.h"
#include "denoiser.h"

namespace UniRt::Denoiser {
struct BmfrRenewDenoiser : Denoiser {
private:
  GLuint tmp_fitting_texture;
  GLuint out_fitting_texture;

  GLuint denoised_texture[2];

  GLuint bmfr_program;
  GLuint bmfr_reprojection_program;

  GLuint per_frame_buffer;

  struct PerFrameCB {
    float target_dim[2]{1280, 720};
    int frame_number{00};
    int horizontal_blocks_count{0};
  };

  PerFrameCB per_frame;

public:
  BmfrRenewDenoiser();
  ~BmfrRenewDenoiser();

  GLuint debug_1;
  GLuint debug_2;
  GLuint debug_3;
  GLuint debug_4;

  GLuint red_tilde;
  GLuint green_tilde;
  GLuint blue_tilde;

  GLuint tilde;
  GLuint r;
  GLuint tmp_in_tilde;
  GLuint tmp_out_tilde;
  GLuint h_tmp;
  GLuint a_tmp;

  bool NeedPreTemporalAccumulation() override { return true; }

  GLuint Denoise(BunchOfTexture &textures, int current_frame) override;

  bool NeedPostTemporalAccumulation() override { return true; }

  void ReprojectSeed(BunchOfTexture &textures, int current_frame) override;
};

} // namespace UniRt::Denoiser
