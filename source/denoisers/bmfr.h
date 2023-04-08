#pragma once

#include "../glad/glad.h"
#include "denoiser.h"

#define W 8
#define M 4

namespace UniRt::Denoiser {
struct BmfrDenoiser : Denoiser {
private:
  GLuint tmp_fitting_texture;
  GLuint out_fitting_texture;

  GLuint denoised_texture[2];

  GLuint bmfr_program;
  GLuint bmfr_reprojection_program;

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

  GLuint debug_1;
  GLuint debug_2;
  GLuint debug_3;
  GLuint debug_4;

  bool NeedPreTemporalAccumulation() override { return false; }

  GLuint Denoise(BunchOfTexture &textures, int current_frame) override;

  bool NeedPostTemporalAccumulation() override { return false; }

  void ReprojectSeed(BunchOfTexture &textures, int current_frame) override;
};

} // namespace UniRt::Denoiser
