#pragma once

#include "glad/glad.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <vector>

#include <embree3/rtcore.h>

#include "embree_utils.h"
#include "lights.h"
#include "material.h"
#include "mesh.h"

struct SDL_Window;

namespace UniRt {
class Scene;

class Render {

public:
  Render();
  ~Render();

  bool Update();

  void SetScene(UniRt::Scene *scene, std::string scene_name);

  struct RotatingCamera {
    float angle;
    float speed;

    float near{0.1f};
    float far{100.0f};

    float distance;

    glm::vec3 offset;

    glm::mat4 proj;
    glm::mat4 view_proj;
    glm::mat4 prev_view_proj;

    void Update(double delta) {
      offset.x = cos(angle) * distance;
      offset.y = distance / 3.0 * 2.0;
      offset.z = sin(angle) * distance;

      angle += delta * speed;

      prev_view_proj = view_proj;
      proj = glm::perspective(glm::radians(45.0f), 16.0f / 9.0f, near, far);
      view_proj =
          proj * glm::lookAt(offset, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
    }
  };

  RotatingCamera camera;

private:
  SDL_Window *window;

  RTCDevice device;

  glm::uvec2 tile_size = glm::uvec2(64);

  std::vector<float> img_color;
  std::vector<float> img_shadow;
  std::vector<float> img_albedo;
  std::vector<std::vector<float>> tiles_color;
  std::vector<std::vector<float>> tiles_shadow;
  std::vector<std::vector<float>> tiles_albedo;
  std::vector<float> position_texture_pixels;
  std::vector<float> normal_texture_pixels;

  unsigned current_frame{0};

  GLuint features_fbo[2];
  GLuint position_texture[2];
  GLuint normal_texture[2];
  // GLuint visibility_texture;
  GLuint depth_texture[2];

  GLuint quad_program;
  GLuint features_program;
  GLuint ta_program;

  // GLuint features_vao;
  GLuint transform_location;
  GLuint view_proj_location;
  GLuint instance_id_location;
  GLuint geometry_id_location;
  GLuint prim_id_location;

  GLuint shadow_texture[2];
  GLuint albedo_texture[2];

  struct ReprojectionCB {
    float view_proj[4][4];
    float inv_view_proj[4][4];
    float prev_view_proj[4][4];
    float proj[4][4];
    float view_pos[4];
    float target_dim[2]{1250.0f, 720.0f};
    float alpha_illum{0.05f};
    float alpha_moments{0.05f};
    float phi_depth{0.369f};
    float phi_normal{0.3f};
    float depth_tolerance{0.75};
    float normal_tolerance{0.75};
    float min_accum_weight{0.15};
    int frame_number;
  };
  ReprojectionCB reprojection = {};

  bool temporal_accumulation{true};

  GLuint accumulated_texture[2];
  GLuint moment_texture[2];
  GLuint history_length[2];
  GLuint reprojection_buffer;


  struct Geometry {
    GLuint vertex_buffer;
    GLuint index_buffer;

    GLuint vao;

    unsigned long triangle_count;
  };

  struct Instance {
    glm::mat4 transform;

    std::vector<Geometry> geometries;
  };

  std::vector<Instance> instances;

  std::string current_scene_name;

  std::vector<ParameterizedMesh> parameterized_meshes;
  std::shared_ptr<embree::TopLevelBVH> scene_bvh;

  std::vector<embree::MaterialParams> material_params;
  std::vector<QuadLight> lights;
  std::vector<Image> textures;
  std::vector<embree::ISPCTexture2D> ispc_textures;

  void SetupEmbree();

  bool UpdateSDL();

  void DrawDenoise();

  void DrawEmbree();

  void DrawGUI();

  void DrawFeatureBuffers();

  void SetSceneOpenGL(UniRt::Scene *scene);

  void SetSceneEmbree(UniRt::Scene *scene);

  void TemporalAccumulation();
};
} // namespace UniRt
