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

    float distance;

    glm::vec3 offset;

    glm::mat4 view_proj;

    void Update(double delta) {
      offset.x = cos(angle) * distance;
      offset.y = distance / 3.0 * 2.0;
      offset.z = sin(angle) * distance;

      angle += delta * speed;

      view_proj =
          glm::perspective(glm::radians(45.0f), 16.0f / 9.0f, 0.1f, 150.0f) *
          glm::lookAt(offset, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
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

  GLuint features_fbo;
  GLuint position_texture;
  GLuint normal_texture;
  // GLuint visibility_texture;
  GLuint depth_texture;

  GLuint features_program;
  // GLuint features_vao;
  GLuint transform_location;
  GLuint view_proj_location;
  GLuint instance_id_location;
  GLuint geometry_id_location;
  GLuint prim_id_location;

  GLuint shadow_texture;
  GLuint albedo_texture;

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
};
} // namespace UniRt
