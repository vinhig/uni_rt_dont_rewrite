#pragma once

#include "glad/glad.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <vector>

struct SDL_Window;

namespace UniRt {
class Scene;

class Render {

public:
  Render();
  ~Render();

  bool Update();

  void SetScene(Scene *scene, std::string scene_name);

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

  GLuint features_fbo;
  GLuint position_texture;
  GLuint normal_texture;
  GLuint depth_texture;
  GLuint features_program;
  // GLuint features_vao;
  GLuint transform_location;
  GLuint view_proj_location;

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
};
} // namespace UniRt
