#pragma once

#include <string>

#include <glm/glm.hpp>

#include "lights.h"
#include "material.h"
#include "mesh.h"
#include "phmap.h"

namespace UniRt {
struct Scene {
  std::vector<Mesh> meshes;
  std::vector<ParameterizedMesh> parameterized_meshes;
  std::vector<Instance> instances;
  std::vector<DisneyMaterial> materials;
  std::vector<Image> textures;
  std::vector<QuadLight> lights;

  Scene(Scene &scene) = delete;
  Scene(const std::string &fname);
  Scene() = default;

  // Compute the unique number of triangles in the scene
  size_t unique_tris() const;

  // Compute the total number of triangles in the scene (after instancing)
  size_t total_tris() const;

  size_t num_geometries() const;

private:
  void load_gltf(const std::string &file);

  void validate_materials();
};
} // namespace UniRt