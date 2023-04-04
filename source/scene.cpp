#include "scene.h"
#include "buffer_view.h"
#include "file_mapping.h"
#include "flatten_gltf.h"
#include "gltf_types.h"
#include "json.hpp"
#include "phmap_utils.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "stb_image.h"
#include "stb_image_write.h"
#include "tiny_gltf.h"

#include "util.h"
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <map>

namespace std {
template <> struct hash<glm::uvec2> {
  size_t operator()(glm::uvec2 const &v) const {
    return phmap::HashState().combine(0, v.x, v.y);
  }
};

bool operator==(const glm::uvec2 &a, const glm::uvec2 &b) {
  return a.x == b.x && a.y == b.y;
}

template <> struct hash<glm::uvec3> {
  size_t operator()(glm::uvec3 const &v) const {
    return phmap::HashState().combine(0, v.x, v.y, v.z);
  }
};
} // namespace std

bool operator==(const glm::uvec3 &a, const glm::uvec3 &b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

namespace UniRt {
Scene::Scene(const std::string &fname) {
  const std::string ext = get_file_extension(fname);
  if (ext == "gltf" || ext == "glb") {
    load_gltf(fname);
  } else {
    std::cout << "Unsupported file type '" << ext << "'\n";
    throw std::runtime_error("Unsupported file type " + ext);
  }
}

size_t Scene::unique_tris() const {
  return std::accumulate(
      meshes.begin(), meshes.end(), 0,
      [](const size_t &n, const Mesh &m) { return n + m.num_tris(); });
}

size_t Scene::total_tris() const {
  return std::accumulate(
      instances.begin(), instances.end(), 0,
      [&](const size_t &n, const Instance &i) {
        return n + meshes[parameterized_meshes[i.parameterized_mesh_id].mesh_id]
                       .num_tris();
      });
}

size_t Scene::num_geometries() const {
  return std::accumulate(
      meshes.begin(), meshes.end(), 0,
      [](const size_t &n, const Mesh &m) { return n + m.geometries.size(); });
}

void Scene::load_gltf(const std::string &fname) {
  std::cout << "Loading GLTF " << fname << "\n";

  tinygltf::Model model;
  tinygltf::TinyGLTF context;
  std::string err, warn;
  bool ret = false;
  if (get_file_extension(fname) == "gltf") {
    ret = context.LoadASCIIFromFile(&model, &err, &warn, fname.c_str());
  } else {
    ret = context.LoadBinaryFromFile(&model, &err, &warn, fname.c_str());
  }

  if (!warn.empty()) {
    std::cout << "TinyGLTF loading: " << fname << " warnings: " << warn << "\n";
  }

  if (!ret || !err.empty()) {
    throw std::runtime_error("TinyGLTF Error loading " + fname +
                             " error: " + err);
  }

  if (model.defaultScene == -1) {
    model.defaultScene = 0;
  }

  // Find lights
  for (const auto &node : model.nodes) {
    // if (node.extensions.contains("KHR_lights_punctual")) {
    //   auto l = node.extensions["KHR_lights_punctual"]; //
    //   ["light"].GetNumberAsInt();
    // }

    if (node.extensions.count("KHR_lights_punctual") != 0) {
      auto l = node.extensions.at("KHR_lights_punctual");
      auto l_idx = l.Get("light").GetNumberAsInt();

      auto lig = model.lights[l_idx];

      QuadLight light;
      light.emission = glm::vec4(lig.intensity) * 0.05f,
      glm::vec4(lig.color[0], lig.color[1], lig.color[2], 1.0);
      light.normal = glm::vec4(glm::normalize(glm::vec3(0.0f, -1.0, 0.0f)), 0);
      ortho_basis(light.v_x, light.v_y, glm::vec3(light.normal));
      light.width = 1.f;
      light.height = 1.f;
      light.position = glm::vec4(node.translation[0], node.translation[1],
                                 node.translation[2], 1.0);

      std::cout << "Adding light" << std::endl;
      lights.push_back(light);
    }
  }

  flatten_gltf(model);

  // Load the meshes. Note: GLTF combines mesh + material parameters into
  // a single entity, so GLTF "meshes" are ChameleonRT "parameterized meshes"
  for (auto &m : model.meshes) {
    Mesh mesh;
    std::vector<uint32_t> material_ids;
    for (auto &p : m.primitives) {
      Geometry geom;
      material_ids.push_back(p.material);

      if (p.mode != TINYGLTF_MODE_TRIANGLES) {
        std::cout
            << "Unsupported primitive mode! File must contain only triangles\n";
        throw std::runtime_error(
            "Unsupported primitive mode! Only triangles are supported");
      }

      // Note: assumes there is a POSITION (is this required by the gltf spec?)
      Accessor<glm::vec3> pos_accessor(
          model.accessors[p.attributes["POSITION"]], model);
      for (size_t i = 0; i < pos_accessor.size(); ++i) {
        geom.vertices.push_back(pos_accessor[i]);
      }

      // Note: GLTF can have multiple texture coordinates used by different
      // textures (owch) I don't plan to support this
      auto fnd = p.attributes.find("TEXCOORD_0");
      if (fnd != p.attributes.end()) {
        Accessor<glm::vec2> uv_accessor(model.accessors[fnd->second], model);
        for (size_t i = 0; i < uv_accessor.size(); ++i) {
          geom.uvs.push_back(uv_accessor[i]);
        }
      }

      // #if 0
      fnd = p.attributes.find("NORMAL");
      if (fnd != p.attributes.end()) {
        Accessor<glm::vec3> normal_accessor(model.accessors[fnd->second],
                                            model);
        for (size_t i = 0; i < normal_accessor.size(); ++i) {
          geom.normals.push_back(normal_accessor[i]);
        }
      }
      // #endif

      if (model.accessors[p.indices].componentType ==
          TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
        Accessor<uint16_t> index_accessor(model.accessors[p.indices], model);
        for (size_t i = 0; i < index_accessor.size() / 3; ++i) {
          geom.indices.push_back(glm::uvec3(index_accessor[i * 3],
                                            index_accessor[i * 3 + 1],
                                            index_accessor[i * 3 + 2]));
        }
      } else if (model.accessors[p.indices].componentType ==
                 TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
        Accessor<uint32_t> index_accessor(model.accessors[p.indices], model);
        for (size_t i = 0; i < index_accessor.size() / 3; ++i) {
          geom.indices.push_back(glm::uvec3(index_accessor[i * 3],
                                            index_accessor[i * 3 + 1],
                                            index_accessor[i * 3 + 2]));
        }
      } else {
        std::cout << "Unsupported index type\n";
        throw std::runtime_error("Unsupported index component type");
      }
      mesh.geometries.push_back(geom);
    }
    parameterized_meshes.emplace_back(meshes.size(), material_ids);
    meshes.push_back(mesh);
  }

  // Load images
  for (const auto &img : model.images) {
    if (img.component != 4) {
      std::cout << "WILL: Check non-4 component image support\n";
    }
    if (img.pixel_type != TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
      std::cout << "Non-uchar images are not supported\n";
      throw std::runtime_error("Unsupported image pixel type");
    }

    Image texture;
    texture.name = img.name;
    texture.width = img.width;
    texture.height = img.height;
    texture.channels = img.component;
    texture.img = img.image;
    // Assume linear unless we find it used as a color texture
    texture.color_space = LINEAR;
    textures.push_back(texture);
  }

  // Load materials
  for (const auto &m : model.materials) {
    DisneyMaterial mat;
    mat.base_color.x = m.pbrMetallicRoughness.baseColorFactor[0];
    mat.base_color.y = m.pbrMetallicRoughness.baseColorFactor[1];
    mat.base_color.z = m.pbrMetallicRoughness.baseColorFactor[2];

    mat.metallic = m.pbrMetallicRoughness.metallicFactor;

    mat.roughness = m.pbrMetallicRoughness.roughnessFactor;

    if (m.pbrMetallicRoughness.baseColorTexture.index != -1) {
      const int32_t id =
          model.textures[m.pbrMetallicRoughness.baseColorTexture.index].source;
      textures[id].color_space = SRGB;

      uint32_t tex_mask = TEXTURED_PARAM_MASK;
      SET_TEXTURE_ID(tex_mask, id);
      mat.base_color.r = *reinterpret_cast<float *>(&tex_mask);
    }
    // glTF: metallic is blue channel, roughness is green channel
    if (m.pbrMetallicRoughness.metallicRoughnessTexture.index != -1) {
      const int32_t id =
          model.textures[m.pbrMetallicRoughness.metallicRoughnessTexture.index]
              .source;
      textures[id].color_space = LINEAR;

      uint32_t tex_mask = TEXTURED_PARAM_MASK;
      SET_TEXTURE_ID(tex_mask, id);
      SET_TEXTURE_CHANNEL(tex_mask, 2);
      mat.metallic = *reinterpret_cast<float *>(&tex_mask);

      tex_mask = TEXTURED_PARAM_MASK;
      SET_TEXTURE_ID(tex_mask, id);
      SET_TEXTURE_CHANNEL(tex_mask, 1);
      mat.roughness = *reinterpret_cast<float *>(&tex_mask);
    }
    materials.push_back(mat);
  }

  for (const auto &nid : model.scenes[model.defaultScene].nodes) {
    const tinygltf::Node &n = model.nodes[nid];
    if (n.mesh != -1) {
      const glm::mat4 transform = read_node_transform(n);
      // Note: GLTF "mesh" == ChameleonRT "parameterized mesh", since materials
      // and meshes are combined in a single entity in GLTF
      instances.emplace_back(transform, n.mesh);
    }
  }

  validate_materials();

  // Does GLTF have lights in the file? If one is missing we should generate
  // one, otherwise we can load them
  std::cout << "Generating light for GLTF scene\n";
  // {
  //   QuadLight light;
  //   light.emission = glm::vec4(20.f);
  //   light.normal = glm::vec4(glm::normalize(glm::vec3(0, -0.8, 0)), 0);
  //   light.position = -8.f * light.normal;
  //   light.position.y *= 2.0f;
  //   ortho_basis(light.v_x, light.v_y, glm::vec3(light.normal));
  //   light.width = 4.f;
  //   light.height = 4.f;
  //   lights.push_back(light);
  // }

  {
    QuadLight light;
    light.emission = glm::vec4(80.f);
    light.normal = glm::vec4(glm::normalize(glm::vec3(0.0, -0.8, 0.0)), 0);
    light.position = -12.f * light.normal;
    // light.position.y *= 2.0f;
    ortho_basis(light.v_x, light.v_y, glm::vec3(light.normal));
    light.width = 4.f;
    light.height = 4.f;
    lights.push_back(light);
  }

  {
    QuadLight light;
    light.emission = glm::vec4(80.f);
    light.normal = glm::vec4(glm::normalize(glm::vec3(0.2, -0.8, -0.2)), 0);
    light.position = -8.f * light.normal;
    light.position.y *= 1.5f;
    ortho_basis(light.v_x, light.v_y, glm::vec3(light.normal));
    light.width = 4.f;
    light.height = 4.f;
    lights.push_back(light);
  }

  {
    QuadLight light;
    light.emission = glm::vec4(80.f);
    light.normal = glm::vec4(glm::normalize(glm::vec3(-0.5, -0.8, 0.0)), 0);
    light.position = -12.f * light.normal;
    light.position.y *= 1.5f;
    ortho_basis(light.v_x, light.v_y, glm::vec3(light.normal));
    light.width = 4.f;
    light.height = 4.f;
    lights.push_back(light);
  }
}

void Scene::validate_materials() {
  const bool need_default_mat =
      std::find_if(parameterized_meshes.begin(), parameterized_meshes.end(),
                   [](const ParameterizedMesh &i) {
                     return std::find(i.material_ids.begin(),
                                      i.material_ids.end(),
                                      uint32_t(-1)) != i.material_ids.end();
                   }) != parameterized_meshes.end();

  if (need_default_mat) {
    std::cout
        << "No materials assigned for some objects, generating a default\n";
    const uint32_t default_mat_id = materials.size();
    materials.push_back(DisneyMaterial());
    for (auto &i : parameterized_meshes) {
      for (auto &m : i.material_ids) {
        if (m == -1) {
          m = default_mat_id;
        }
      }
    }
  }
}

} // namespace UniRt