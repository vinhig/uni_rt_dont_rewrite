#include "render.h"

#include "glad/glad.h"
#include <SDL2/SDL.h>
#include <backends/imgui_impl_opengl3.h>
#include <backends/imgui_impl_sdl2.h>
#include <imgui.h>
#include <stdio.h>
#include <string>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <glm/ext.hpp>

#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include "embree_ispc.h"
#include "embree_utils.h"
#include "scene.h"
#include "util.h"

#include "denoisers/a-svgf.h"
#include "denoisers/accumulator.h"
#include "denoisers/bmfr.h"
#include "denoisers/bmfr_renew.h"
#include "denoisers/none.h"
#include "denoisers/oidn.h"

#include "stb_image.h"

const std::string fullscreen_quad_vs = R"(
#version 430 core

const vec4 pos[4] = vec4[4](
	vec4(-1, 1, 0.5, 1),
	vec4(-1, -1, 0.5, 1),
	vec4(1, 1, 0.5, 1),
	vec4(1, -1, 0.5, 1)
);

void main(void){
	gl_Position = pos[gl_VertexID];
}
)";

const std::string fullscreen_quad_fs = R"(
#version 430 core

layout(binding = 0) uniform sampler2D denoised;
layout(binding = 1) uniform sampler2D albedo;

out vec4 color;

void main(void){ 
	vec2 uv = vec2(gl_FragCoord.x, gl_FragCoord.y) / vec2(1280, 720);

	color = vec4(texture(denoised, uv, 0).xyz * texture(albedo, uv, 0).xyz, 1.0);

  // Apply gamma correction
  // color = pow(color, vec4(1.0/2.2));
})";

static void GLAPIENTRY OglDebugOutput(GLenum source, GLenum type, GLuint id,
                                      GLenum severity, GLsizei length,
                                      const GLchar *message,
                                      const void *userParam) {
  if (type == GL_DEBUG_TYPE_ERROR) {
    fprintf(stderr,
            "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
            (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""), type,
            severity, message);
  }
}

namespace UniRt {
GLuint CompileShader(const char *path, const char *source, GLenum shaderType) {
  // Compile shader
  GLuint shader = glCreateShader(shaderType);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);

  // Check result
  int result = GL_FALSE;
  int infoLength;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &result);
  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLength);
  if (infoLength > 0) {
    char *errorMsg = new char[infoLength + 1];
    glGetShaderInfoLog(shader, infoLength, nullptr, &errorMsg[0]);
    printf("%s\n", source);
    printf("%s -> %s\n", path, errorMsg);
    throw std::runtime_error(errorMsg);
  }
  // auto err = glGetError();
  // assert(err == GL_NO_ERROR);

  return shader;
}

Render::Render() {
  if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
    printf("oh no: %s\n", SDL_GetError());
    return;
  }

  SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS,
                      SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);

  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

  window =
      SDL_CreateWindow("UniRT", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                       1280, 720, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);

  auto gl_context = SDL_GL_CreateContext(window);
  if (gl_context == NULL) {
    printf("oh noooo: %s\n", SDL_GetError());
  }
  SDL_GL_MakeCurrent(window, gl_context);

  gladLoadGLLoader(SDL_GL_GetProcAddress);

  glEnable(GL_TEXTURE_3D);
  glEnable(GL_DEBUG_OUTPUT);
  glDisable(GL_CULL_FACE);
  glDebugMessageCallback(OglDebugOutput, nullptr);

  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;
  io.Fonts->AddFontFromFileTTF("../verdana.ttf", 18.0f, NULL, NULL);

  if (!ImGui_ImplSDL2_InitForOpenGL(window, gl_context)) {
    printf("sheeeesh\n");
  };
  ImGui_ImplOpenGL3_Init("#version 330");
  {
    std::ifstream features_vertex_file("../shaders/features.vert.glsl");
    std::ostringstream fv_ss;
    fv_ss << features_vertex_file.rdbuf();
    std::string features_vertex_source = fv_ss.str();

    std::ifstream features_fragment_file("../shaders/features.frag.glsl");
    std::ostringstream ff_ss;
    ff_ss << features_fragment_file.rdbuf();
    std::string features_fragment_source = ff_ss.str();

    GLuint features_fragment =
        CompileShader("../shader/features.frag.glsl",
                      features_fragment_source.c_str(), GL_FRAGMENT_SHADER);
    GLuint features_vertex =
        CompileShader("../shader/features.vert.glsl",
                      features_vertex_source.c_str(), GL_VERTEX_SHADER);

    features_program = glCreateProgram();
    glAttachShader(features_program, features_vertex);
    glAttachShader(features_program, features_fragment);
    glLinkProgram(features_program);

    glDeleteShader(features_vertex);
    glDeleteShader(features_fragment);
  }

  transform_location = glGetUniformLocation(features_program, "transform");
  view_proj_location = glGetUniformLocation(features_program, "view_proj");
  instance_id_location = glGetUniformLocation(features_program, "instance_id");

  {
    std::ifstream features_ta_file("../shaders/ta.comp.glsl");
    std::ostringstream ta_ss;
    ta_ss << features_ta_file.rdbuf();
    std::string features_ta_source = ta_ss.str();

    GLuint ta_compute =
        CompileShader("../shader/ta.comp.glsl", features_ta_source.c_str(),
                      GL_COMPUTE_SHADER);

    ta_program = glCreateProgram();
    glAttachShader(ta_program, ta_compute);
    glLinkProgram(ta_program);

    glDeleteShader(ta_compute);
  }

  {
    quad_program = glCreateProgram();
    GLuint quad_vertex =
        CompileShader("fullscreen quad vertex shader", &fullscreen_quad_vs[0],
                      GL_VERTEX_SHADER);
    GLuint quad_fragment =
        CompileShader("fullscreen quad fragment shader", &fullscreen_quad_fs[0],
                      GL_FRAGMENT_SHADER);

    glAttachShader(quad_program, quad_vertex);
    glAttachShader(quad_program, quad_fragment);
    glLinkProgram(quad_program);
  }

  glEnable(GL_DEPTH_TEST);

  // camera.angle = 25.0f;
  // camera.speed = 0.0f;
  // camera.distance = 24.0f;
  // camera.center = glm::vec3(0.0f, 3.5f, 0.0f);

  // camera.angle = 105.0f;
  // camera.speed = 0.0f;
  // camera.distance = 10.0f;
  // camera.center = glm::vec3(0.0f, 1.0f, 0.0f);

  camera.angle = 0.0f;
  camera.speed = 0.0f;
  camera.distance = 9.0f;
  camera.center = glm::vec3(0.0f, 1.0f, 0.0f);

  int yo = 0;
  glGetIntegerv(GL_MAX_LABEL_LENGTH, &yo);
  printf("GL_MAX_LABEL_LENGTH = %d;", yo);

  glGenFramebuffers(2, features_fbo);
  { glGenTextures(2, position_texture); }
  { glGenTextures(2, normal_texture); }
  { glGenTextures(2, geo_normal_texture); }
  { glGenTextures(2, depth_texture); }
  { glGenTextures(2, motion_texture); }
  glGenTextures(2, accumulated_noisy_texture);
  glGenTextures(2, accumulated_denoised_texture);
  glGenTextures(2, noisy_accumulation.moment_texture);
  glGenTextures(2, noisy_accumulation.history_length);
  glGenTextures(2, denoised_accumulation.moment_texture);
  glGenTextures(2, denoised_accumulation.history_length);
  glGenTextures(2, visibility_texture);
  { glGenTextures(2, rng_seed_texture); }

  for (int i = 0; i < 2; i++) {
    glBindTexture(GL_TEXTURE_2D, position_texture[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
                 nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, normal_texture[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
                 nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, geo_normal_texture[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
                 nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, motion_texture[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
                 nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, visibility_texture[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, 1280, 720, 0, GL_RED_INTEGER,
                 GL_INT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, rng_seed_texture[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32UI, 1280, 720, 0, GL_RGBA_INTEGER,
                 GL_INT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, depth_texture[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, 1280, 720, 0,
                 GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, accumulated_denoised_texture[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
                 nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, accumulated_noisy_texture[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
                 nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, noisy_accumulation.moment_texture[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
                 nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, noisy_accumulation.history_length[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8UI, 1280, 720, 0, GL_RED_INTEGER,
                 GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, denoised_accumulation.moment_texture[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
                 nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, denoised_accumulation.history_length[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8UI, 1280, 720, 0, GL_RED_INTEGER,
                 GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindFramebuffer(GL_FRAMEBUFFER, features_fbo[i]);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           position_texture[i], 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D,
                           normal_texture[i], 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D,
                           visibility_texture[i], 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D,
                           geo_normal_texture[i], 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_2D,
                           motion_texture[i], 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D,
                           depth_texture[i], 0);

    GLenum draw_buffers[] = {
        GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2,
        GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4,
    };
    glDrawBuffers(5, draw_buffers);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      printf("oh noooo\n");
    }

    glGenTextures(1, &shadow_texture[i]);
    glBindTexture(GL_TEXTURE_2D, shadow_texture[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
                 nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenTextures(1, &albedo_texture[i]);
    glBindTexture(GL_TEXTURE_2D, albedo_texture[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
                 nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  }

  camera.Update(16 / 1000.0);

  glGenBuffers(1, &reprojection_buffer);
  glBindBuffer(GL_UNIFORM_BUFFER, reprojection_buffer);
  glBufferData(GL_UNIFORM_BUFFER, sizeof(ReprojectionCB), &reprojection,
               GL_DYNAMIC_DRAW);

  SetupEmbree();

  current_denoiser = new Denoiser::BmfrDenoiser();

  glObjectLabel(GL_TEXTURE, position_texture[0], -1, "position_texture[0]");
  glObjectLabel(GL_TEXTURE, position_texture[1], -1, "position_texture[1]");

  glObjectLabel(GL_TEXTURE, normal_texture[0], -1, "normal_texture[0]");
  glObjectLabel(GL_TEXTURE, normal_texture[1], -1, "normal_texture[1]");

  glObjectLabel(GL_TEXTURE, geo_normal_texture[0], -1, "geo_normal_texture[0]");
  glObjectLabel(GL_TEXTURE, geo_normal_texture[1], -1, "geo_normal_texture[1]");

  glObjectLabel(GL_TEXTURE, depth_texture[0], -1, "depth_texture[0]");
  glObjectLabel(GL_TEXTURE, depth_texture[1], -1, "depth_texture[1]");

  glObjectLabel(GL_TEXTURE, motion_texture[0], -1, "motion_texture[0]");
  glObjectLabel(GL_TEXTURE, motion_texture[1], -1, "motion_texture[1]");

  glObjectLabel(GL_TEXTURE, rng_seed_texture[0], -1, "rng_seed_texture[0]");
  glObjectLabel(GL_TEXTURE, rng_seed_texture[1], -1, "rng_seed_texture[1]");

  bunch_of_textures.noisy_texture[0] = accumulated_noisy_texture[0];
  bunch_of_textures.noisy_texture[1] = accumulated_noisy_texture[1];
  bunch_of_textures.position_texture[0] = position_texture[0];
  bunch_of_textures.position_texture[1] = position_texture[1];
  bunch_of_textures.normal_texture[0] = normal_texture[0];
  bunch_of_textures.normal_texture[1] = normal_texture[1];
  bunch_of_textures.visibility_texture[0] = visibility_texture[0];
  bunch_of_textures.visibility_texture[1] = visibility_texture[1];
  bunch_of_textures.albedo_texture[0] = albedo_texture[0];
  bunch_of_textures.albedo_texture[1] = albedo_texture[1];
  bunch_of_textures.depth_texture[0] = depth_texture[0];
  bunch_of_textures.depth_texture[1] = depth_texture[1];
  bunch_of_textures.geo_normal_texture[0] = geo_normal_texture[0];
  bunch_of_textures.geo_normal_texture[1] = geo_normal_texture[1];
  bunch_of_textures.motion_texture[0] = motion_texture[0];
  bunch_of_textures.motion_texture[1] = motion_texture[1];
  bunch_of_textures.rng_seed_texture[0] = rng_seed_texture[0];
  bunch_of_textures.rng_seed_texture[1] = rng_seed_texture[1];

  bunch_of_textures.reprojection_buffer = reprojection_buffer;

  glGenQueries(1, &start_time_query);
  glGenQueries(1, &stop_time_query);
}

void Render::SetupEmbree() {
  device = rtcNewDevice(nullptr);

  img_color.resize(1280 * 720 * 4);
  img_shadow.resize(1280 * 720 * 4);
  img_albedo.resize(1280 * 720 * 4);

  const glm::uvec2 ntiles(1280 / tile_size.x +
                              (1280 % tile_size.x != 0 ? 1 : 0),
                          720 / tile_size.y + (720 % tile_size.y != 0 ? 1 : 0));

  tiles_color.resize(ntiles.x * ntiles.y);
  tiles_shadow.resize(ntiles.x * ntiles.y);
  tiles_albedo.resize(ntiles.x * ntiles.y);

  position_texture_pixels.resize(1280 * 720 * 4, 0.0f);
  normal_texture_pixels.resize(1280 * 720 * 4, 0.0f);
  depth_texture_pixels.resize(1280 * 720 * 1, 0.0f);

  rng_seed_texture_pixels[0].resize(1280 * 720 * 4, 0.0f);
  rng_seed_texture_pixels[1].resize(1280 * 720 * 4, 0.0f);

  for (int i = 0; i < tiles_color.size(); i++) {
    tiles_color[i].resize(tile_size.x * tile_size.y * 3, 0.f);
    tiles_shadow[i].resize(tile_size.x * tile_size.y * 3, 0.f);
    tiles_albedo[i].resize(tile_size.x * tile_size.y * 3, 0.f);
  }
}

void Render::SetSceneEmbree(UniRt::Scene *scene) {
  std::vector<std::shared_ptr<embree::TriangleMesh>> meshes;
  for (const auto &mesh : scene->meshes) {
    std::vector<std::shared_ptr<embree::Geometry>> geometries;
    for (const auto &geom : mesh.geometries) {
      geometries.push_back(std::make_shared<embree::Geometry>(
          device, geom.vertices, geom.indices, geom.normals, geom.uvs));
    }

    meshes.push_back(
        std::make_shared<embree::TriangleMesh>(device, geometries));
  }

  parameterized_meshes = scene->parameterized_meshes;

  std::vector<std::shared_ptr<embree::Instance>> instances;
  for (const auto &inst : scene->instances) {
    const auto &pm = parameterized_meshes[inst.parameterized_mesh_id];
    instances.push_back(std::make_shared<embree::Instance>(
        device, meshes[pm.mesh_id], inst.transform, pm.material_ids));
  }

  scene_bvh = std::make_shared<embree::TopLevelBVH>(device, instances);

  textures = scene->textures;

  // Linearize any sRGB textures beforehand, since we don't have fancy sRGB
  // texture interpolation support in hardware
  tbb::parallel_for(size_t(0), textures.size(), [&](size_t i) {
    auto &img = textures[i];
    if (img.color_space == LINEAR) {
      return;
    }
    img.color_space = LINEAR;
    const int convert_channels = std::min(3, img.channels);
    tbb::parallel_for(
        size_t(0), size_t(img.width) * img.height, [&](size_t px) {
          for (int c = 0; c < convert_channels; ++c) {
            float x = img.img[px * img.channels + c] / 255.f;
            x = srgb_to_linear(x);
            img.img[px * img.channels + c] = glm::clamp(x * 255.f, 0.f, 255.f);
          }
        });
  });

  ispc_textures.reserve(textures.size());
  std::transform(textures.begin(), textures.end(),
                 std::back_inserter(ispc_textures),
                 [](const Image &img) { return embree::ISPCTexture2D(img); });

  material_params.reserve(scene->materials.size());
  for (const auto &m : scene->materials) {
    embree::MaterialParams p;

    p.base_color = m.base_color;
    p.metallic = m.metallic;
    p.specular = m.specular;
    p.roughness = m.roughness;
    p.specular_tint = m.specular_tint;
    p.anisotropy = m.anisotropy;
    p.sheen = m.sheen;
    p.sheen_tint = m.sheen_tint;
    p.clearcoat = m.clearcoat;
    p.clearcoat_gloss = m.clearcoat_gloss;
    p.ior = m.ior;
    p.specular_transmission = m.specular_transmission;

    material_params.push_back(p);
  }

  lights = scene->lights;

  for (int i = 0; i < 128; i++) {
    int w, h, n;
    char buf[1024];

    snprintf(buf, sizeof buf,
             "../textures/blue_noise/256_256/HDR_RGBA_%04d.png", i);

    stbi_uc *img_data = stbi_load(buf, &w, &h, &n, 4);

    if (img_data == NULL) {
      printf("Couldn't load blue noise texture '%s'\n", buf);
    }

    blue_noise_texture_ispc[i].width = w;
    blue_noise_texture_ispc[i].height = h;
    blue_noise_texture_ispc[i].channels = n;
    blue_noise_texture_ispc[i].data =
        static_cast<uint8_t *>(malloc(sizeof(unsigned char) * w * h * n));

    memcpy((void *)blue_noise_texture_ispc[0].data, img_data,
           256 * 256 * 4 * sizeof(unsigned char));

    stbi_image_free(img_data);
  }
}

void Render::SetSceneOpenGL(Scene *scene) {

  std::vector<Instance> gpu_instances;

  for (auto &instance : scene->instances) {
    auto transform = instance.transform;
    auto &geometries = scene->meshes[instance.parameterized_mesh_id].geometries;

    std::vector<Geometry> gpu_geometries;

    for (auto &geometry : geometries) {
      GLuint indices;
      GLuint vertices;
      glGenBuffers(1, &indices);
      glGenBuffers(1, &vertices);

      auto indices_size = geometry.indices.size() * 3;
      unsigned *indices_data = new unsigned[indices_size];

      for (int i = 0; i < geometry.indices.size(); i++) {
        indices_data[i * 3] = geometry.indices[i].x;
        indices_data[i * 3 + 1] = geometry.indices[i].y;
        indices_data[i * 3 + 2] = geometry.indices[i].z;
      }

      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indices);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_size * sizeof(unsigned),
                   indices_data, GL_STATIC_DRAW);

      auto vertices_size = geometry.vertices.size() * (3 + 3 + 2);
      float *interleaved_vertices = new float[vertices_size];

      for (int i = 0; i < geometry.vertices.size(); i++) {
        interleaved_vertices[i * 8 + 0] = geometry.vertices[i].x;
        interleaved_vertices[i * 8 + 1] = geometry.vertices[i].y;
        interleaved_vertices[i * 8 + 2] = geometry.vertices[i].z;
        interleaved_vertices[i * 8 + 3] = geometry.normals[i].x;
        interleaved_vertices[i * 8 + 4] = geometry.normals[i].y;
        interleaved_vertices[i * 8 + 5] = geometry.normals[i].z;
        interleaved_vertices[i * 8 + 6] = geometry.uvs[i].x;
        interleaved_vertices[i * 8 + 7] = geometry.uvs[i].y;
      }

      glBindBuffer(GL_ARRAY_BUFFER, vertices);
      glBufferData(GL_ARRAY_BUFFER, vertices_size * sizeof(float),
                   interleaved_vertices, GL_STATIC_DRAW);

      GLuint this_vao;

      glGenVertexArrays(1, &this_vao);
      glBindVertexArray(this_vao);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indices);
      glBindBuffer(GL_ARRAY_BUFFER, vertices);

      glEnableVertexAttribArray(0); // position
      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                            sizeof(float) * (3 + 3 + 2), 0);

      glEnableVertexAttribArray(1); // normal
      glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
                            sizeof(float) * (3 + 3 + 2),
                            (void *)(sizeof(float) * 3));

      glEnableVertexAttribArray(2);
      glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE,
                            sizeof(float) * (3 + 3 + 2),
                            (void *)(sizeof(float) * 3 * 2));
      glBindVertexArray(0);

      Geometry hey;
      hey.index_buffer = indices;
      hey.vertex_buffer = vertices;
      hey.vao = this_vao;
      hey.triangle_count = geometry.num_tris();

      gpu_geometries.push_back(hey);

      delete indices_data;
      delete interleaved_vertices;
    }

    gpu_instances.push_back(Instance{
        .transform = transform,
        .geometries = gpu_geometries,
    });
  }

  instances = gpu_instances;
}

void Render::DrawFeatureBuffers() {
  glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1,
                   "RENDER DRAW FEATURE BUFFERS");

  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, features_fbo[current_frame % 2]);
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

  glUseProgram(features_program);
  glBindBufferBase(GL_UNIFORM_BUFFER, 0, reprojection_buffer);
  int instance_id = 0;
  for (auto &instance : instances) {
    glUniformMatrix4fv(transform_location, 1, GL_FALSE,
                       &instance.transform[0][0]);
    // glUniformMatrix4fv(view_proj_location, 1, GL_FALSE,
    //                    &camera.view_proj[0][0]);

    for (auto &geometry : instance.geometries) {
      glUniform1i(instance_id_location, instance_id);
      glBindVertexArray(geometry.vao);
      // glBindBuffer(GL_ARRAY_BUFFER, geometry.vertex_buffer);
      // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, geometry.index_buffer);
      glDrawElements(GL_TRIANGLES, geometry.triangle_count * 3, GL_UNSIGNED_INT,
                     NULL);
      instance_id += 1;
    }
  }

  glPopDebugGroup();
}

bool Render::UpdateSDL() {
  bool quit = false;
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    ImGui_ImplSDL2_ProcessEvent(&event);
    if (event.type == SDL_QUIT) {
      quit = true;
    }
  }

  return quit;
}

void Render::DrawGUI() {
  static int chosen = 0;
  static char *denoisers[] = {
      "bmfr", "bmfr_renew", "a-svgf", "oidn", "none", "accum",
  };

  glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "RENDER DRAW GUI");

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplSDL2_NewFrame(window);
  ImGui::NewFrame();

  ImGui::SetNextWindowSize(ImVec2(450.0, 450.0), ImGuiCond_Always);

  ImGui::Begin("My awesome memory thesis", NULL, ImGuiCond_Always);
  ImGui::Text("Platform: %s", SDL_GetPlatform());
  ImGui::Text("Current scene: %s", current_scene_name.c_str());
  ImGui::Text("Current frame: %d", current_frame);

  GLuint64 start_time;
  GLuint64 stop_time;
  glGetQueryObjectui64v(start_time_query, GL_QUERY_RESULT, &start_time);
  glGetQueryObjectui64v(stop_time_query, GL_QUERY_RESULT, &stop_time);

  float real_start_time = (float)((double)start_time / 1000000.0);
  float real_stop_time = (float)((double)stop_time / 1000000.0);

  std::ofstream benchmark_file;

  benchmark_file.open(std::string(denoisers[chosen]) + "_benchmark.csv",
                      std::ios_base::app);
  benchmark_file << real_stop_time - real_start_time << std::endl;
  benchmark_file.close();

  ImGui::Text("Denoise time: %f", real_stop_time - real_start_time);

  ImGui::SliderFloat("Camera.far", &camera.far, 10.0f, 320.0f);
  ImGui::SliderFloat("Camera.angle", &camera.angle, -360.0f, 360.0f);
  ImGui::SliderFloat("Camera.near", &camera.near, 0.01, 1.0f);

  ImGui::SliderFloat("Camera.speed", &camera.speed, -14.0f, 14.0f);

  if (ImGui::Button("Reset Camera.speed")) {
    camera.speed = 0.0;
  }

  if (demo_mode &&
      (chosen != 5 || (chosen == 5 && current_frame % 256 == 255))) {
    struct stat st = {0};

    char folder_0[1024];
    sprintf(folder_0, "./movies/");
    char folder_1[1024];
    sprintf(folder_1, "./movies/%s/", denoisers[chosen]);
    char folder_2[1024];
    sprintf(folder_2, "./movies/%s/%s/", denoisers[chosen],
            current_scene_name.c_str());

    mkdir(folder_0, 0700);
    mkdir(folder_1, 0700);
    mkdir(folder_2, 0700);

    char denoised_path[1024];
    sprintf(denoised_path, "./movies/%s/%s/denoised_blob_%d.buff",
            denoisers[chosen], current_scene_name.c_str(), current_frame);

    char noisy_path[1024];
    sprintf(noisy_path, "./movies/%s/%s/noisy_blob_%d.buff", denoisers[chosen],
            current_scene_name.c_str(), current_frame);

    if (chosen == 5) {
      ((Denoiser::AccumulatorDenoiser *)current_denoiser)->IncreaseOffset(256);
    }

    Screenshot(&denoised_path[0], &noisy_path[0]);
  } else {
    if (ImGui::Button("Screenshot") || (current_frame == 1024 && chosen == 5) ||
        (current_frame == 30 && chosen != 5)) {
      char denoised_path[1024];
      sprintf(denoised_path, "screenshot_blob_%d_%s.buff", current_frame,
              denoisers[chosen]);

      char noisy_path[1024];
      sprintf(denoised_path, "noisy_blob.buff", current_frame,
              denoisers[chosen]);

      Screenshot(denoised_path, noisy_path);
    }
  }
  ImGui::Checkbox("Temporal Accumulation", &temporal_accumulation);

  if (temporal_accumulation) {
    ImGui::SliderFloat("Repro.phi_depth", &reprojection.phi_depth, 0.01f, 0.8f);
    ImGui::SliderFloat("Repro.phi_normal", &reprojection.phi_normal, 0.01f,
                       0.8f);

    ImGui::SliderFloat("Repro.depth_tolerance", &reprojection.depth_tolerance,
                       0.01f, 0.9f);
    ImGui::SliderFloat("Repro.normal_tolerance", &reprojection.normal_tolerance,
                       0.01f, 0.9f);

    ImGui::SliderFloat("Repro.min_accum_weight", &reprojection.min_accum_weight,
                       0.01f, 0.9f);

    ImGui::SliderFloat("Gradient.gradient_cap", &reprojection.gradient_cap,
                       0.0f, 1.0f);
  }

  if (ImGui::BeginCombo("Denoiser", denoisers[chosen])) {
    {
      for (auto i = 0; i < 5; i++) {
        if (ImGui::Selectable(denoisers[i], i == chosen)) {
          chosen = i;
          switch (chosen) {
          case 0: {
            delete current_denoiser;
            current_denoiser = new Denoiser::BmfrDenoiser();
            break;
          }
          case 1: {
            delete current_denoiser;
            current_denoiser = new Denoiser::BmfrRenewDenoiser();
            break;
          }
          case 2: {
            delete current_denoiser;
            current_denoiser = new Denoiser::ASvgfDenoiser();
            break;
          }
          case 3: {
            delete current_denoiser;
            current_denoiser = new Denoiser::OidnDenoiser();
            break;
          }
          case 4: {
            delete current_denoiser;
            current_denoiser = new Denoiser::NoneDenoiser();
            break;
          }
          case 5: {
            delete current_denoiser;
            current_denoiser = new Denoiser::AccumulatorDenoiser();
            break;
          }
          default: {
            break;
          }
          }
        }
      }
    }
    ImGui::EndCombo();
  }
  {
    ImVec2 wsize;
    wsize.x = 1280 / 5 * 1.9;
    wsize.y = 720 / 5 * 1.9;
    ImGui::Image((ImTextureID)albedo_texture[current_frame % 2], wsize,
                 ImVec2(0, 1), ImVec2(1, 0));
  }
  {
    ImVec2 wsize;
    wsize.x = 1280 / 5 * 1.9;
    wsize.y = 720 / 5 * 1.9;
    ImGui::Image((ImTextureID)shadow_texture[current_frame % 2], wsize,
                 ImVec2(0, 1), ImVec2(1, 0));
  }
  {
    ImVec2 wsize;
    wsize.x = 1280 / 5 * 1.9;
    wsize.y = 720 / 5 * 1.9;
    ImGui::Image((ImTextureID)position_texture[current_frame % 2], wsize,
                 ImVec2(0, 1), ImVec2(1, 0));
  }
  {
    ImVec2 wsize;
    wsize.x = 1280 / 5 * 1.9;
    wsize.y = 720 / 5 * 1.9;
    ImGui::Image((ImTextureID)normal_texture[current_frame % 2], wsize,
                 ImVec2(0, 1), ImVec2(1, 0));
  }
  // {
  //   ImVec2 wsize;
  //   wsize.x = 1280 / 5;
  //   wsize.y = 720 / 5;
  //   ImGui::Image((ImTextureID)depth_texture, wsize, ImVec2(0, 1), ImVec2(1,
  //   0));
  // }
  ImGui::End();

  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

  glPopDebugGroup();
}

void Render::DrawDenoise() {
  denoised_texture[current_frame % 2] =
      current_denoiser->Denoise(bunch_of_textures, current_frame);
}

void Render::DrawEmbree() {
  embree::SceneContext ispc_scene;
  ispc_scene.scene = scene_bvh->handle;
  ispc_scene.instances = scene_bvh->ispc_instances.data();
  ispc_scene.materials = material_params.data();
  ispc_scene.textures = ispc_textures.data();
  ispc_scene.lights = lights.data();
  ispc_scene.num_lights = lights.size();

  const glm::uvec2 ntiles(1280 / tile_size.x +
                              (1280 % tile_size.x != 0 ? 1 : 0),
                          720 / tile_size.y + (720 % tile_size.y != 0 ? 1 : 0));
  auto start = std::chrono::high_resolution_clock::now();
  tbb::parallel_for(uint32_t(0), ntiles.x * ntiles.y, [&](uint32_t tile_id) {
    const glm::uvec2 tile = glm::uvec2(tile_id % ntiles.x, tile_id / ntiles.x);
    const glm::uvec2 tile_pos = tile * tile_size;
    const glm::uvec2 tile_end =
        glm::min(tile_pos + tile_size, glm::uvec2(1280, 720));
    const glm::uvec2 actual_tile_dims = tile_end - tile_pos;

    embree::Tile ispc_tile;
    ispc_tile.x = tile_pos.x;
    ispc_tile.y = tile_pos.y;
    ispc_tile.width = actual_tile_dims.x;
    ispc_tile.height = actual_tile_dims.y;
    ispc_tile.fb_width = 1280;
    ispc_tile.fb_height = 720;
    ispc_tile.color = tiles_color[tile_id].data();
    ispc_tile.shadow = tiles_shadow[tile_id].data();
    ispc_tile.albedo = tiles_albedo[tile_id].data();
    ispc_tile.position = position_texture_pixels.data();
    ispc_tile.depth = depth_texture_pixels.data();
    ispc_tile.normal = normal_texture_pixels.data();
    ispc_tile.curr_rng_seed = rng_seed_texture_pixels[current_frame % 2].data();
    ispc_tile.prev_rng_seed =
        rng_seed_texture_pixels[1 - current_frame % 2].data();
    ispc_tile.camera_x = camera.offset.x;
    ispc_tile.camera_y = camera.offset.y;
    ispc_tile.camera_z = camera.offset.z;
    ispc_tile.frame_id = current_frame;
    ispc_tile.blue_noise = &blue_noise_texture_ispc[0];

    ispc::trace_rays(&ispc_scene, &ispc_tile);

    ispc::tile_to_buffer(&ispc_tile, img_shadow.data(), img_albedo.data());
#ifdef REPORT_RAY_STATS
    num_rays[tile_id] = std::accumulate(
        ray_stats[tile_id].begin(), ray_stats[tile_id].end(), uint64_t(0),
        [](const uint64_t &total, const uint16_t &c) { return total + c; });
#endif
  });
  auto end = std::chrono::high_resolution_clock::now();
}

bool Render::Update() {

  if (demo_mode) {
    camera.speed = 19.0;
    // camera.
    // camera.distance = cosf(glm::radians((float)current_frame));
    // camera.center.z += 0.02;
    // if ((current_frame % 256) == 0) {
    //   camera.Update(16 / 1000.0);
    //   float black[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    //   glClearTexImage(denoised_texture[current_frame % 2], 0, GL_RGBA, GL_FLOAT,
    //                   &black[0]);
    //   glClearTexImage(denoised_texture[1 - current_frame % 2], 0, GL_RGBA,
    //                   GL_FLOAT, &black[0]);
    // }
    camera.Update(16 / 1000.0);
  } else {
    camera.Update(16 / 1000.0);
  }

  bool quit = UpdateSDL();

  for (unsigned i = 0; i < 4; i++) {
    for (unsigned j = 0; j < 4; j++) {
      reprojection.prev_view_proj[i][j] = camera.prev_view_proj[i][j];
    }
  }

  auto inv_view_proj = glm::inverse(camera.view_proj);
  for (unsigned i = 0; i < 4; i++) {
    for (unsigned j = 0; j < 4; j++) {
      reprojection.inv_view_proj[i][j] = inv_view_proj[i][j];
    }
  }

  for (unsigned i = 0; i < 4; i++) {
    for (unsigned j = 0; j < 4; j++) {
      reprojection.view_proj[i][j] = camera.view_proj[i][j];
    }
  }

  for (unsigned i = 0; i < 4; i++) {
    for (unsigned j = 0; j < 4; j++) {
      reprojection.proj[i][j] = camera.proj[i][j];
    }
  }

  for (unsigned i = 0; i < 3; i++) {
    reprojection.view_pos[i] = camera.offset[i];
  }
  reprojection.view_pos[3] = 1.0f;
  reprojection.frame_number = current_frame;

  glBindBuffer(GL_UNIFORM_BUFFER, reprojection_buffer);
  glBufferData(GL_UNIFORM_BUFFER, sizeof(ReprojectionCB), &reprojection,
               GL_DYNAMIC_DRAW);

  glMemoryBarrier(GL_ALL_BARRIER_BITS);

  DrawFeatureBuffers();

  glMemoryBarrier(GL_ALL_BARRIER_BITS);

  current_denoiser->ReprojectSeed(bunch_of_textures, current_frame);

  glMemoryBarrier(GL_ALL_BARRIER_BITS);

  glBindTexture(GL_TEXTURE_2D, position_texture[current_frame % 2]);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT,
                position_texture_pixels.data());

  glBindTexture(GL_TEXTURE_2D, depth_texture[current_frame % 2]);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
                depth_texture_pixels.data());
  float ma = 0.0;

  // for (int i = 0; i < position_texture_pixels.size(); i++) {
  //   ma = std::max(ma, position_texture_pixels[i]);
  // }

  // printf("max => %f, current_frame => %d\n", ma, current_frame);
  // // assert(false);

  glBindTexture(GL_TEXTURE_2D, normal_texture[current_frame % 2]);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT,
                normal_texture_pixels.data());

  glBindTexture(GL_TEXTURE_2D, rng_seed_texture[current_frame % 2]);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT,
                rng_seed_texture_pixels[current_frame % 2].data());

  glMemoryBarrier(GL_ALL_BARRIER_BITS);

  DrawEmbree();

  glBindTexture(GL_TEXTURE_2D, shadow_texture[current_frame % 2]);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
               img_shadow.data());

  glBindTexture(GL_TEXTURE_2D, albedo_texture[current_frame % 2]);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
               img_albedo.data());

  glBindTexture(GL_TEXTURE_2D, normal_texture[current_frame % 2]);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
               normal_texture_pixels.data());

  glBindTexture(GL_TEXTURE_2D, depth_texture[current_frame % 2]);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, 1280, 720, 0,
               GL_DEPTH_COMPONENT, GL_FLOAT, depth_texture_pixels.data());

  glBindTexture(GL_TEXTURE_2D, rng_seed_texture[1 - current_frame % 2]);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32UI, 1280, 720, 0, GL_RGBA_INTEGER,
               GL_INT, rng_seed_texture_pixels[1 - current_frame % 2].data());

  glFinish();
  glMemoryBarrier(GL_ALL_BARRIER_BITS);

  if (temporal_accumulation &&
      current_denoiser->NeedPreTemporalAccumulation()) {
    TemporalAccumulationNoisy();
  } else {
    glCopyImageSubData(shadow_texture[current_frame % 2], GL_TEXTURE_2D, 0, 0,
                       0, 0, accumulated_noisy_texture[current_frame % 2],
                       GL_TEXTURE_2D, 0, 0, 0, 0, 1280, 720, 1);

    glMemoryBarrier(GL_ALL_BARRIER_BITS);
  }

  glQueryCounter(start_time_query, GL_TIMESTAMP);
  DrawDenoise();
  glQueryCounter(stop_time_query, GL_TIMESTAMP);

  if (current_denoiser->NeedPostTemporalAccumulation()) {
    TemporalAccumulationDenoised();
  } else {
    glCopyImageSubData(denoised_texture[current_frame % 2], GL_TEXTURE_2D, 0, 0,
                       0, 0, accumulated_denoised_texture[current_frame % 2],
                       GL_TEXTURE_2D, 0, 0, 0, 0, 1280, 720, 1);

    glMemoryBarrier(GL_ALL_BARRIER_BITS);
  }

  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
  glClearColor(1.0, 0.9, 0.8, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glUseProgram(quad_program);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, accumulated_denoised_texture[current_frame % 2]);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, albedo_texture[current_frame % 2]);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  DrawGUI();

  SDL_GL_SwapWindow(window);

  current_frame += 1;

  return quit;
}

Render::~Render() {
  rtcReleaseDevice(device);
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImGui::DestroyContext();
  SDL_DestroyWindow(window);
  SDL_Quit();
}

void Render::SetScene(UniRt::Scene *scene, std::string scene_name) {
  SetSceneEmbree(scene);
  SetSceneOpenGL(scene);

  current_scene_name = scene_name;
}

void Render::TemporalAccumulationNoisy() {
  glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1,
                   "RENDER TEMPORAL ACCUMULATION NOISY");
  glUseProgram(ta_program);

  // t_curr_normal
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, normal_texture[current_frame % 2]);
  // t_prev_normal
  glActiveTexture(GL_TEXTURE3);
  glBindTexture(GL_TEXTURE_2D, normal_texture[1 - current_frame % 2]);

  // t_prev_moments
  glActiveTexture(GL_TEXTURE4);
  glBindTexture(GL_TEXTURE_2D,
                noisy_accumulation.moment_texture[1 - current_frame % 2]);

  // t_curr_indirect
  glActiveTexture(GL_TEXTURE5);
  glBindTexture(GL_TEXTURE_2D, shadow_texture[current_frame % 2]);

  // t_prev_accumulated
  glActiveTexture(GL_TEXTURE6);
  glBindTexture(GL_TEXTURE_2D,
                accumulated_noisy_texture[1 - current_frame % 2]);

  // t_out_accumulated
  glBindImageTexture(7, accumulated_noisy_texture[current_frame % 2], 0, 0, 0,
                     GL_READ_WRITE, GL_RGBA32F);

  // t_out_moments
  glBindImageTexture(8, noisy_accumulation.moment_texture[current_frame % 2], 0,
                     0, 0, GL_READ_WRITE, GL_RG16F);

  glBindImageTexture(9, noisy_accumulation.history_length[0], 0, 0, 0,
                     GL_READ_WRITE, GL_R8UI);

  // t_curr_depth
  glActiveTexture(GL_TEXTURE11);
  glBindTexture(GL_TEXTURE_2D, depth_texture[current_frame % 2]);
  // t_prev_depth
  glActiveTexture(GL_TEXTURE12);
  glBindTexture(GL_TEXTURE_2D, depth_texture[1 - current_frame % 2]);

  glActiveTexture(GL_TEXTURE13);
  glBindTexture(GL_TEXTURE_2D, motion_texture[current_frame % 2]);

  glBindBufferBase(GL_UNIFORM_BUFFER, 0, reprojection_buffer);

  glDispatchCompute(1280 / 8, 720 / 8, 1);

  glMemoryBarrier(GL_ALL_BARRIER_BITS);

  glPopDebugGroup();
}

void Render::TemporalAccumulationDenoised() {
  glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1,
                   "RENDER TEMPORAL ACCUMULATION DENOISED");

  glUseProgram(ta_program);

  // t_curr_normal
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, normal_texture[current_frame % 2]);
  // t_prev_normal
  glActiveTexture(GL_TEXTURE3);
  glBindTexture(GL_TEXTURE_2D, normal_texture[1 - current_frame % 2]);

  // t_prev_moments
  glActiveTexture(GL_TEXTURE4);
  glBindTexture(GL_TEXTURE_2D,
                denoised_accumulation.moment_texture[1 - current_frame % 2]);

  // t_curr_indirect
  glActiveTexture(GL_TEXTURE5);
  glBindTexture(GL_TEXTURE_2D, denoised_texture[current_frame % 2]);

  // t_prev_accumulated
  glActiveTexture(GL_TEXTURE6);
  glBindTexture(GL_TEXTURE_2D,
                accumulated_denoised_texture[1 - current_frame % 2]);

  // t_out_accumulated
  glBindImageTexture(7, accumulated_denoised_texture[current_frame % 2], 0, 0,
                     0, GL_READ_WRITE, GL_RGBA32F);

  // t_out_moments
  glBindImageTexture(8, denoised_accumulation.moment_texture[current_frame % 2],
                     0, 0, 0, GL_READ_WRITE, GL_RG16F);

  // t_out_history_length
  glBindImageTexture(9, denoised_accumulation.history_length[current_frame % 2],
                     0, 0, 0, GL_READ_WRITE, GL_R8UI);

  // t_curr_depth
  glActiveTexture(GL_TEXTURE11);
  glBindTexture(GL_TEXTURE_2D, depth_texture[current_frame % 2]);

  // t_prev_depth
  glActiveTexture(GL_TEXTURE12);
  glBindTexture(GL_TEXTURE_2D, depth_texture[1 - current_frame % 2]);

  glActiveTexture(GL_TEXTURE13);
  glBindTexture(GL_TEXTURE_2D, motion_texture[current_frame % 2]);

  glBindBufferBase(GL_UNIFORM_BUFFER, 0, reprojection_buffer);

  glDispatchCompute(1280 / 8, 720 / 8, 1);

  glMemoryBarrier(GL_ALL_BARRIER_BITS);

  glPopDebugGroup();
}

void Render::Screenshot(char *denoised_path, char *noisy_path) {
  float *denoised = new float[1280 * 720 * 4];
  glBindTexture(GL_TEXTURE_2D, accumulated_denoised_texture[current_frame % 2]);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, denoised);

  float *noisy = new float[1280 * 720 * 4];
  glBindTexture(GL_TEXTURE_2D, shadow_texture[current_frame % 2]);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, noisy);

  float *albedo = new float[1280 * 720 * 4];
  glBindTexture(GL_TEXTURE_2D, albedo_texture[current_frame % 2]);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, albedo);

  for (int u = 0; u < 1280 * 720 * 4; u++) {
    denoised[u] *= albedo[u];
    noisy[u] *= albedo[u];
  }
  {
    FILE *f = fopen(denoised_path, "wb");

    if (!f) {
      printf("Couldn't save screenshot to %s...\n", denoised_path);
    } else {
      fwrite(denoised, 1280 * 720 * 4 * sizeof(float), 1, f);

      fclose(f);
    }
  }
  // {
  //   FILE *f = fopen(noisy_path, "wb");

  //   if (!f) {
  //     printf("Couldn't save screenshot to %s...\n", noisy_path);
  //   } else {
  //     fwrite(noisy, 1280 * 720 * 4 * sizeof(float), 1, f);
  //     fclose(f);
  //   }
  // }

  delete denoised;
  delete noisy;
  delete albedo;
}

} // namespace UniRt
