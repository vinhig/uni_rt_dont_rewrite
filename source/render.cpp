#include "render.h"

#include "glad/glad.h"
#include <SDL2/SDL.h>
#include <backends/imgui_impl_opengl3.h>
#include <backends/imgui_impl_sdl2.h>
#include <imgui.h>
#include <stdio.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include "scene.h"

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
  auto err = glGetError();
  assert(err == GL_NO_ERROR);

  return shader;
}

Render::Render() {
  printf("hellooooo\n");
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

  glEnable(GL_DEBUG_OUTPUT);
  glDebugMessageCallbackARB(OglDebugOutput, nullptr);

  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;
  io.Fonts->AddFontFromFileTTF("../verdana.ttf", 18.0f, NULL, NULL);

  if (!ImGui_ImplSDL2_InitForOpenGL(window, gl_context)) {
    printf("sheeeesh\n");
  };
  ImGui_ImplOpenGL3_Init("#version 330");

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

  transform_location = glGetUniformLocation(features_program, "transform");
  view_proj_location = glGetUniformLocation(features_program, "view_proj");

  glEnable(GL_DEPTH_TEST);

  camera.speed = 1.0;
  camera.angle = 0.0;
  camera.distance = 20.0;

  glGenFramebuffers(1, &features_fbo);
  glGenTextures(1, &position_texture);
  glGenTextures(1, &normal_texture);
  glGenTextures(1, &depth_texture);

  glBindTexture(GL_TEXTURE_2D, position_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
               nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glBindTexture(GL_TEXTURE_2D, normal_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1280, 720, 0, GL_RGBA, GL_FLOAT,
               nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glBindTexture(GL_TEXTURE_2D, depth_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, 1280, 720, 0,
               GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glBindFramebuffer(GL_FRAMEBUFFER, features_fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         position_texture, 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D,
                         normal_texture, 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D,
                         depth_texture, 0);

  GLenum draw_buffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
  glDrawBuffers(2, draw_buffers);

  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    printf("oh noooo\n");
  }
}

bool Render::Update() {
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, features_fbo);
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

  camera.Update(16 / 1000.0);

  glUseProgram(features_program);
  for (auto &instance : instances) {
    glUniformMatrix4fv(transform_location, 1, GL_FALSE,
                       &instance.transform[0][0]);
    glUniformMatrix4fv(view_proj_location, 1, GL_FALSE,
                       &camera.view_proj[0][0]);

    for (auto &geometry : instance.geometries) {
      glBindVertexArray(geometry.vao);
      // glBindBuffer(GL_ARRAY_BUFFER, geometry.vertex_buffer);
      // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, geometry.index_buffer);
      glDrawElements(GL_TRIANGLES, geometry.triangle_count * 3, GL_UNSIGNED_INT,
                     NULL);
    }
  }

  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
  glClearColor(1.0, 0.9, 0.8, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

  bool quit = false;
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    ImGui_ImplSDL2_ProcessEvent(&event);
    if (event.type == SDL_QUIT) {
      quit = true;
    }
  }

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplSDL2_NewFrame(window);
  ImGui::NewFrame();

  ImGui::SetNextWindowSize(ImVec2(640.0 / 3 * 2, 480), ImGuiCond_Always);

  ImGui::Begin("My awesome memory thesis", NULL, NULL);
  ImGui::Text("Platform: %s", SDL_GetPlatform());

  // a list of options in imgui containing "bmfr", "none", "optix"
  static char *denoisers[] = {
      "bmfr",
      "svgf",
      "optix",
      "none",
  };
  static int chosen = 0;
  if (ImGui::BeginCombo("Denoiser", denoisers[chosen])) {
    {
      for (auto i = 0; i < 4; i++) {
        if (ImGui::Selectable(denoisers[i], i == chosen)) {
          chosen = i;
          printf("HELLO\n");
        }
      }
    }
    ImGui::EndCombo();
  }
  {
    ImVec2 wsize;
    wsize.x = 1280 / 5;
    wsize.y = 720 / 5;
    ImGui::Image((ImTextureID)position_texture, wsize, ImVec2(0, 1),
                 ImVec2(1, 0));
  }
  {
    ImVec2 wsize;
    wsize.x = 1280 / 5;
    wsize.y = 720 / 5;
    ImGui::Image((ImTextureID)normal_texture, wsize, ImVec2(0, 1),
                 ImVec2(1, 0));
  }
  {
    ImVec2 wsize;
    wsize.x = 1280 / 5;
    wsize.y = 720 / 5;
    ImGui::Image((ImTextureID)depth_texture, wsize, ImVec2(0, 1), ImVec2(1, 0));
  }
  ImGui::End();

  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

  SDL_GL_SwapWindow(window);

  return quit;
}

Render::~Render() {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImGui::DestroyContext();
  SDL_DestroyWindow(window);
  SDL_Quit();
}

void Render::SetScene(Scene *scene, std::string scene_name) {

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

      printf("indices_size => %d\n", indices_size);

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

      printf("vertices_size => %d\n", vertices_size);

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
} // namespace UniRt
