#include "render.h"
#include "scene.h"
#include <iostream>

int main() {
  std::cout << "Hello, World!" << std::endl;

  auto render = new UniRt::Render();

  auto scene = new UniRt::Scene("/home/vincent/Projects/glTF-Sample-Models/2.0/Sponza/glTF/Sponza.gltf");

  render->SetScene(scene, "sponza");
  render->demo_mode = true;

  while (!render->Update()) {
  }
}
