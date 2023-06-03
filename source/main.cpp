#include "render.h"
#include "scene.h"
#include <iostream>

// oidn-1 -> all
// oidn-2 -> only albedo+noisy
// oidn-3 -> only noisy

int main() {
  std::cout << "Hello, World!" << std::endl;

  auto render = new UniRt::Render();

  auto scene = new UniRt::Scene("/home/vincent/Projects/glTF-Sample-Models/2.0/Sponza/glTF/Sponza.gltf");
  //  auto scene = new UniRt::Scene("/home/vincent/Documents/dragon.glb");

  render->SetScene(scene, "dragon");
  render->demo_mode = true;
  render->dataset_mode = false;

  while (!render->Update()) {
  }
}
