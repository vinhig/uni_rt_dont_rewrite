#include "render.h"
#include "scene.h"
#include <iostream>

int main() {
  std::cout << "Hello, World!" << std::endl;

  auto render = new UniRt::Render();

  auto scene = new UniRt::Scene("/home/vincent/Documents/dragon.glb");

  // render->SetScene(scene, "/home/vincent/Documents/cute_map.glb");
  render->SetScene(scene, "/home/vincent/Documents/dragon.glb");

  while (!render->Update()) {
  }
}
