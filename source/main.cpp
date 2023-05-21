#include "render.h"
#include "scene.h"
#include <iostream>

int main() {
  std::cout << "Hello, World!" << std::endl;

  auto render = new UniRt::Render();

  auto scene = new UniRt::Scene("/home/vincent/Documents/delete_me2.glb");

  render->SetScene(scene, "deleteme");
  render->demo_mode = true;

  while (!render->Update()) {
  }
}
