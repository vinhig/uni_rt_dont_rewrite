#include <iostream>
#include "render.h"
#include "scene.h"

int main() {
    std::cout << "Hello, World!" << std::endl;

    auto render = new UniRt::Render();

    auto scene = new UniRt::Scene("/home/vincent/Documents/delete_me2.gltf");

    // render->SetScene(scene, "/home/vincent/Documents/cute_map.glb");
    render->SetScene(scene, "/home/vincent/Documents/delete_me2.gltf");

    while (!render->Update()) {

    }
}
