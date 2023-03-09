#include <iostream>
#include "render.h"
#include "scene.h"

int main() {
    std::cout << "Hello, World!" << std::endl;

    auto render = new UniRt::Render();

    auto scene = new UniRt::Scene("/home/vincent/Documents/delete_me.gltf");

    render->SetScene(scene, "/home/vincent/Documents/delete_me.gltf");

    while (!render->Update()) {

    }
}