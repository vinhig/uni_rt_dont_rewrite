#include <iostream>
#include "render.h"
#include "scene.h"

int main() {
    std::cout << "Hello, World!" << std::endl;

    auto render = new UniRt::Render();

    auto scene = new UniRt::Scene("/home/vincent/Projects/glTF-Sample-Models/2.0/Sponza/glTF/SponzaLights.gltf");

    render->SetScene(scene, "/home/vincent/Projects/glTF-Sample-Models/2.0/Sponza/glTF/Sponza.gltf");

    while (!render->Update()) {

    }
}
