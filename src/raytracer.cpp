#include <print>
#include <chrono>

#include "raytracer.h"
#include "io.h"

void printHelpExit(char* argv0, int status){
    std::println(stderr, "Usage: {} <path to json scene file>", argv0);
    std::exit(status);
}


int main(int argc, char *argv[]) {
#ifndef NDEBUG
    std::println("Running in debug mode, this will be slow");
#endif
    if(argc != 2)
        printHelpExit(argv[0], EXIT_FAILURE);

    // measure time

    // my own time measurement macros/primitives (see jmrt.sh/monaco)
    // steady_clock == MONOTONIC, best for performance measurements
#define MEASURE_TIME_START(point) auto point ## _start = std::chrono::steady_clock::now()

#define MEASURE_TIME_END(point) auto point ## _end = std::chrono::steady_clock::now()

#define MEASURED_TIME_AS_SECONDS(point, iterations) std::chrono::duration_cast<std::chrono::duration<double>>(point ## _end - point ## _start).count()/(static_cast<double>(iterations))

    MEASURE_TIME_START(jsonTime);
    auto renderer = jsonFileToRenderer(argv[1]);
    MEASURE_TIME_END(jsonTime);

    MEASURE_TIME_START(renderTime);

    // "convert" render mode to constexpr - unfortunate boiler plate, but this is the only way to template this, and that has performance benefits
    auto renderMode = renderer->scene.renderMode;
    if(renderMode == RenderMode::BINARY)
        renderer->template render<RenderMode::BINARY>();
    else if(renderMode == RenderMode::PHONG)
        renderer->template render<RenderMode::PHONG>();
    else if(renderMode == RenderMode::DEBUG_BVH)
        renderer->template render<RenderMode::DEBUG_BVH>();
    else if(renderMode == RenderMode::DEBUG_NORMALS)
        renderer->template render<RenderMode::DEBUG_NORMALS>();
    else if(renderMode == RenderMode::PATHTRACE)
        renderer->template render<RenderMode::PATHTRACE>();
    else if(renderMode == RenderMode::PATHTRACE_INCREMENTAL)
        renderer->template render<RenderMode::PATHTRACE_INCREMENTAL>();
    else{
        std::println(stderr, "Render mode not supported yet");
        std::exit(EXIT_FAILURE);
    }

    MEASURE_TIME_END(renderTime);

    auto jsonSeconds = MEASURED_TIME_AS_SECONDS(jsonTime, 1);
    auto renderSeconds = MEASURED_TIME_AS_SECONDS(renderTime, 1);

    std::println(stderr, "Timing:\n- Wall: {}\n- Json to Scene: {}\n- Render: {}", jsonSeconds + renderSeconds, jsonSeconds, renderSeconds);

    return EXIT_SUCCESS;
}
