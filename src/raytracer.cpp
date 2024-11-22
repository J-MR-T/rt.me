#include <print>
#include <fstream>
// TODO disable exceptions and make json not throw
//#define JSON_NOEXCEPTION
//#define JSON_THROW_USER(ignored) do while(0){ std::println(stderr, "Not a valid json file"); std::exit(EXIT_FAILURE); };
#include "thirdparty/json.h"
#include "util.h"

using json = nlohmann::json;

void printHelpExit(char* argv0, int status){
    std::println(stderr, "Usage: {} <path to json scene file>", argv0);
    std::exit(status);
}

Texture readPPMTexture(std::string_view path){
    std::ifstream textureIfstream((std::string(path)));
    if(textureIfstream.fail()){
        std::perror("Couldn't read from texture file");
        std::exit(EXIT_FAILURE);
    }

    // on failure, just exit
    auto fail = [] [[noreturn]] (auto message) {
        std::println(stderr, "Invalid Texture PPM: {}", message);
        std::exit(EXIT_FAILURE);
    };

    auto maybeSkipComments = [&textureIfstream](){
        char c = textureIfstream.peek();
        if(c == '#'){
            // skip comments
            while((c = textureIfstream.get()) != '\n' && c != EOF);

            return;
        }
    };

    auto checkAndSkipWhitespaceComments = [&textureIfstream, &fail, &maybeSkipComments](std::string_view message = "Invalid spacing/whitespace in header"){
        maybeSkipComments();

        if(!std::isspace(textureIfstream.get()))
            fail(message);

        maybeSkipComments();
    };

    // for reading parts of header
    char buf[3] = {0};

    // read magic number, support P3 and P6
    textureIfstream.read(buf, 2);

    checkAndSkipWhitespaceComments("Invalid magic number for PPM");

    bool isP3 = std::strcmp(buf, "P3") == 0;
    if(!isP3 && std::strcmp(buf, "P6") != 0)
        fail("Invalid magic number: PPM P3 and P6 are supported");

    maybeSkipComments();

    // read width as ascii
    uint32_t width;
    if(!(textureIfstream >> width))
        fail("Invalid width");

    checkAndSkipWhitespaceComments();

    // read height as ascii
    uint32_t height;
    if(!(textureIfstream >> height))
        fail("Invalid height");

    checkAndSkipWhitespaceComments();

    // read maxval as ascii
    uint32_t maxval;
    if(!(textureIfstream >> maxval))
        fail("Invalid maxval");

    checkAndSkipWhitespaceComments();

    // we only support maxval 255
    if(maxval != 255)
        fail("Invalid maxval: only 255 is supported");

    // now depending on P3 or P6, read the rest of the file as binary or ascii
    std::vector<Color8Bit> pixels;
    pixels.reserve(width * height);
    for(uint64_t i = 0; i < width * height; i++){
        uint8_t r, g, b;
        if(isP3){
            // read ascii, need to read ints not chars
            uint32_t r32, g32, b32;
            if(!(textureIfstream >> r32) || r32 > 255)
                fail("Invalid red value");
            checkAndSkipWhitespaceComments();

            if(!(textureIfstream >> g32) || g32 > 255)
                fail("Invalid green value");
            checkAndSkipWhitespaceComments();

            if(!(textureIfstream >> b32) || b32 > 255)
                fail("Invalid blue value");
            checkAndSkipWhitespaceComments();

            r = r32;
            g = g32;
            b = b32;
        }else{
            // read binary
            if(!textureIfstream.read(reinterpret_cast<char*>(&r), 1))
                fail("Invalid red value");
            if(!textureIfstream.read(reinterpret_cast<char*>(&g), 1))
                fail("Invalid green value");
            if(!textureIfstream.read(reinterpret_cast<char*>(&b), 1))
                fail("Invalid blue value");
        }
        pixels.emplace_back(r, g, b);
    }
    return Texture(width, height, pixels);
}

Scene jsonFileToScene(std::string_view path){
    std::ifstream jsonIfstream((std::string(path)));

    if(jsonIfstream.fail()){
        std::perror("Couldn't read from json file");
        std::exit(EXIT_FAILURE);
    }
    json root;
    jsonIfstream >> root;

    // on failure, just exit
    auto fail = [] [[noreturn]] (auto message)  {
        std::println(stderr, "Invalid Json: {}", message);
        std::exit(EXIT_FAILURE);
    };

    auto getOrFail = [&fail](const auto& j, const auto& key){
        if(!j.contains(key))
            // TODO add "missing/no"
            fail(key);
        return j[key];
    };

    auto getOrElse = []<typename T>(const auto& j, const auto& key, const T& otherwise) -> T {
        if(!j.contains(key))
            return otherwise;

        return j[key];
    };

    auto jsonToVec2 = [](const auto& j){
        return Vec2(j[0], j[1]);
    };

    auto jsonToVec3 = [](const auto& j){
        return Vec3(j[0], j[1], j[2]);
    };

    if(!root.is_object())
        fail("not an object");

    json cameraJ            = getOrFail(root, "camera");
    uint32_t nBounces       = getOrElse(root, "nbounces", 1);
    std::string renderModeS = getOrFail(root, "rendermode");
    json sceneJ             = getOrFail(root, "scene");
    json sceneObjectsJ      = getOrFail(sceneJ, "shapes");
    if(!sceneObjectsJ.is_array())
        fail("shapes is not an array");
    Vec3 backgroundColor    = jsonToVec3(getOrFail(sceneJ, "backgroundcolor"));

    // map from path to texture, to deduplicate textures
    // shared pointers are a bit slow, and new/delete could be faster, but ownership is easier to track this way
    std::unordered_map<std::string, std::shared_ptr<Texture>> textures;

    uint32_t pathtracingSamplesPerPixel = 0;
    uint32_t pathtracingApertureSamplesPerPixelSample = 0;
    uint32_t pathtracingPointLightsamplesPerBounce = 0;
    uint32_t pathtracingHemisphereSamplesPerBounce = 0;

    // get render mode
    RenderMode renderMode;
    if(renderModeS == "phong")
        renderMode = RenderMode::PHONG;
    else if(renderModeS == "binary")
        renderMode = RenderMode::BINARY;
    else if(renderModeS == "debugbvh")
        renderMode = RenderMode::DEBUG_BVH;
    else if(renderModeS == "debugnormals")
        renderMode = RenderMode::DEBUG_NORMALS;
    else if(renderModeS == "pathtrace"){
        renderMode = RenderMode::PATHTRACE;

        json pathtracingOpts = getOrFail(root, "pathtracingOpts");

        pathtracingSamplesPerPixel = getOrFail(pathtracingOpts, "samplesPerPixel");
        pathtracingApertureSamplesPerPixelSample = getOrFail(pathtracingOpts, "apertureSamplesPerPixelSample");
        pathtracingPointLightsamplesPerBounce = getOrFail(pathtracingOpts, "pointLightSamplesPerBounce");
        pathtracingHemisphereSamplesPerBounce = getOrFail(pathtracingOpts, "hemisphereSamplesPerBounce");
        if(getOrFail(pathtracingOpts, "incremental"))
            renderMode = RenderMode::PATHTRACE_INCREMENTAL;
        
        // warn about hemispehre samples per bounce
        if(pathtracingHemisphereSamplesPerBounce > 1)
            std::println(stderr, "Warning: Hemisphere samples per bounce > 1 is supported, but not recommended, as it will result in exponential time complexity. Use at your own peril.");
    } else
        fail("Invalid rendermode");


    // get camera
    std::string cameraType = getOrFail(cameraJ, "type");
    std::unique_ptr<Camera> camera;
    if(cameraType == "pinhole"){
        camera = std::make_unique<PinholePerspectiveCamera>(
            jsonToVec3(getOrFail(cameraJ, "position")),
            jsonToVec3(getOrFail(cameraJ, "lookAt")),
            jsonToVec3(getOrFail(cameraJ, "upVector")),
            (float_t) getOrFail(cameraJ,  "fov"),
            (float_t) getOrFail(cameraJ,  "width"),
            (float_t) getOrFail(cameraJ,  "height"),
            (float_t) getOrFail(cameraJ,  "exposure")
        );
    }else if(cameraType == "orthographic"){
        camera = std::make_unique<OrthographicCamera>(
            jsonToVec3(getOrFail(cameraJ, "position")),
            jsonToVec3(getOrFail(cameraJ, "lookAt")),
            jsonToVec3(getOrFail(cameraJ, "upVector")),
            (float_t) getOrFail(cameraJ,  "width"),
            (float_t) getOrFail(cameraJ,  "height"),
            (float_t) getOrFail(cameraJ,  "exposure")
        );
    }else if(cameraType == "thinlens"){
        camera = std::make_unique<SimplifiedThinLensCamera>(
            jsonToVec3(getOrFail(cameraJ, "position")),
            jsonToVec3(getOrFail(cameraJ, "lookAt")),
            jsonToVec3(getOrFail(cameraJ, "upVector")),
            (float_t) getOrFail(cameraJ,  "fov"),
            (float_t) getOrFail(cameraJ,  "width"),
            (float_t) getOrFail(cameraJ,  "height"),
            (float_t) getOrFail(cameraJ,  "exposure"),
            (float_t) getOrFail(cameraJ,  "fstop"),
            (float_t) getOrFail(cameraJ,  "focalLength"),
            (float_t) getOrFail(cameraJ,  "focusDistance")
        );
    }

    std::vector<PointLight> lights;

    json lightsJ = getOrFail(sceneJ, "lightsources");
    if(!lightsJ.is_array())
        fail("lightsources is not an array");
    for(auto& lightJ: lightsJ){
        std::string type = getOrFail(lightJ, "type");
        if(type == "pointlight"){
            Vec3 position = jsonToVec3(getOrFail(lightJ, "position"));
            Vec3 intensityPerColor = jsonToVec3(getOrFail(lightJ, "intensity"));
            float_T shadowSoftness = getOrElse(lightJ, "shadowSoftness", 0.);
            if(shadowSoftness < 0)
                fail("Shadow softness must be positive");

            if(renderMode != RenderMode::PATHTRACE && renderMode != RenderMode::PATHTRACE_INCREMENTAL && shadowSoftness != 0.)
                fail("Shadow softness > 0 only supported in pathtracing mode");

            lights.emplace_back(position, intensityPerColor, shadowSoftness);
        }else{
            // TODO area lights
            fail("Invalid light type (if you're trying area lights: these are implemented as emissive objects - add an emissioncolor to an object instead)");
        }
    }

    // material

    auto jsonToMaterial = [&](const auto& j) -> std::optional<PhongMaterial>{
        std::optional<float_t> reflectivity = getOrFail(j, "reflectivity");
        if(!getOrFail(j, "isreflective"))
            reflectivity = std::nullopt;

        std::optional<float_t> refractiveIndex = getOrFail(j, "refractiveindex");
        if(!getOrFail(j, "isrefractive"))
            refractiveIndex = std::nullopt;


        // textures
        std::optional<std::shared_ptr<Texture>> texture = std::nullopt;
        if(j.contains("texture")){
            std::string texturePath = getOrFail(j, "texture");
            if(auto existingTexture = textures.find(texturePath); existingTexture != textures.end())
                texture = existingTexture->second;
            else
                texture = textures[texturePath] = std::make_shared<Texture>(readPPMTexture(texturePath));
        }

        return PhongMaterial(
            jsonToVec3(getOrFail(j, "diffusecolor")),
            jsonToVec3(getOrFail(j, "specularcolor")),
            (float_t) getOrFail(j,  "ks"),
            (float_t) getOrFail(j,  "kd"),
            (uint64_t) getOrFail(j, "specularexponent"),
            reflectivity,
            refractiveIndex,
            getOrElse(j, "emissioncolor", std::optional<json>()).and_then([&jsonToVec3](auto j){return std::optional(jsonToVec3(j));}),
            texture
        );
    };

    // objects
    std::vector<SceneObject> sceneObjects;
    sceneObjects.reserve(sceneObjectsJ.size());

    for(auto& sceneObjectJ:sceneObjectsJ){
        std::string type = getOrFail(sceneObjectJ, "type");
        // get material if it exists, transform the json into a material
        std::optional<PhongMaterial> material = getOrElse(sceneObjectJ, "material", std::optional<json>()).and_then(jsonToMaterial);

        if(type == "triangle"){
            // spheres and cylinders have textures mapped automatically, triangles need to be mapped manually
            Vec2 texCoordv0;
            Vec2 texCoordv1;
            Vec2 texCoordv2;
            if(material.has_value() && material->texture.has_value()){
                // get triangle text coords
                auto materialJ = sceneObjectJ["material"];
                texCoordv0 = jsonToVec2(getOrFail(materialJ, "txv0"));
                texCoordv1 = jsonToVec2(getOrFail(materialJ, "txv1"));
                texCoordv2 = jsonToVec2(getOrFail(materialJ, "txv2"));
            }
            sceneObjects.emplace_back(Triangle(
                jsonToVec3(getOrFail(sceneObjectJ, "v0")),
                jsonToVec3(getOrFail(sceneObjectJ, "v1")),
                jsonToVec3(getOrFail(sceneObjectJ, "v2")),
                std::move(material),
                texCoordv0,
                texCoordv1,
                texCoordv2
            ));
        }else if(type == "sphere"){
            sceneObjects.emplace_back(Sphere(
                jsonToVec3(getOrFail(sceneObjectJ, "center")),
                (float_t) getOrFail(sceneObjectJ, "radius"),
                std::move(material)
            ));
        }else if(type == "cylinder"){
            sceneObjects.emplace_back(Cylinder(
                jsonToVec3(getOrFail(sceneObjectJ, "center")),
                (float_t) getOrFail(sceneObjectJ, "radius"),
                (float_t) getOrFail(sceneObjectJ, "height"),
                jsonToVec3(getOrFail(sceneObjectJ, "axis")),
                std::move(material)
            ));
        }else{
            fail("Invalid shape");
        }
    }
    return Scene(nBounces,
        renderMode,
        std::move(camera),
        backgroundColor,
        lights,
        sceneObjects,
        pathtracingSamplesPerPixel,
        pathtracingApertureSamplesPerPixelSample,
        pathtracingPointLightsamplesPerBounce,
        pathtracingHemisphereSamplesPerBounce
    );
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

    MEASURE_TIME_START(wallTime);

    MEASURE_TIME_START(jsonTime);
    Renderer renderer(jsonFileToScene(argv[1]), "out.ppm");
    MEASURE_TIME_END(jsonTime);

    MEASURE_TIME_START(renderTime);

    // "convert" render mode to constexpr
    if(renderer.scene.renderMode == RenderMode::BINARY)
        renderer.render<RenderMode::BINARY>();
    else if(renderer.scene.renderMode == RenderMode::PHONG)
        renderer.render<RenderMode::PHONG>();
    else if(renderer.scene.renderMode == RenderMode::DEBUG_BVH)
        renderer.render<RenderMode::DEBUG_BVH>();
    else if(renderer.scene.renderMode == RenderMode::DEBUG_NORMALS)
        renderer.render<RenderMode::DEBUG_NORMALS>();
    else if(renderer.scene.renderMode == RenderMode::PATHTRACE)
        renderer.render<RenderMode::PATHTRACE>();
    else if(renderer.scene.renderMode == RenderMode::PATHTRACE_INCREMENTAL)
        renderer.render<RenderMode::PATHTRACE_INCREMENTAL>();
    else{
        std::println(stderr, "Render mode not supported yet");
        std::exit(EXIT_FAILURE);
    }

    MEASURE_TIME_END(renderTime);

    MEASURE_TIME_END(wallTime);

    std::println("Timing:\n- Wall: {}\n- Json to Scene: {}\n- Render: {}", MEASURED_TIME_AS_SECONDS(wallTime, 1), MEASURED_TIME_AS_SECONDS(jsonTime, 1), MEASURED_TIME_AS_SECONDS(renderTime, 1));

    return EXIT_SUCCESS;
}
