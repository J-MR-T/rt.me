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
    std::string pathStr(path);
    std::ifstream textureIfstream(pathStr);
    if(textureIfstream.fail()){
        std::perror(("Couldn't read from texture file \"" + pathStr + "\"").c_str());
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
        }
    };

    /// ensures that there is exactly one whitespace, preceded or followed by comments
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

    if(width != height)
        std::println(stderr, "Warning: non-square textures are not supported - use at your own risk");

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

Renderer jsonFileToRenderer(std::string_view path){
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

    /// gets a value from the json object
    /// if more than one key is specified, these keys are aliases, and only *exactly* one of them must be present
    auto getOrFail = [&fail](const auto& j, const auto&... keys){
        unsigned numContained = ((unsigned) j.contains(keys) + ...);

        if(numContained == 0)
            fail("exactly one of " + ((std::string) keys + ... + "/") + " must be present. Object: " + j.dump());
        else if(numContained > 1)
            fail("ambiguous keys, only one of " + ((std::string) keys + ... + "/") + " must be present. Object: " + j.dump());

        // annoying template magic to get the one that is contained
        auto findContained = [&j](this auto& findContained, const auto& key, const auto&... remainingKeys){
            if(j.contains(key))
                return j[key];

            if constexpr(sizeof...(remainingKeys) > 0)
                return findContained(std::forward<decltype(remainingKeys)>(remainingKeys)...);
            else
                std::unreachable();
        };

        return findContained(std::forward<decltype(keys)>(keys)...);
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

    // === start actually parsing ===

    std::string outputPathS = getOrElse(root, "outfile", std::string("out.ppm"));

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
            (float_T) getOrFail(cameraJ,  "fov"),
            (float_T) getOrFail(cameraJ,  "width"),
            (float_T) getOrFail(cameraJ,  "height"),
            (float_T) getOrFail(cameraJ,  "exposure")
        );
    }else if(cameraType == "orthographic"){
        camera = std::make_unique<OrthographicCamera>(
            jsonToVec3(getOrFail(cameraJ, "position")),
            jsonToVec3(getOrFail(cameraJ, "lookAt")),
            jsonToVec3(getOrFail(cameraJ, "upVector")),
            (float_T) getOrFail(cameraJ,  "width"),
            (float_T) getOrFail(cameraJ,  "height"),
            (float_T) getOrFail(cameraJ,  "exposure")
        );
    }else if(cameraType == "thinlens"){
        camera = std::make_unique<SimplifiedThinLensCamera>(
            jsonToVec3(getOrFail(cameraJ, "position")),
            jsonToVec3(getOrFail(cameraJ, "lookAt")),
            jsonToVec3(getOrFail(cameraJ, "upVector")),
            (float_T) getOrFail(cameraJ,  "fov"),
            (float_T) getOrFail(cameraJ,  "width"),
            (float_T) getOrFail(cameraJ,  "height"),
            (float_T) getOrFail(cameraJ,  "exposure"),
            (float_T) getOrFail(cameraJ,  "fstop"),
            (float_T) getOrFail(cameraJ,  "focalLength"),
            (float_T) getOrFail(cameraJ,  "focusDistance")
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

            if(renderMode == RenderMode::PHONG && shadowSoftness != 0.)
                fail("Shadow softness > 0 only supported in pathtracing mode");

            lights.emplace_back(position, intensityPerColor, shadowSoftness);
        }else{
            // TODO area lights
            fail("Invalid light type (if you're trying area lights: these are implemented as emissive objects - add an emissioncolor to an object instead)");
        }
    }

    auto jsonTryGetTexture = [&](const auto& j) {
        std::optional<std::shared_ptr<Texture>> texture = std::nullopt;
        if(j.contains("texture")){
            std::string texturePath = getOrFail(j, "texture");
            if(auto existingTexture = textures.find(texturePath); existingTexture != textures.end())
                texture = existingTexture->second;
            else
                texture = textures[texturePath] = std::make_shared<Texture>(readPPMTexture(texturePath));
        }
        return texture;
    };


    // material

    auto jsonToPhongMaterial = [&](const auto& j) -> std::optional<PhongMaterial>{
        // warn if (some) keys are present that are specific to BRDF materials
        if(j.contains("emissive") || j.contains("emissiveness") || j.contains("emission"))
            std::println(stderr, "Warning: Phong materials don't support emissive parameters - will be ignored");

        std::optional<float_T> reflectivity = getOrFail(j, "reflectivity");
        if(!getOrFail(j, "isreflective"))
            reflectivity = std::nullopt;

        std::optional<float_T> refractiveIndex = getOrFail(j, "refractiveindex");
        if(!getOrFail(j, "isrefractive"))
            refractiveIndex = std::nullopt;

        // textures
        std::optional<std::shared_ptr<Texture>> texture = jsonTryGetTexture(j);

        return PhongMaterial(
            jsonToVec3(getOrFail(j, "diffusecolor")),
            jsonToVec3(getOrFail(j, "specularcolor")),
            (float_T) getOrFail(j,  "ks"),
            (float_T) getOrFail(j,  "kd"),
            (uint64_t) getOrFail(j, "specularexponent"),
            reflectivity,
            refractiveIndex,
            texture
        );
    };

    auto jsonToBRDFMaterial = [&](const auto& j) -> std::optional<PrincipledBRDFMaterial>{
        // TODO use
        //std::optional<float_T> reflectivity = getOrFail(j, "reflectivity");
        //if(!getOrFail(j, "isreflective"))
        //    reflectivity = std::nullopt;
        //
        //std::optional<float_T> refractiveIndex = getOrFail(j, "refractiveindex");
        //if(!getOrFail(j, "isrefractive"))
        //    refractiveIndex = std::nullopt;

        // warn if (some) keys are present that are specific to phong materials
        if(j.contains("specularcolor") || j.contains("specularexponent"))
            std::println(stderr, "Warning: Principled BRDF materials don't support specularcolor, specularexponent parameters - will be ignored");

        // textures
        std::optional<std::shared_ptr<Texture>> texture = jsonTryGetTexture(j);

        // try to maintain some compatibility with phong materials:
        // - "basecolor" can also be "diffusecolor"
        // - "basecolorintensity" can also be "kd"
        // - "specular" can also be "ks"
        // ...

        return PrincipledBRDFMaterial(
            jsonToVec3(getOrFail(j, "diffusecolor", "basecolor")),
            texture,
            (float_T) getOrElse(j,  "emissive", 0.),
            (float_T) getOrFail(j,  "kd", "basecolorintensity"),
            (float_T) getOrFail(j, "metallic", "metallicness"),
            (float_T) getOrFail(j, "subsurface"),
            (float_T) getOrFail(j,  "ks", "specular"),
            (float_T) getOrFail(j, "roughness"),
            (float_T) getOrFail(j, "speculartint"),
            (float_T) getOrFail(j, "anisotropic"),
            (float_T) getOrFail(j, "sheen"),
            (float_T) getOrFail(j, "sheentint"),
            (float_T) getOrFail(j, "clearcoat"),
            (float_T) getOrFail(j, "clearcoatgloss")
        );
    };

    // objects
    std::vector<SceneObject> sceneObjects;
    sceneObjects.reserve(sceneObjectsJ.size());

    for(auto& sceneObjectJ:sceneObjectsJ){
        std::string type = getOrFail(sceneObjectJ, "type");
        // get material if it exists, transform the json into a material
        auto material = [&] -> Material{
            if(renderMode == RenderMode::BINARY && sceneObjectJ.contains("material")){
                fail("Binary mode doesn't support materials");
            }else if(renderMode == RenderMode::PHONG){
                PhongMaterial phongMaterial = getOrElse(sceneObjectJ, "material", std::optional<json>()).and_then(jsonToPhongMaterial).value_or(defaultPhongMaterial);
                return Material(phongMaterial);
            }else{
                PrincipledBRDFMaterial principledBRDFMaterial = getOrElse(sceneObjectJ, "material", std::optional<json>()).and_then(jsonToBRDFMaterial).value_or(defaultPrincipledBRDFMaterial);
                return Material(principledBRDFMaterial);
            }
        }();

        if(type == "triangle"){
            // triangles have fully customizable texture mapping
            Vec2 texCoordv0;
            Vec2 texCoordv1;
            Vec2 texCoordv2;
            if(sceneObjectJ.contains("material") && material.hasTexture()){
                // get triangle text coords
                auto materialJ = sceneObjectJ["material"];
                texCoordv0 = jsonToVec2(getOrFail(materialJ, "txv0"));
                texCoordv1 = jsonToVec2(getOrFail(materialJ, "txv1"));
                texCoordv2 = jsonToVec2(getOrFail(materialJ, "txv2"));
            }else if(sceneObjectJ.contains("txv0") || sceneObjectJ.contains("txv1") || sceneObjectJ.contains("txv2")){
                std::println(stderr, "Warning: texture coordinates specified, but no texture present - will be ignored");
            }


            // determine whether the triangle has vertex normals, choose appropriate triangle type
            bool shouldHaveVertexNormals = sceneObjectJ.contains("v0Normal");
            if(shouldHaveVertexNormals) {
                Vec3 v0Normal = jsonToVec3(getOrFail(sceneObjectJ, "v0Normal"));
                Vec3 v1Normal = jsonToVec3(getOrFail(sceneObjectJ, "v1Normal"));
                Vec3 v2Normal = jsonToVec3(getOrFail(sceneObjectJ, "v2Normal"));
                
                sceneObjects.emplace_back(TriangleWithVertexNormals(
                    jsonToVec3(getOrFail(sceneObjectJ, "v0")),
                    jsonToVec3(getOrFail(sceneObjectJ, "v1")),
                    jsonToVec3(getOrFail(sceneObjectJ, "v2")),
                    std::move(material),
                    texCoordv0,
                    texCoordv1,
                    texCoordv2,
                    {v0Normal, v1Normal, v2Normal}
                ));
            }else{
                sceneObjects.emplace_back(TriangleWithConstantFaceNormal(
                    jsonToVec3(getOrFail(sceneObjectJ, "v0")),
                    jsonToVec3(getOrFail(sceneObjectJ, "v1")),
                    jsonToVec3(getOrFail(sceneObjectJ, "v2")),
                    std::move(material),
                    texCoordv0,
                    texCoordv1,
                    texCoordv2
                ));
            }
        }else if(type == "sphere"){
            // spheres are texture-mapped automatically
            sceneObjects.emplace_back(Sphere(
                jsonToVec3(getOrFail(sceneObjectJ, "center")),
                (float_T) getOrFail(sceneObjectJ, "radius"),
                std::move(material)
            ));
        }else if(type == "cylinder"){
            // cylinders are mostly mapped automatically, but have some additional parameters
            sceneObjects.emplace_back(Cylinder(
                jsonToVec3(getOrFail(sceneObjectJ, "center")),
                (float_T) getOrFail(sceneObjectJ, "radius"),
                (float_T) getOrFail(sceneObjectJ, "height"),
                jsonToVec3(getOrFail(sceneObjectJ, "axis")),
                std::move(material),
                getOrElse(sceneObjectJ, "textureSideStretchFactor", 1.),
                getOrElse(sceneObjectJ, "textureCapScale", 1.)
            ));
        }else{
            fail("Invalid shape");
        }
    }

    return Renderer(
        Scene(nBounces,
            renderMode,
            std::move(camera),
            backgroundColor,
            lights,
            sceneObjects,
            pathtracingSamplesPerPixel,
            pathtracingApertureSamplesPerPixelSample,
            pathtracingPointLightsamplesPerBounce,
            pathtracingHemisphereSamplesPerBounce
        ),
        outputPathS
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
    Renderer renderer = jsonFileToRenderer(argv[1]);
    MEASURE_TIME_END(jsonTime);

    MEASURE_TIME_START(renderTime);

    // "convert" render mode to constexpr - unfortunate boiler plate, but this is the only way to template this
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
