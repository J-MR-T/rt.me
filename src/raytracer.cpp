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

Scene jsonFileToScene(char* path){
    std::ifstream jsonIfstream(path);

    if(jsonIfstream.fail()){
        std::perror("Couldn't read from json file");
        std::exit(EXIT_FAILURE);
    }
    json root;
    jsonIfstream >> root;

    // on failure, just exit
    auto fail = [](auto message){
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

    auto jsonToVec3 = [](const auto& j){
        return Vec3(j[0], j[1], j[2]);
    };

    if(!root.is_object())
        fail("not an object");

    json cameraJ            = getOrFail(root, "camera");
    uint32_t nBounces       = getOrElse(root, "nBounces", 1);
    std::string renderModeS = getOrFail(root, "rendermode");
    json sceneJ             = getOrFail(root, "scene");
    json sceneObjectsJ      = getOrFail(sceneJ, "shapes");
    if(!sceneObjectsJ.is_array())
        fail("shapes is not an array");
    Vec3 backgroundColor    = jsonToVec3(getOrFail(sceneJ, "backgroundcolor"));

    // get render mode
    RenderMode renderMode;
    if(renderModeS == "phong")
        renderMode = RenderMode::PHONG;
    else if(renderModeS == "binary")
        renderMode = RenderMode::BINARY;
    else
        fail("Invalid rendermode");

    // get camera
    std::string cameraType = getOrFail(cameraJ, "type");
    // TODO for now just pinhole allowed
    if(cameraType != "pinhole")
        fail("Invalid camera type");

    // TODO this needs to be a pinhole camera in the future
    OrthographicCamera camera(
        jsonToVec3(getOrFail(cameraJ, "position")),
        jsonToVec3(getOrFail(cameraJ, "lookAt")),
        jsonToVec3(getOrFail(cameraJ, "upVector")),
        //(float_t) getOrFail(cameraJ, "fov"),
        (float_t) getOrFail(cameraJ, "width"),
        (float_t) getOrFail(cameraJ, "height"),
        (float_t) getOrFail(cameraJ, "exposure")
    );

    std::vector<PointLight> lights;

    // get lights if rendermode is phong
    if(renderMode == RenderMode::PHONG){
        json lightsJ = getOrFail(sceneJ, "lightsources");
        if(!lightsJ.is_array())
            fail("lightsources is not an array");
        for(auto& lightJ: lightsJ){
            std::string type = getOrFail(lightJ, "type");
            if(type == "pointlight"){
                Vec3 position = jsonToVec3(getOrFail(lightJ, "position"));
                Vec3 intensityPerColor = jsonToVec3(getOrFail(lightJ, "intensity"));
                // there's no color specified, so we'll just always use white for now
                lights.emplace_back(position, intensityPerColor);
            }else{
                // TODO area lights
                fail("Invalid light type (if you're trying area lights: not supported yet)");
            }
        }
    }

    // material

    auto jsonToMaterial = [&](const auto& j) -> std::optional<PhongMaterial>{
        auto reflectivity = getOrElse(j, "reflectivity", std::optional<float_t>());

        auto refractivity = getOrElse(j, "refractivity", std::optional<float_t>());

        return PhongMaterial(
            jsonToVec3(getOrFail(j, "diffusecolor")),
            jsonToVec3(getOrFail(j, "specularcolor")),
            (float_t) getOrFail(j,  "ks"),
            (float_t) getOrFail(j,  "kd"),
            (uint64_t) getOrFail(j, "specularexponent"),
            reflectivity,
            refractivity
        );
    };

    // objects

    std::vector<std::variant<Triangle, Sphere, Cylinder>> sceneObjects;
    sceneObjects.reserve(sceneObjectsJ.size());

    for(auto& sceneObjectJ:sceneObjectsJ){
        std::string type = getOrFail(sceneObjectJ, "type");
        // get material if it exists, transform the json into a material
        std::optional<PhongMaterial> material = getOrElse(sceneObjectJ, "material", std::optional<json>()).and_then(jsonToMaterial);

        if(type == "triangle"){
            sceneObjects.emplace_back(Triangle(
                jsonToVec3(getOrFail(sceneObjectJ, "v0")),
                jsonToVec3(getOrFail(sceneObjectJ, "v1")),
                jsonToVec3(getOrFail(sceneObjectJ, "v2")),
                std::move(material)
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
    return Scene(nBounces, renderMode, std::make_unique<OrthographicCamera>(camera), backgroundColor, sceneObjects);
}

int main(int argc, char *argv[]) {
    if(argc != 2)
        printHelpExit(argv[0], EXIT_FAILURE);

    Renderer renderer(jsonFileToScene(argv[1]), "out.ppm");
    renderer.render<RenderMode::BINARY>();
}
