#include <print>
#include <fstream>
// TODO disable exceptions and make json not throw
//#define JSON_NOEXCEPTION
//#define JSON_THROW_USER(ignored) do while(0){ std::println(stderr, "Not a valid json file"); std::exit(EXIT_FAILURE); };
#include "thirdparty/json.h"
#include "src/util.h"

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

    auto getOrElse = [](const auto& j, const auto& key, const auto& otherwise) -> json{
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
    // TODO
    //json lightsJ            = getOrFail(sceneJ, "lightsources");
    // if(!lightsJ.is_array())
        // fail("lightsources is not an array");

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

    PinholePerspectiveCamera camera(
        jsonToVec3(getOrFail(cameraJ, "position")),
        jsonToVec3(getOrFail(cameraJ, "lookAt")),
        jsonToVec3(getOrFail(cameraJ, "upVector")),
        (float_t) getOrFail(cameraJ, "fov"),
        (float_t) getOrFail(cameraJ, "width"),
        (float_t) getOrFail(cameraJ, "height"),
        (float_t) getOrFail(cameraJ, "exposure")
    );

    // TODO get lights

    // material

    auto jsonToMaterial = [&](const auto& j){
        std::optional<float_t> reflectivity = getOrFail(j, "reflectivity");
        if(!getOrFail(j, "isreflective"))
            reflectivity = std::nullopt;

        std::optional<float_t> refractivity = getOrFail(j, "refractivity");
        if(!getOrFail(j, "isrefractive"))
            refractivity = std::nullopt;

        return PhongMaterial(
            jsonToVec3(getOrFail(j, "diffuseColor")),
            jsonToVec3(getOrFail(j, "specularColor")),
            (float_t) getOrFail(j,  "ks"),
            (float_t) getOrFail(j,  "kd"),
            (uint64_t) getOrFail(j, "specularExponent"),
            reflectivity,
            refractivity
        );
    };

    // objects

    std::vector<SceneObject> objects(sceneObjectsJ.size());

    for(auto& sceneObjectJ:sceneObjectsJ){
        std::string type = getOrFail(sceneObjectJ, "type");
        std::optional<Material> material = std::nullopt;
        if(sceneObjectJ.contains("material"))
            material = jsonToMaterial(sceneObjectJ["material"]);

        if(type == "triangle"){
            objects.emplace_back(Triangle(
                jsonToVec3(getOrFail(sceneObjectJ, "v0")),
                jsonToVec3(getOrFail(sceneObjectJ, "v1")),
                jsonToVec3(getOrFail(sceneObjectJ, "v2")),
                material
            ));
        }else if(type == "sphere"){
            objects.emplace_back(Sphere(
                jsonToVec3(getOrFail(sceneObjectJ, "center")),
                (float_t) getOrFail(sceneObjectJ, "radius"),
                material
            ));
        }else if(type == "cylinder"){
            objects.emplace_back(Cylinder(
                jsonToVec3(getOrFail(sceneObjectJ, "center")),
                (float_t) getOrFail(sceneObjectJ, "radius"),
                (float_t) getOrFail(sceneObjectJ, "height"),
                jsonToVec3(getOrFail(sceneObjectJ, "axis")),
                material
            ));
        }else{
            fail("Invalid shape");
        }
    }
    return Scene(nBounces, renderMode, camera, backgroundColor, objects);
}

int main(int argc, char *argv[]) {
    if(argc != 2)
        printHelpExit(argv[0], EXIT_FAILURE);

    //jsonFileToScene(argv[1]);

    PPMWriter ppmWriter("test.ppm", 3, 2);
    ppmWriter.writePixel(255, 0, 0);
    ppmWriter.writePixel(0, 255, 0);
    ppmWriter.writePixel(0, 0, 255);
    ppmWriter.writePixel(255, 255, 0);
    ppmWriter.writePixel(255, 0, 255);
    ppmWriter.writePixel(0, 255, 255);
}
