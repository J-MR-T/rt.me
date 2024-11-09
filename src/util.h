#pragma once
#include <cassert>
#include <cmath>
#include <memory>
#include <vector>
#include <optional>
#include <cstdint>
#include <string>
#include <fstream>
#include <print>

// single precision for now
using float_t = float;

const float_t equalityEpsilon = 1e-6;

struct Vec2{
    float_t x,y;
};

struct Vec3{
    float_t x,y,z;

    Vec3(float_t x, float_t y, float_t z) : x(x), y(y), z(z){ }

    explicit Vec3(float_t uniform) : x(uniform), y(uniform), z(uniform){ }

    Vec3 operator+(const Vec3& other) const{
        return Vec3(x+other.x, y+other.y, z+other.z);
    }

    Vec3 operator-(const Vec3& other) const{
        return Vec3(x-other.x, y-other.y, z-other.z);
    }

    // scalar multiplication/division
    Vec3 operator*(float_t scalar) const{
        return Vec3(x*scalar, y*scalar, z*scalar);
    }

    Vec3 operator/(float_t scalar) const{
        return Vec3(x/scalar, y/scalar, z/scalar);
    }

    // elementwise multiplication
    Vec3 operator*(Vec3 other) const{
        return Vec3(x*other.x, y*other.y, z*other.z);
    }

    bool operator==(const Vec3& other) const{
        // epsilon comparison
        return std::abs(x - other.x) < equalityEpsilon &&
            std::abs(y - other.y) < equalityEpsilon &&
            std::abs(z - other.z) < equalityEpsilon;
    }

    Vec3 operator-() {
        return *this * -1;
    }

    // state modifying operators
    Vec3& operator+=(const Vec3& other){
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    float_t dot(const Vec3& other) const{
        return x*other.x + y*other.y + z*other.z;
    }

    float_t length() const{
        return std::sqrt(dot(*this));
    }

    Vec3 normalized() const{
        return *this / length();
    }

    Vec3 cross(const Vec3& other) const {
        return Vec3(
                 y*other.z - z*other.y,
                 z*other.x - x*other.z,
                 x*other.y - y*other.x
               );
    }
};

struct PPMWriter{
private:
    std::ofstream file;

public:
    std::string filePath;
    uint32_t width, height;

    PPMWriter(std::string_view filePath, uint32_t width, uint32_t height)
        : filePath(filePath), width(width), height(height){
        file = std::ofstream(this->filePath, std::ios::binary);
        if(file.fail()){
            std::perror("Couldn't open file for writing");
            std::exit(EXIT_FAILURE);
        }
        // we're writing binary ppm, i.e. P6

        // write header
        file << "P6\n" << width << " " << height << "\n255\n";
        // the rest is the pixel data, which we'll write later
    }

    ~PPMWriter(){
        file.close();
    }

    /// write a single pixel in binary format, pixels are iterated over row by row
    void writePixel(uint8_t r, uint8_t g, uint8_t b){
        file.put(r);
        file.put(g);
        file.put(b);
    }

    void writePixel(Vec3 color){
        writePixel(
            static_cast<uint8_t>(color.x * 255),
            static_cast<uint8_t>(color.y * 255),
            static_cast<uint8_t>(color.z * 255)
        );
    }
};


struct Ray{
    Vec3 origin;
    Vec3 direction;

    /// assumes that the direction is normalized!
    Ray(Vec3 origin, Vec3 direction)
        : origin(origin), direction(direction){
        assert(direction == direction.normalized() && "Ray direction must be normalized");
    }
};

// TODO could use CRTP later, but normal dynamic dispatch is fine for now

struct Camera{
    virtual ~Camera() = default; 

    virtual Ray generateRay(Vec2 pixelInPixelScreenSpace) = 0;
};

struct OrthographicCamera : public Camera{
    // TODO think about these
    Vec3 position;
    Vec3 direction;
    Vec3 up;
    Vec3 right;
    float_t width;
    float_t height;
    float_t exposure;

    // TODO maybe experiment with && and std::move to avoid some copies
    OrthographicCamera(
            Vec3 position,
            Vec3 direction,
            Vec3 up,
            float_t width,
            float_t height,
            float_t exposure)
        : position(position), direction(direction.normalized()), up(up.normalized()), right(direction.cross(up).normalized()), width(width), height(height), exposure(exposure){
        const float_t aspectRatio = width / height;
        imagePlaneDimensions = Vec2(aspectRatio*imagePlaneHeight, imagePlaneHeight);
    }

private:
    float_t imagePlaneHeight = 1.0;
    Vec2 imagePlaneDimensions;

public:

    /// gets a pixel in pixel screen space, i.e. [0,width]x[0,height]
    /// outputs a ray in world space, i.e. adjusting for the camera's position
    virtual Ray generateRay(Vec2 pixelInPixelScreenSpace) override{
        // so, the pixel is in the range [0,width]x[0,height]
        // we want to map this to [-0.5,0.5]x[-0.5,0.5] in the camera's space
        // which is then mapped to the image plane in world space

        const float_t pixelWidthHeight = 1.0;

        Vec2 pixelCenterInCameraSpace = Vec2(
            (pixelInPixelScreenSpace.x + /* use center of pixel */ pixelWidthHeight/2) / width - 0.5,
            (pixelInPixelScreenSpace.y + pixelWidthHeight/2) / height - 0.5
        );


        // scale to world space scale (not translation yet) by multiplying by the image plane dimensions
        Vec2 pixelScaledByWorldSpace = Vec2(pixelCenterInCameraSpace.x * imagePlaneDimensions.x, pixelCenterInCameraSpace.y * imagePlaneDimensions.y);

        // now after scaling to world space, translate to wrold space, and add the camera's right/up directions
        Vec3 rayOrigin = position + right*pixelScaledByWorldSpace.x + up*pixelScaledByWorldSpace.y;

        // for an orthographic camera, basically just shoot a ray in the look direction
        return Ray(
            rayOrigin,
            direction
        );
    }
};

struct PinholePerspectiveCamera : public Camera{
    Vec3 position;
    // TODO direction vs look at ?
    Vec3 direction;
    Vec3 up;
    float_t fov;
    float_t width;
    float_t height;
    float_t exposure;

    PinholePerspectiveCamera(
            Vec3 position,
            Vec3 lookAt,
            Vec3 up,
            float_t fov,
            float_t width,
            float_t height,
            float_t exposure)
        : position(position), direction(lookAt), up(up), fov(fov), width(width), height(height), exposure(exposure){ }
};


struct PhongMaterial {
    // technically this is not mandated in the json format, but it is part of the phong model, so we'll keep it here
    Vec3 ambientColor = Vec3(0,0,0);
    Vec3 diffuseColor;
    Vec3 specularColor;
    float_t ks,kd;
    uint64_t specularExponent;
    std::optional<float_t> reflectivity;
    std::optional<float_t> refractivity;

    PhongMaterial(
            Vec3 diffuseColor,
            Vec3 specularColor,
            float_t ks,
            float_t kd,
            uint64_t specularExponent,
            std::optional<float_t> reflectivity,
            std::optional<float_t> refractivity)
        : diffuseColor(diffuseColor), specularColor(specularColor), ks(ks), kd(kd), specularExponent(specularExponent), reflectivity(reflectivity), refractivity(refractivity){ }
};

//struct Material{
//    // TODO this would actually be brdfs in the future
//    std::variant<PhongMaterial, int> material;
//
//    Material(std::variant<PhongMaterial, int> material)
//        : material(material){ }
//};

struct Intersection{
    // ray that caused the intersection
    Ray incomingRay;
    Vec3 point;
    Vec3 surfaceNormal;
    PhongMaterial material;

    Intersection(Ray incomingray, Vec3 point, Vec3 surfaceNormal, PhongMaterial material)
        : incomingRay(incomingray), point(point), surfaceNormal(surfaceNormal), material(material){ }

    float_t distance() const{
        return (point - incomingRay.origin).length();
    }
};

inline PhongMaterial givenMaterialOrDefault(std::optional<PhongMaterial> material){
    if(material.has_value())
        return std::move(material.value());
    else
        // default material, because the json format allows for no material to be specified
        return PhongMaterial(Vec3(1,1,1), Vec3(1,1,1), 0.5, 0.5, 32, 0.0, 0.0);
}

struct Sphere {
    Vec3 center;
    float_t radius;
    PhongMaterial material;

    Sphere(Vec3 center, float_t radius, std::optional<PhongMaterial> material)
        : center(center), radius(radius), material(givenMaterialOrDefault(std::move(material))){ }

    std::optional<Intersection> intersect(const Ray& ray) {
        // chatgpt generated code

        // Vector from the ray's origin to the sphere's center
        Vec3 oc = ray.origin - center;

        // Coefficients of the quadratic equation (a*t^2 + b*t + c = 0)
        float_t a = ray.direction.dot(ray.direction);               // a = D•D
        float_t b = 2.0 * oc.dot(ray.direction);                    // b = 2 * oc•D
        float_t c = oc.dot(oc) - radius * radius;                   // c = (oc•oc - r^2)

        // Discriminant of the quadratic equation
        float_t discriminant = b * b - 4 * a * c;

        // No intersection if the discriminant is negative
        if (discriminant < 0) {
            return std::nullopt;
        }

        // Calculate the two intersection distances along the ray
        float_t sqrtDiscriminant = std::sqrt(discriminant);
        float_t t1 = (-b - sqrtDiscriminant) / (2.0 * a);
        float_t t2 = (-b + sqrtDiscriminant) / (2.0 * a);


        // Choose the closest intersection point in front of the ray origin
        float_t t = (t1 > 0) ? t1 : ((t2 > 0) ? t2 : -1);
        // if both are behind the ray, return no intersection
        if (t < 0) {
            return std::nullopt;
        }

        // Calculate intersection details
        Vec3 intersectionPoint = ray.origin + ray.direction * t;
        Vec3 intersectionNormal = (intersectionPoint - center).normalized();

        return Intersection(ray, std::move(intersectionPoint), std::move(intersectionNormal), material);
    }

};

struct Cylinder {
    Vec3 center;
    float_t radius;
    float_t height;
    Vec3 axis;
    PhongMaterial material;

    Cylinder(Vec3 center, float_t radius, float_t height, Vec3 axis, std::optional<PhongMaterial> material)
        : center(center), radius(radius), height(height), axis(axis), material(givenMaterialOrDefault(std::move(material))){ }

    std::optional<Intersection> intersect(const Ray& ray) {
        // TODO
        return std::nullopt;
    }
};

struct Triangle {
    Vec3 v0,v1,v2;
    // TODO could try deduplicating materials for triangle objects later on, for big meshes that all have the same material
    PhongMaterial material;

    Triangle(Vec3 v0, Vec3 v1, Vec3 v2, std::optional<PhongMaterial> material)
        : v0(v0), v1(v1), v2(v2), material(givenMaterialOrDefault(std::move(material))){ }

    Vec3 faceNormal() const{
        return (v1-v0).cross(v2-v0);
    }

    // TODO normal vector interpolation at any point for smooth shading

    std::optional<Intersection> intersect(const Ray& ray) {
        // TODO
        return std::nullopt;
    }
};

struct PointLight{
    Vec3 position;
    // the json files seem to integrate intensity and color into one vector
    Vec3 intensityPerColor;
};

enum class RenderMode{
    BINARY,
    PHONG,
};

struct Scene{
    uint32_t nBounces;
    RenderMode renderMode;
    std::unique_ptr<Camera> camera;
    Vec3 backgroundColor;

    std::vector<PointLight> lights;

    // use separate vectors for each type of object (and dont use polymorphism) for better cache locality
    // would have to use
    // a dynamic dispatch for each intersection (very expensive), and
    // b pointers inside the vectors, ruining cache locality
    //std::vector<Triangle> triangles;
    //std::vector<Sphere> spheres;
    //std::vector<Cylinder> cylinders;
    //
    std::vector<std::variant<Triangle, Sphere, Cylinder>> objects;

    //Scene(uint32_t nBounces, RenderMode renderMode, std::unique_ptr<Camera> camera, Vec3 backgroundColor, std::vector<Triangle> triangles, std::vector<Sphere> spheres, std::vector<Cylinder> cylinders)
    //    : nBounces(nBounces), renderMode(renderMode), camera(std::move(camera)), backgroundColor(backgroundColor), triangles(triangles), spheres(spheres), cylinders(cylinders){ }
    Scene(uint32_t nBounces, RenderMode renderMode, std::unique_ptr<Camera> camera, Vec3 backgroundColor, std::vector<std::variant<Triangle, Sphere, Cylinder>> objects)
        : nBounces(nBounces), renderMode(renderMode), camera(std::move(camera)), backgroundColor(backgroundColor), objects(objects){ }
};

struct Renderer{

    Scene scene;
    PPMWriter writer;

    Renderer(Scene&& scene, std::string_view outputFilePath)
        // TODO adjust writer
        : scene(std::move(scene)), writer(outputFilePath, 1200, 800){ }

    /// shades a single intersection point
    Vec3 blinnPhongShading(const Intersection& intersection, uint32_t bounces = 1){
        // mostly chatgpt generated

        if (bounces > scene.nBounces)
            return Vec3(0.0f);

        Vec3 color(0.0f);

        Vec3 ambient = intersection.material.ambientColor;
        Vec3 diffuse = intersection.material.diffuseColor;
        Vec3 specular = intersection.material.specularColor;
        float specularExponentShinyness = intersection.material.specularExponent;

        // repeat for all lights in the scene
        for(auto& light: scene.lights){
            Vec3 ambientTerm = ambient * light.intensityPerColor;
            Vec3 L = (light.position - intersection.point).normalized();
            Vec3 N = intersection.surfaceNormal;

            float diff = std::max(N.dot(L), 0.0f);
            Vec3 diffuseTerm = diffuse * diff * light.intensityPerColor;

            Vec3 V = -intersection.incomingRay.direction.normalized();  // Use the ray direction from the intersection
            Vec3 H = (L + V).normalized();
            float spec = std::pow(std::max(N.dot(H), 0.0f), specularExponentShinyness);
            Vec3 specularTerm = specular * spec * light.intensityPerColor;

            color += ambientTerm + diffuseTerm + specularTerm;

            Vec3 reflectionDir = intersection.incomingRay.direction - N * 2 * (intersection.incomingRay.direction.dot(N));
            Ray reflectionRay(intersection.point + reflectionDir * 0.001f, reflectionDir);

            if (auto reflectionIntersection = traceRayToClosestIntersection(reflectionRay)) {
                // only reflect if material is reflective
                if(!intersection.material.reflectivity.has_value())
                    continue;

                color += blinnPhongShading(*reflectionIntersection, bounces + 1) * *intersection.material.reflectivity;
            }
        }

        return color;
    }

    std::optional<Intersection> traceRayToClosestIntersection(const Ray& ray){
        auto closestIntersection = std::optional<Intersection>();
        for(auto& genericObject: scene.objects){
            std::visit([&](auto&& object){
                if(auto intersection = object.intersect(ray))
                    if(!closestIntersection.has_value() || intersection->distance() < closestIntersection->distance())
                        closestIntersection = *intersection;
            }, genericObject);
        }
        return closestIntersection;
    }

    template<RenderMode mode>
    void render(){
        for(uint32_t y = 0; y < 800; y++){
            for(uint32_t x = 0; x < 1200; x++){
                Ray cameraRay = scene.camera->generateRay(Vec2(x, y));
                //std::println("Camera ray direction: ({},{},{}) from ({},{},{})", cameraRay.direction.x, cameraRay.direction.y, cameraRay.direction.z, cameraRay.origin.x, cameraRay.origin.y, cameraRay.origin.z);
                // TODO other objects etc
                // TODO get closest intersection, not just test one
                auto closestIntersection = traceRayToClosestIntersection(cameraRay);

                if(closestIntersection.has_value()){
                    if constexpr (mode == RenderMode::BINARY){
                        writer.writePixel(255, 255, 255);
                    }else if constexpr (mode == RenderMode::PHONG){
                        writer.writePixel(blinnPhongShading(*closestIntersection));
                    }else{
                        static_assert(false, "Invalid render mode");
                    }
                }else{
                    writer.writePixel(scene.backgroundColor);
                }
            }
        }
    }



};
