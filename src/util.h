#pragma once
#include <vector>
#include <optional>
#include <cstdint>
#include <string>
#include <fstream>

struct PPMWriter{
private:
    std::ofstream file;

public:
    std::string filePath;
    uint32_t width, height;

    PPMWriter(std::string filePath, uint32_t width, uint32_t height)
        : filePath(filePath), width(width), height(height){
        file = std::ofstream(filePath);
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
};

// single precision for now
using float_t = float;

struct Vec2{
    float_t x,y;
};

struct Vec3{
    float_t x,y,z;

    Vec3 operator+(const Vec3& other) const{
        return Vec3(x+other.x, y+other.y, z+other.z);
    }

    Vec3 operator-(const Vec3& other) const{
        return Vec3(x-other.x, y-other.y, z-other.z);
    }

    Vec3 cross(const Vec3& other) const {
        return Vec3(
                 y*other.z - z*other.y,
                 z*other.x - x*other.z,
                 x*other.y - y*other.x
               );

    }
};

// TODO could use CRTP later, but normal dynamic dispatch is fine for now

struct Camera{
    virtual ~Camera() = default; 
};

struct PinholeOrthographicCamera : public Camera{
    // TODO
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

struct Material{
    virtual ~Material() = default;
};

struct PhongMaterial : public Material{
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

struct SceneObject{
    virtual ~SceneObject() = default;
};

struct Sphere : public SceneObject{
    Vec3 center;
    float_t radius;
    std::optional<Material> material;

    Sphere(Vec3 center, float_t radius, std::optional<Material> material)
        : center(center), radius(radius), material(material){ }
};

struct Cylinder : public SceneObject{
    Vec3 center;
    float_t radius;
    float_t height;
    Vec3 axis;
    std::optional<Material> material;

    Cylinder(Vec3 center, float_t radius, float_t height, Vec3 axis, std::optional<Material> material)
        : center(center), radius(radius), height(height), axis(axis), material(material){ }
};

struct Triangle : public SceneObject{
    Vec3 v0,v1,v2;
    // TODO could try deduplicating materials for triangle objects later on, for big meshes that all have the same material
    std::optional<Material> material;

    Triangle(Vec3 v0, Vec3 v1, Vec3 v2, std::optional<Material> material)
        : v0(v0), v1(v1), v2(v2), material(material){ }

    Vec3 faceNormal() const{
        return (v1-v0).cross(v2-v0);
    }

    // TODO normal vector interpolation at any point for smooth shading
};

enum class RenderMode{
    BINARY,
    PHONG,
};

struct Scene{
    uint32_t nBounces;
    RenderMode renderMode;
    Camera camera;
    Vec3 backgroundColor;
    std::vector<SceneObject> objects;
    // TODO lights at some point
};
