#pragma once
#include <cassert>
#include <cmath>
#include <memory>
#include <random>
#include <ranges>
#include <vector>
#include <optional>
#include <cstdint>
#include <string>
#include <fstream>
#include <print>
#include <atomic>
#include <thread>

// single precision for now, ~15% faster than double, but double precision is an option for maximum accuracy
using float_T = float;

const float_T epsilon = 1e-6;

inline bool implies(bool a, bool b){
    return !a || b;
}

struct Vec2{
    float_T x,y;

    // see Vec3 for details on these operators

    Vec2 operator+(const Vec2& other) const{
        return Vec2(x+other.x, y+other.y);
    }

    Vec2 operator-(const Vec2& other) const{
        return Vec2(x-other.x, y-other.y);
    }

    Vec2 operator*(float_T scalar) const{
        return Vec2(x*scalar, y*scalar);
    }

    Vec2 operator/(float_T scalar) const{
        return Vec2(x/scalar, y/scalar);
    }

    friend Vec2 operator*(float_T scalar, const Vec2& vec) {
        return vec * scalar;
    }

    float_T dot(const Vec2& other) const{
        return x*other.x + y*other.y;
    }

    float_T length() const{
        return std::sqrt(dot(*this));
    }

    Vec2 normalized() const{
        return *this / length();
    }
};

struct Vec3{
    float_T x,y,z;

    Vec3(float_T x, float_T y, float_T z) : x(x), y(y), z(z){ }

    explicit Vec3(float_T uniform) : x(uniform), y(uniform), z(uniform){ }

    Vec3 operator+(const Vec3& other) const{
        return Vec3(x+other.x, y+other.y, z+other.z);
    }

    Vec3 operator-(const Vec3& other) const{
        return Vec3(x-other.x, y-other.y, z-other.z);
    }

    // scalar multiplication/division
    Vec3 operator*(float_T scalar) const{
        return Vec3(x*scalar, y*scalar, z*scalar);
    }

    // Friend function to overload for scalar * vector
    friend Vec3 operator*(float_T scalar, const Vec3& vec) {
        return vec * scalar;
    }
    friend Vec3 operator/(float_T scalar, const Vec3& vec) {
        return vec / scalar;
    }

    Vec3 operator/(float_T scalar) const{
        return Vec3(x/scalar, y/scalar, z/scalar);
    }

    // elementwise multiplication/division
    Vec3 operator*(Vec3 other) const{
        return Vec3(x*other.x, y*other.y, z*other.z);
    }

    Vec3 operator/(Vec3 other) const{
        return Vec3(x/other.x, y/other.y, z/other.z);
    }

    bool operator==(const Vec3& other) const{
        // epsilon comparison
        return std::abs(x - other.x) < epsilon &&
            std::abs(y - other.y) < epsilon &&
            std::abs(z - other.z) < epsilon;
    }

    float_T operator[](size_t index) const{
        assert(index < 3 && "Index out of bounds");
        return index == 0 ? x : (index == 1 ? y : z);
    }

    Vec3 operator-() const {
        return *this * -1;
    }

    // state modifying operators
    Vec3& operator+=(const Vec3& other){
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
    Vec3 operator*=(float_T scalar){
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    float_T dot(const Vec3& other) const{
        return x*other.x + y*other.y + z*other.z;
    }

    float_T length() const{
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

    Vec3 clamp(float_T min, float_T max){
        return Vec3(
            std::clamp(x, min, max),
            std::clamp(y, min, max),
            std::clamp(z, min, max)
        );
    }

    Vec3 lerp(const Vec3& other, float_T t) const{
        return *this * (1-t) + other * t;
    }

    Vec3 min(const Vec3& other) const{
        return Vec3(
            std::min(x, other.x),
            std::min(y, other.y),
            std::min(z, other.z)
        );
    }

    Vec3 max(const Vec3& other) const{
        return Vec3(
            std::max(x, other.x),
            std::max(y, other.y),
            std::max(z, other.z)
        );
    }

    float_T distance(const Vec3& other) const{
        return (*this - other).length();
    }
};

struct PPMWriter{
private:
    std::ofstream file;

    std::ofstream::pos_type pixelDataStart;


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
        this->pixelDataStart = file.tellp();
    }

    /// write a single pixel in binary format, pixels are iterated over row by row
    void writePixel(uint8_t r, uint8_t g, uint8_t b){
        file.put(r);
        file.put(g);
        file.put(b);
    }

    void writePixel(Vec3 color){
        assert(color == color.clamp(0,1) && "Color must be in the range [0,1]");
        writePixel(
            static_cast<uint8_t>(color.x * 255),
            static_cast<uint8_t>(color.y * 255),
            static_cast<uint8_t>(color.z * 255)
        );
    }

    void rewind(){
        file.flush();
        file.seekp(pixelDataStart);
    }
};


struct Ray{
    Vec3 origin;
    Vec3 direction;
    Vec3 invDirection;

    // TODO maybe make some kind of "prevent self intersection" constructor to dedupliate some code

    /// assumes that the direction is normalized!
    Ray(Vec3 origin, Vec3 direction, Vec3 invDirection)
        : origin(origin), direction(direction), invDirection(invDirection){
        assert(direction == direction.normalized() && "Ray direction must be normalized");
        assert(invDirection == 1./direction && "Ray inverse direction must be the reciprocal of the direction");
    }

    Ray(Vec3 origin, Vec3 direction) : Ray(origin, direction, 1./direction){}

};

// TODO could use CRTP later, but normal dynamic dispatch is fine for now

struct Camera{
    // TODO think about these
    Vec3 position;
    Vec3 direction;
    Vec3 down; // we're using a right-handed coordinate system, as PPM has (0,0) in the top left, so we want to go down not up
    Vec3 right;
    float_T width;
    uint64_t widthPixels;
    float_T height;
    uint64_t heightPixels;
    float_T exposure; // TODO use

    virtual ~Camera() = default; 

    // TODO maybe experiment with && and std::move to avoid some copies
    Camera(Vec3 position,
           Vec3 lookAt,
           Vec3 up,
           float_T width,
           float_T height,
           float_T exposure) : position(position), direction((lookAt - position).normalized()), down(-(up.normalized())), right(direction.cross(down).normalized()), width(width), widthPixels(std::round(width)), height(height), heightPixels(std::round(height)), exposure(exposure){
        const float_T aspectRatio = width / height;
        imagePlaneDimensions = Vec2(aspectRatio*imagePlaneHeight, imagePlaneHeight);
    }

    /// gets a pixel in pixel screen space, i.e. [0,width]x[0,height]
    /// outputs a ray in world space, i.e. adjusting for the camera's position
    virtual Ray generateRay(Vec2 pixelInScreenSpace) const = 0;

protected:
    float_T imagePlaneHeight = 1.0;
    Vec2 imagePlaneDimensions;

    /// gets a pixel in pixel screen space, i.e. [0,width]x[0,height]
    Vec3 pixelInWorldSpace(Vec2 pixelInScreenSpace) const {
        // so, the pixel is in the range [0,width]x[0,height]
        // we want to map this to [-0.5,0.5]x[-0.5,0.5] in the camera's space
        // which is then mapped to the image plane in world space

        constexpr float_T pixelWidthHeight = 1.0;

        Vec2 pixelCenterInCameraSpace = Vec2(
            (pixelInScreenSpace.x + /* use center of pixel */ pixelWidthHeight/2) / width - 0.5,
            (pixelInScreenSpace.y + pixelWidthHeight/2) / height - 0.5
        );


        // scale to world space scale (not translation yet) by multiplying by the image plane dimensions
        Vec2 pixelScaledByWorldSpace = Vec2(pixelCenterInCameraSpace.x * imagePlaneDimensions.x, pixelCenterInCameraSpace.y * imagePlaneDimensions.y);

        // now after scaling to world space, translate to world space, and add the camera's right/up directions
        Vec3 pixelOrigin = position + right*pixelScaledByWorldSpace.x + down*pixelScaledByWorldSpace.y;
        return pixelOrigin;
    }
};

struct OrthographicCamera : public Camera{

    OrthographicCamera(Vec3 position, Vec3 direction, Vec3 up, float_T width, float_T height, float_T exposure)
        : Camera(position, direction, up, width, height, exposure){ }

    virtual Ray generateRay(Vec2 pixelInScreenSpace) const override{
        // for an orthographic camera, basically just shoot a ray in the look direction, through the pixel center
        return Ray(
            pixelInWorldSpace(pixelInScreenSpace),
            direction
        );
    }
};

struct PinholePerspectiveCamera : public Camera{
    float_T fovDegrees;

    PinholePerspectiveCamera(
        Vec3 position,
        Vec3 direction,
        Vec3 up,
        float_T fovDegrees,
        float_T width,
        float_T height,
        float_T exposure)
        : Camera(position, direction, up, width, height, exposure),
          fovDegrees(fovDegrees) {
        // TODO something about this is off I think, the image seems a little bit stretched
        // TODO I think part of it is the assumed 1 unit distance to the image plane
        // Calculate image plane height based on FOV and set image plane dimensions
        const float_T verticalFOVRad = fovDegrees * (M_PI / 180.0); // Convert FOV to radians
        imagePlaneHeight = 2. * tan(verticalFOVRad / 2.); // Distance to image plane is 1 unit
        const float_T aspectRatio = width / height;
        imagePlaneDimensions = Vec2(imagePlaneHeight * aspectRatio, imagePlaneHeight);
    }

    /// gets a pixel in pixel screen space, i.e. [0,width]x[0,height]
    /// outputs a ray in world space, i.e. adjusting for the camera's position
    /// Generates a ray from the camera position through the specified pixel
    virtual Ray generateRay(Vec2 pixelInScreenSpace) const override {
        // Use pixelInWorldSpace to get the point on the image plane in world space
        Vec3 pointOnImagePlane = pixelInWorldSpace(pixelInScreenSpace) + /* place image plane 1 unit away from camera */ direction;

        // Calculate ray direction from camera position to point on image plane
        Vec3 rayDirection = (pointOnImagePlane - position).normalized();

        return Ray(position, rayDirection);
    }
};

struct Color8Bit{
    uint8_t r,g,b;
};

struct Texture{
    uint32_t width, height;
    // dont use vectors, would waste memory
    // TODO check vs performance benefit of not having to convert to Vec3 at runtime
    std::vector<Color8Bit> pixels;

    Texture(uint32_t width, uint32_t height, std::vector<Color8Bit> pixels)
        : width(width), height(height), pixels(pixels){
        assert(width * height == pixels.size() && "Texture dimensions don't match pixel count");
    }

    /// initialize an empty texture to be filled later
    Texture(uint32_t width, uint32_t height)
        : width(width), height(height), pixels(width*height){ }

    void fillUninitialized(std::vector<Color8Bit>&& pixels){
        assert(width * height == pixels.size() && "Texture dimensions don't match pixel count");
        this->pixels = std::move(pixels);
    }

    Vec3 colorAt(const Vec2& textureCoords) const{
        // wrap around
        float_T x = std::fmod(textureCoords.x, 1.);
        float_T y = std::fmod(textureCoords.y, 1.);

        // scale to pixel space
        uint32_t pixelX = static_cast<uint32_t>(x * width);
        uint32_t pixelY = static_cast<uint32_t>(y * height);

        auto pixel = pixels[pixelY * width + pixelX];
        return Vec3(pixel.r / 255., pixel.g / 255., pixel.b / 255.);
    }
};

struct PhongMaterial {
    Vec3 diffuseColor;
    Vec3 specularColor;
    float_T ks,kd;
    uint64_t specularExponent;
    std::optional<float_T> reflectivity;
    std::optional<float_T> refractiveIndex;
    // TODO somehow indicate that this is not used for phong (and rename this material accordingly)
    std::optional<Vec3> emissionColor;
    // share textures to reduce memory usage
    std::optional<std::shared_ptr<Texture>> texture;

    PhongMaterial(
            Vec3 diffuseColor,
            Vec3 specularColor,
            float_T ks,
            float_T kd,
            uint64_t specularExponent,
            std::optional<float_T> reflectivity,
            std::optional<float_T> refractiveIndex,
            std::optional<Vec3> emissionColor,
            std::optional<std::shared_ptr<Texture>> texture)
        : diffuseColor(diffuseColor), specularColor(specularColor), ks(ks), kd(kd), specularExponent(specularExponent), reflectivity(reflectivity), refractiveIndex(refractiveIndex), emissionColor(emissionColor), texture(texture){ }

    /// if the material has a texture, get the diffuse color at the given texture coordinates,
    /// otherwise just return the diffuse color
    /// The texture coordinates here need to be interpolated from the vertices of e.g. the triangle
    Vec3 diffuseColorAtTextureCoords(const Vec2& textureCoords) const {
        if(texture.has_value())
            return (*texture)->colorAt(textureCoords);
        else
            return diffuseColor;
    }

    /// returns the reflected color at the given texture coordinates for the given incomnig/outgoing ray pair
    /// the diffuse part of this does *not multiply* by PI for the lambertian BRDF, as this is accounted for by *not dividing* by PI in the pathtracing shading function
    /// note that the ray direction of the incoming ray is actually away from the intersection point, and the outgoing ray is towards the intersection point, because
    /// we're tracing from the camera!
    Vec3 pathtracingBRDF(const Vec2& textureCoords, const Ray& incomingRay, const Ray& outgoingRay, const Vec3& surfaceNormal) const {
        // for now, just diffuse 
        return diffuseColorAtTextureCoords(textureCoords) * kd;
    }

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
    const Ray* incomingRay;
    // TODO instead of storing the intersection point, could store the distance along the ray
    Vec3 point;
    Vec3 surfaceNormal;
    const PhongMaterial* material;
    Vec2 textureCoords;

    Intersection(const Ray* incomingray, Vec3 point, Vec3 surfaceNormal, const PhongMaterial* material, Vec2 textureCoords)
        : incomingRay(incomingray), point(point), surfaceNormal(surfaceNormal), material(material), textureCoords(textureCoords){
        assert(surfaceNormal == surfaceNormal.normalized() && "Surface normal must be normalized");
    }

    float_T distance() const{
        return (point - incomingRay->origin).length();
    }


};

inline PhongMaterial givenMaterialOrDefault(std::optional<PhongMaterial> material){
    if(material.has_value())
        return std::move(material.value());
    else
        // default material, because the json format allows for no material to be specified
        return PhongMaterial(Vec3(1,1,1), Vec3(1,1,1), 0.5, 0.5, 32, 0.0, 0.0, std::nullopt, std::nullopt);
}

struct Sphere {
    Vec3 center;
    float_T radius;
    PhongMaterial material;

    Sphere(Vec3 center, float_T radius, std::optional<PhongMaterial> material)
        : center(center), radius(radius), material(givenMaterialOrDefault(std::move(material))){ }

    std::optional<Intersection> intersect(const Ray& ray) const {
        // mostly generated by chatgpt

        // Vector from the ray's origin to the sphere's center
        Vec3 oc = ray.origin - center;

        // Coefficients of the quadratic equation (a*t^2 + b*t + c = 0)
        float_T a = ray.direction.dot(ray.direction);               // a = D•D
        float_T b = 2.0 * oc.dot(ray.direction);                    // b = 2 * oc•D
        float_T c = oc.dot(oc) - radius * radius;                   // c = (oc•oc - r^2)

        // Discriminant of the quadratic equation
        float_T discriminant = b * b - 4 * a * c;

        // No intersection if the discriminant is negative
        if (discriminant < 0) {
            return std::nullopt;
        }

        // Calculate the two intersection distances along the ray
        float_T sqrtDiscriminant = std::sqrt(discriminant);
        float_T t1 = (-b - sqrtDiscriminant) / (2.0 * a);
        float_T t2 = (-b + sqrtDiscriminant) / (2.0 * a);


        // Choose the closest intersection point in front of the ray origin
        float_T t = (t1 > 0) ? t1 : ((t2 > 0) ? t2 : -1);
        // if both are behind the ray, return no intersection
        if (t < 0) {
            return std::nullopt;
        }

        // Calculate intersection details
        Vec3 intersectionPoint = ray.origin + ray.direction * t;
        Vec3 intersectionNormal = (intersectionPoint - center).normalized();

        // Calculate texture coordinates
        Vec3 localPoint = (intersectionPoint - center) / radius; // Normalize to unit sphere

        // Calculate spherical coordinates
        float_T theta = std::atan2(localPoint.z, localPoint.x); // Longitude
        float_T phi = std::acos(localPoint.y);                  // Latitude

        // Map spherical coordinates to texture coordinates (u, v)
        float_T u = (theta + M_PI) / (2 * M_PI);
        float_T v = phi / M_PI;

        return Intersection(&ray, std::move(intersectionPoint), std::move(intersectionNormal), &material, Vec2(u, v));
    }

    /// equality (explicitly do not check for material, because two spheres with the exact same position and geometry should not have different materials; this is the same for all other shapes)
    bool operator==(const Sphere& other) const {
        return center == other.center && radius == other.radius;
    }
};

struct Cylinder {
    Vec3 center;
    float_T radius;
    float_T eachSideHeight;
    Vec3 axis;
    PhongMaterial material;

    Cylinder(Vec3 center, float_T radius, float_T height, Vec3 axis, std::optional<PhongMaterial> material)
        : center(center), radius(radius), eachSideHeight(height), axis(axis), material(givenMaterialOrDefault(std::move(material))){ }

    std::optional<Intersection> intersect(const Ray& ray) const {
        // mostly generated by chatgpt

        Vec3 d = ray.direction - axis * ray.direction.dot(axis);  // Projected ray direction onto the cylinder's plane
        Vec3 oc = ray.origin - center;
        Vec3 oc_proj = oc - axis * oc.dot(axis);                  // Projected ray origin onto the cylinder's plane

        float_T a = d.dot(d);
        float_T b = 2. * d.dot(oc_proj);
        float_T c = oc_proj.dot(oc_proj) - radius * radius;
        std::optional<Intersection> closestIntersection = std::nullopt;

        // Quadratic discriminant for side wall intersection
        float_T discriminant = b * b - 4 * a * c;
        if (discriminant >= 0) {
            float_T sqrtDiscriminant = std::sqrt(discriminant);
            for (float_T t : { (-b - sqrtDiscriminant) / (2.0 * a), (-b + sqrtDiscriminant) / (2.0 * a) }) {
                if (t < 0) continue;

                Vec3 point = ray.origin + ray.direction * t;
                Vec3 localPoint = point - center;
                float_T projectionOnAxis = localPoint.dot(axis);

                // Check if intersection point is within height limits of the cylinder
                if (projectionOnAxis >= -eachSideHeight && projectionOnAxis <= eachSideHeight) {
                    Vec3 normal = (localPoint - axis * projectionOnAxis).normalized();
                    Intersection intersection(&ray, point, normal, &material, textCoordsOfSideIntersection(point));


                    // Update closest intersection
                    if (!closestIntersection || t < (closestIntersection->point - ray.origin).length()) {
                        closestIntersection = intersection;
                    }
                }
            }
        }

        auto checkCapIntersection = [&](const Vec3& capCenter, const Vec3& capNormal) -> std::optional<Intersection> {
            float_T denom = ray.direction.dot(capNormal);
            if (std::abs(denom) < 1e-6) return std::nullopt;

            float_T tCap = (capCenter - ray.origin).dot(capNormal) / denom;
            if (tCap < 0) return std::nullopt;

            Vec3 point = ray.origin + ray.direction * tCap;
            if ((point - capCenter).length() <= radius) {  // Check if within radius of cap
                Intersection intersection(&ray, point, capNormal, &material, textCoordsOfCapIntersection(point));
                return intersection;
            }
            return std::nullopt;
        };

        // Check intersections with the base and top caps
        for (auto& cap : { std::make_pair(center - axis * eachSideHeight, -axis), 
                std::make_pair(center + axis * eachSideHeight, axis) }) {
            if (auto capIntersection = checkCapIntersection(cap.first, cap.second); capIntersection) {
                float_T capDistance = (capIntersection->point - ray.origin).length();
                if (!closestIntersection || capDistance < (closestIntersection->point - ray.origin).length()) {
                    closestIntersection = capIntersection;
                }
            }
        }

        return closestIntersection;
    }

    bool operator==(const Cylinder& other) const {
        return center == other.center && radius == other.radius && eachSideHeight == other.eachSideHeight && axis == other.axis;
    }

private:
    Vec2 textCoordsOfSideIntersection(const Vec3& intersectionPoint) const {
        Vec3 baseToIntersection = intersectionPoint - (center - axis * eachSideHeight);
        float_T vPosAlongAxis = baseToIntersection.dot(axis);        
        float_T v = vPosAlongAxis / (2 * eachSideHeight);  // Map height position to v in [0, 1]

        Vec3 circumferentialDir = (baseToIntersection - axis * vPosAlongAxis).normalized();
        float_T theta = std::atan2(circumferentialDir.z, circumferentialDir.x);        
        float_T u = (theta + M_PI) / (2 * M_PI);  // Map angle to u in [0, 1]

        return Vec2(u,v);
    }

    Vec2 textCoordsOfCapIntersection(const Vec3& intersectionPoint) const {
        // Determine which cap (top or bottom) we're on based on axis direction and height
        Vec3 capCenter = intersectionPoint.dot(axis) > 0 ? (center + axis * eachSideHeight) : (center - axis * eachSideHeight);
        Vec3 localCapPoint = intersectionPoint - capCenter;

        // Map `localCapPoint` to polar coordinates within the cap radius
        float_T r = localCapPoint.length() / radius;  // Distance from center mapped to [0, 1]
        float_T capTheta = std::atan2(localCapPoint.z, localCapPoint.x);

        float_T u = 0.5 + r * std::cos(capTheta) / 2;  // Map radial distance and angle to texture u
        float_T v = 0.5 + r * std::sin(capTheta) / 2;  // Map radial distance and angle to texture v

        return Vec2(u,v);
    }
};

struct Triangle {
    Vec3 v0,v1,v2;
    // TODO could try deduplicating materials for triangle objects later on, for big meshes that all have the same material
    PhongMaterial material;
    // these are only valid if the material has a texture
    Vec2 texCoordv0, texCoordv1, texCoordv2;

    // TODO could precompute bounding box or mins/maxes for faster building of the BVH

    Vec3 normal = (v1-v0).cross(v2-v0).normalized();
    Triangle(Vec3 v0, Vec3 v1, Vec3 v2, std::optional<PhongMaterial> material, Vec2 texCoordv0, Vec2 texCoordv1, Vec2 texCoordv2)
        : v0(v0), v1(v1), v2(v2), material(givenMaterialOrDefault(std::move(material))), texCoordv0(texCoordv0), texCoordv1(texCoordv1), texCoordv2(texCoordv2){ }

    Vec3 faceNormal() const{
        // TODO in the future, could interpolate this with the rest of the mesh
        return normal;
    }

    // TODO normal vector interpolation at any point for smooth shading (requires knowledge of the rest of the mesh)

    std::optional<Intersection> intersect(const Ray& ray) const {
        // Möller–Trumbore intersection
        // mostly generated by chatgpt
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        Vec3 h = ray.direction.cross(edge2);
        float_T a = edge1.dot(h);

        // If a is near zero, the ray is parallel to the triangle
        if (std::abs(a) < epsilon) return std::nullopt;

        float_T f = 1.0 / a;
        Vec3 s = ray.origin - v0;
        float_T u = f * s.dot(h);

        // Check if the intersection is outside the triangle
        if (u < 0.0 || u > 1.0) return std::nullopt;

        Vec3 q = s.cross(edge1);
        float_T v = f * ray.direction.dot(q);

        // Check if the intersection is outside the triangle
        if (v < 0.0 || u + v > 1.0) return std::nullopt;

        // Calculate the distance along the ray to the intersection point
        float_T t = f * edge2.dot(q);

        // Only accept intersections that are in front of the ray origin
        if (t > epsilon) {
            Vec3 intersectionPoint = ray.origin + ray.direction * t;
            Vec3 normal = faceNormal();  // Use the constant normal for the triangle

            // Ensure the normal points against the ray's direction,
            // we want to make sure that backfaces look like frontfaces
            // TODO I think this makes stuff righter, in particular shadow calculations for flipped normals
            if (normal.dot(ray.direction) > 0) {
                normal = -normal;
            }

            // interpolate texture coordinates
            // calculate barycentric coordinate `w`
            float_T w = 1. - u - v;

            // interpolate the texture coordinates using barycentric weights
            Vec2 interpolatedTexCoord = texCoordv0 * w + texCoordv1 * u + texCoordv2 * v;

            return Intersection(&ray, intersectionPoint, normal, &material, interpolatedTexCoord);
        }

        return std::nullopt;
    }
    
    bool operator==(const Triangle& other) const{
        return v0 == other.v0 && v1 == other.v1 && v2 == other.v2;
    }
};

struct SceneObject {
    std::variant<Triangle, Sphere, Cylinder> variant;

    std::optional<Intersection> intersect(const Ray& ray) const {
        return std::visit([&](auto&& object){
            return object.intersect(ray);
        }, variant);
    }

    bool operator==(const SceneObject& other) const{
        return variant == other.variant;
    }
};

/// lightweight scene object reference
/// (basically just an index into the scene's object list)
struct SceneObjectReference{
    size_t index;

    explicit SceneObjectReference(size_t index)
        : index(index){ }

    /// finds the index of the object in the list
    SceneObjectReference(const SceneObject& object, const std::vector<SceneObject>& objects){
        index = std::distance(objects.begin(), std::find_if(objects.begin(), objects.end(), [&](const SceneObject& other){
            if constexpr(std::is_same_v<decltype(object), decltype(other)>){
                return object == other;
            }
            return false;
        }));
    }

    SceneObject& dereference(auto& scene){
        return scene.objects[index];
    }

    SceneObject& dereference(std::vector<SceneObject>& objects){
        return objects[index];
    }
};

struct BoundingBox{
    Vec3 min, max;

    BoundingBox() :
          min(Vec3(std::numeric_limits<float_T>::max()))
        , max(Vec3(-std::numeric_limits<float_T>::max())) 
    { }

    /// assumes min and max are actually <= each other, componentwise
    BoundingBox(Vec3 min, Vec3 max)
        : min(min), max(max){
        assert(min.x <= max.x && min.y <= max.y && min.z <= max.z && "Trying to construct invalid bounding box");
    }

    // TODO look at the min/max 0 things, there has to be a better way
    explicit BoundingBox(SceneObject object): min(0.), max(0.){
        std::visit([this](auto&& object){
            using T = std::decay_t<decltype(object)>;
            if constexpr(std::is_same_v<T, Triangle>){
                min = Vec3(
                    std::min({object.v0.x, object.v1.x, object.v2.x}),
                    std::min({object.v0.y, object.v1.y, object.v2.y}),
                    std::min({object.v0.z, object.v1.z, object.v2.z})
                );
                max = Vec3(
                    std::max({object.v0.x, object.v1.x, object.v2.x}),
                    std::max({object.v0.y, object.v1.y, object.v2.y}),
                    std::max({object.v0.z, object.v1.z, object.v2.z})
                );
            }else if constexpr(std::is_same_v<T, Sphere>){
                min = object.center - Vec3(object.radius);
                max = object.center + Vec3(object.radius);
            }else if constexpr(std::is_same_v<T, Cylinder>){
                // cylinder can be encompassed in a box with one corner at one side of the bottom cap, and the other at the other side of the top cap
                const Vec3 bottomCapCenter = object.center - object.axis * object.eachSideHeight;
                const Vec3 topCapCenter = object.center + object.axis * object.eachSideHeight;

                // we'll be shifting points along the 2 "cap" axes (by the radius), so mask out the "height" axis of the cylinder
                const Vec3 axisMask = Vec3(1.) - object.axis;
                // and then invert the axis for one corner, and use it directly for the other
                const Vec3 bottomCapCorner = bottomCapCenter - axisMask * Vec3(object.radius);
                const Vec3 topCapOppositeCorner = topCapCenter + axisMask * Vec3(object.radius);

                min = bottomCapCorner;
                max = topCapOppositeCorner;
            }else{
                static_assert(false, "Unexpected object type");
            }
        }, object.variant);
        assert(min.x <= max.x && min.y <= max.y && min.z <= max.z && "Internal error: invalid bounding box constructed");
    }

    Vec3 center() const { 
        return (min + max) * 0.5; 
    }

    Vec3 extent() const { 
        return max - min; 
    }

    BoundingBox merge(const BoundingBox& other) const {
        return BoundingBox(
            min.min(other.min),
            max.max(other.max)
        );
    }

    bool contains(const Vec3& point) const {
        return point.x >= min.x && point.x <= max.x &&
               point.y >= min.y && point.y <= max.y &&
               point.z >= min.z && point.z <= max.z;
    }

    /*
       TODO probably either remove or merge this with the intersects function
    float_T intersection_distance(const Ray& ray) const {
        Vec3 invDir = Vec3(1.) / ray.direction;

        Vec3 t0 = (min - ray.origin) * invDir;
        Vec3 t1 = (max - ray.origin) * invDir;

        Vec3 tmin = t0.min(t1);
        Vec3 tmax = t0.max(t1);

        float_T tenter = std::max(std::max(tmin.x, tmin.y), tmin.z);
        float_T texit = std::min(std::min(tmax.x, tmax.y), tmax.z);

        return tenter <= texit && texit >= 0 ? tenter : std::numeric_limits<float_T>::max();
    }
    */

    bool intersects(const Ray& ray) const {
        Vec3 invDir = Vec3(1.) / ray.direction;
        
        Vec3 t0 = (min - ray.origin) * invDir;
        Vec3 t1 = (max - ray.origin) * invDir;
        
        Vec3 tmin = t0.min(t1);
        Vec3 tmax = t0.max(t1);
        
        float_T tenter = std::max(std::max(tmin.x, tmin.y), tmin.z);
        float_T texit = std::min(std::min(tmax.x, tmax.y), tmax.z);
        
        return tenter <= texit && texit >= 0;
    }

    bool overlaps(const BoundingBox& other) const {
        return min.x <= other.max.x && max.x >= other.min.x &&
               min.y <= other.max.y && max.y >= other.min.y &&
               min.z <= other.max.z && max.z >= other.min.z;
    }

    float_T surface_area() const {
        Vec3 d = extent();
        return 2. * (d.x * d.y + d.y * d.z + d.z * d.x);
    }
};

struct ObjectRange{
    /// left-inclusive, right-exclusive
    std::pair<size_t, size_t> objectRange;

    // allow implicit conversion
    ObjectRange(std::pair<size_t, size_t> objectRange)
        : objectRange(objectRange){ }

    ObjectRange(size_t first, size_t last)
        : objectRange(std::make_pair(first, last)){ }

    size_t size() const{
        return objectRange.second - objectRange.first;
    }

    bool empty() const{
        return objectRange.first == objectRange.second;
    }

    bool operator==(const ObjectRange& other) const{
        return objectRange == other.objectRange;
    }

    size_t begin() const{
        return objectRange.first;
    }

    size_t end() const{
        return objectRange.second;
    }

    // implicit conversion to pair
    operator std::pair<size_t, size_t>() const{
        return objectRange;
    }
};

/// bounding volume hierarchy
struct BVHNode{
    BoundingBox bounds;
    std::unique_ptr<BVHNode> left, right;
    /// range of objects in the scene object list that this node represents
    ObjectRange objectRange;
    // -> from this, it is clear that objects can only overlap "linearly", i.e. only if they are adjacent in the list

    static constexpr size_t MAX_DEPTH = 16;
    // TODO maybe remove in future, or make an option; not in use currently, because even though it reduces memory usage, it doesn't improve performance
    //static constexpr size_t MIN_OBJECTS = 4;

public:
    /// REORDERS THE OBJECTS VECTOR
    /// but does not store it anywhere, the objects explicitly live outside the BVH
    BVHNode(ObjectRange objectRangeP, std::vector<SceneObject>& objects, uint32_t depth = 0) 
        : objectRange(std::move(objectRangeP))
    {
        assert(objectRange.begin() <= objectRange.end() && "Invalid objectRange");
        assert(objectRange.end() <= objects.size() && "objectRange exceeds object vector");

        bounds = boundsFromObjectRange(objectRange, objects);
        
        if (depth >= MAX_DEPTH)
            return;

        // Find the axis with greatest extent
        Vec3 extent = bounds.extent();
        int splitAxis = 0;
        if (extent.y > extent.x) splitAxis = 1;
        if (extent.z > extent[splitAxis]) splitAxis = 2;

        auto begin = objects.begin() + objectRange.begin();
        auto end = objects.begin() + objectRange.end();
        std::nth_element(begin, begin + (end - begin)/2, end,
            [splitAxis](const SceneObject& a, const SceneObject& b) {
                return computeCentroid(a)[splitAxis] < computeCentroid(b)[splitAxis];
            }
        );
        size_t midIndex = objectRange.begin() + (objectRange.end() - objectRange.begin()) / 2;
        
        if (midIndex > objectRange.begin() && midIndex < objectRange.end()) {
            left = std::make_unique<BVHNode>(ObjectRange(objectRange.begin(), midIndex), objects, depth + 1);
            right = std::make_unique<BVHNode>(ObjectRange(midIndex, objectRange.end()), objects, depth + 1);
        }
    }

    std::optional<Intersection> intersect(const Ray& ray, const std::vector<SceneObject>& objects) const {
        if (!bounds.intersects(ray))
            return std::nullopt;

        if (isLeaf()) {
            auto closestIntersection = std::optional<Intersection>();

            for (auto i = objectRange.begin(); i < objectRange.end(); ++i)
                if (auto intersection = objects[i].intersect(ray))
                    if(!closestIntersection.has_value() || intersection->distance() < closestIntersection->distance())
                        closestIntersection = *intersection;

            return closestIntersection;
        }

        // here, we're just checking both boxes
        // TODO but we could check the least number of nodes by:
        // a) checking the closer box first
        // b) only checking the other box if the boxes overlap, or if the closer box has no intersection

        auto leftIntersection = left->intersect(ray, objects);
        auto rightIntersection = right->intersect(ray, objects);
        if(leftIntersection.has_value() && rightIntersection.has_value())
            return leftIntersection->distance() < rightIntersection->distance() ? leftIntersection : rightIntersection;
        else if(leftIntersection.has_value())
            // don't std::move this, to allow copy elision
            return leftIntersection;
        else
            // if the right intersection is empty, this will also return nullopt
            return rightIntersection;
    }

    bool isLeaf() const { 
        return !left && !right; 
    }

    // === The following public methods are all for debugging purposes ===

    void verifyBVH(const std::vector<SceneObject>& objects) const {
#ifdef NDEBUG
        std::println(stderr, "verifyBVH should only be called in debug mode");
        std::abort();
#endif
        // Verify range validity
        assert(objectRange.begin() <= objectRange.end() && "Invalid objectRange");
        assert(objectRange.end() <= objects.size() && "objectRange exceeds vector size");

        if (!isLeaf()) {
            // Verify children exist
            // (this is a bit stupid, because isLeaf would prob fail first, but just in case)
            assert(left && right && "Non-leaf node missing children");

            // Recursively verify children
            left->verifyBVH(objects);
            right->verifyBVH(objects);

            // Verify child objectRanges overlap linearly, i.e.
        }
    }

    void recursivelyCollectIntersectedBoxes(const Ray& ray, std::vector<std::pair<BoundingBox, int>>& boxes, int depth = 0) const {
        if (bounds.intersects(ray)) {
            boxes.push_back({bounds, depth});
            
            if (!isLeaf()) {
                left->recursivelyCollectIntersectedBoxes(ray, boxes, depth + 1);
                right->recursivelyCollectIntersectedBoxes(ray, boxes, depth + 1);
            }
        }
    }

    uint64_t numNodes() const {
        if (isLeaf())
            return 1;
        return 1 + left->numNodes() + right->numNodes();
    }

private:
    static Vec3 computeCentroid(const SceneObject& object) {
        return std::visit([](const auto& obj) -> Vec3 {
            using T = std::decay_t<decltype(obj)>;
            if constexpr (std::is_same_v<T, Triangle>) {
                return (obj.v0 + obj.v1 + obj.v2) / 3.;
            } else if constexpr (std::is_same_v<T, Sphere>) {
                return obj.center;
            } else if constexpr (std::is_same_v<T, Cylinder>) {
                return obj.center;
            } else{
                static_assert(false, "Unexpected object type");
            }
        }, object.variant);
    }

    static BoundingBox boundsFromObjectRange(ObjectRange range, 
                                    const std::vector<SceneObject>& objects) {
        BoundingBox bounds;
        for (size_t i = range.begin(); i < range.end(); ++i) {
            bounds = bounds.merge(BoundingBox(objects[i]));
        }
        return bounds;
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
    DEBUG_BVH,
    DEBUG_NORMALS,
    PATHTRACE,
    // continue rendering after rendering the first frame, and average the results
    PATHTRACE_INCREMENTAL,
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
    std::vector<SceneObject> objects;

    uint32_t pathtracingSamplesPerPixel;

    //Scene(uint32_t nBounces, RenderMode renderMode, std::unique_ptr<Camera> camera, Vec3 backgroundColor, std::vector<Triangle> triangles, std::vector<Sphere> spheres, std::vector<Cylinder> cylinders)
    //    : nBounces(nBounces), renderMode(renderMode), camera(std::move(camera)), backgroundColor(backgroundColor), triangles(triangles), spheres(spheres), cylinders(cylinders){ }
    Scene(uint32_t nBounces, RenderMode renderMode, std::unique_ptr<Camera> camera, Vec3 backgroundColor, std::vector<PointLight> lights, std::vector<SceneObject> objects, uint32_t pathtracingSamplesPerPixel)
        : nBounces(nBounces), renderMode(renderMode), camera(std::move(camera)), backgroundColor(backgroundColor), lights(std::move(lights)), objects(std::move(objects)), pathtracingSamplesPerPixel(pathtracingSamplesPerPixel){ }
};

struct Renderer{
    Scene scene;
    PPMWriter writer;
    // ordered the same way a PPM file is, row by row
    // use a buffer instead of writing to the file immediately to be able to do it in parallel
    std::vector<Vec3> hdrPixelBuffer;

    BVHNode bvh;


    Renderer(Scene&& scene, std::string_view outputFilePath)
        : scene(std::move(scene)),
          writer(outputFilePath, this->scene.camera->widthPixels, this->scene.camera->heightPixels),
          hdrPixelBuffer(this->scene.camera->widthPixels*this->scene.camera->heightPixels, Vec3(0.)),
          // this reorders the objects in the scene to be able to build the BVH
          bvh(BVHNode(ObjectRange(0, this->scene.objects.size()), this->scene.objects))
    {}

    void bufferSpecificPixel(Vec2 pixel, Vec3 color){
        assert(pixel.x >= 0 && pixel.x < scene.camera->width && pixel.y >= 0 && pixel.y < scene.camera->height && "Pixel out of range");
        assert(scene.camera->width * pixel.y + pixel.x < hdrPixelBuffer.size() && "Pixel out of range");
        hdrPixelBuffer[pixel.y * scene.camera->width + pixel.x] = color;
    }

    void writeBufferToFile(){
        writer.rewind();
        // write buffer to file
        assert(hdrPixelBuffer.size() == scene.camera->widthPixels * scene.camera->heightPixels && "Pixel buffer size mismatch");
        for(auto& hdrPixel: hdrPixelBuffer){
            auto ldrPixel = hdrPixel.clamp(0., 1.);
            writer.writePixel(ldrPixel);
        }
    }

    /// shades a single intersection point
    /// outputs an un-tonemapped color, not for immediate display
    Vec3 blinnPhongShading(const Intersection& intersectionToShade, uint32_t bounces = 1, float_T currentIOR = 1.) {
        if (bounces > scene.nBounces)
            return Vec3(0.);

        // material properties
        Vec3 diffuse = intersectionToShade.material->diffuseColorAtTextureCoords(intersectionToShade.textureCoords);
        float_T ambientIntensity = 0.25;
        Vec3 ambient = diffuse * ambientIntensity;
        Vec3 specular = intersectionToShade.material->specularColor;
        float_T ks = intersectionToShade.material->ks;
        float_T specularExponentShinyness = intersectionToShade.material->specularExponent;

        auto isInShadow = [&](const PointLight& light, const Vec3& L) -> bool {
            Vec3 shadowRayOrigin = intersectionToShade.point + L * (100 * epsilon);
            Ray shadowRay(shadowRayOrigin, L);

            if (auto shadowIntersection = traceRayToClosestSceneIntersection(shadowRay)) {
                return (shadowIntersection->point - intersectionToShade.point).length() < (light.position - intersectionToShade.point).length() - 100 * epsilon;
            }
            return false;
        };

        // Helper function to calculate specular highlights
        auto calculateSpecularHighlights = [&]() -> Vec3 {
            Vec3 specularSum(0.);

            for(const auto& light: scene.lights) {
                Vec3 L = (light.position - intersectionToShade.point).normalized();
                if(isInShadow(light, L))
                    continue;

                // TODO probably doulbe normalize here right?
                Vec3 V = -intersectionToShade.incomingRay->direction.normalized();
                Vec3 H = (L + V).normalized();
                float_T spec = std::pow(std::max(intersectionToShade.surfaceNormal.dot(H), (float_T) 0.), 
                        specularExponentShinyness);
                specularSum += specular * spec * light.intensityPerColor * ks;
            }
            return specularSum;
        };

        // Handle transparent (refractive) materials differently
        if (intersectionToShade.material->refractiveIndex.has_value()) {
            // make sure to still have specular highlights on transparent objects
            // these would be weighted by the objects transmissiveness, but as that doesnt exist, just add them for now
            Vec3 finalColor = calculateSpecularHighlights();
            float_T materialIOR = *intersectionToShade.material->refractiveIndex;

            bool entering = intersectionToShade.incomingRay->direction.dot(intersectionToShade.surfaceNormal) < 0;
            Vec3 normal = entering ? intersectionToShade.surfaceNormal : -intersectionToShade.surfaceNormal;
            float_T etaRatio = entering ? currentIOR / materialIOR : materialIOR / currentIOR;

            float_T cosTheta_i = -normal.dot(intersectionToShade.incomingRay->direction);
            float_T sinTheta_t_squared = etaRatio * etaRatio * (1. - cosTheta_i * cosTheta_i);

            if (sinTheta_t_squared <= 1.) {
                float_T cosTheta_t = std::sqrt(1. - sinTheta_t_squared);
                Vec3 refractedDir = etaRatio * intersectionToShade.incomingRay->direction + 
                    (etaRatio * cosTheta_i - cosTheta_t) * normal;

                Ray refractedRay(intersectionToShade.point + refractedDir * (10 * epsilon), refractedDir);
                // TODO refracting exiting being air doesnt really work I think, it should somehow be dependent on whether the intersection is inside the current intersected object or outside
                float_T nextIOR = entering ? materialIOR : 1.;

                Vec3 refractedColor = scene.backgroundColor;
                if (auto refractedIntersection = traceRayToClosestSceneIntersection(refractedRay)) {
                    refractedColor = blinnPhongShading(*refractedIntersection, bounces + 1, nextIOR);
                }

                // tint the refraction by the diffuse color, to be able to make e.g. red glass
                // TODO could add something like a density parameter to the material, to decide how much to tint, but for now, thats just encoded in the diffuse color
                finalColor += refractedColor * diffuse;
                return finalColor;
            }else{
                // Total internal reflection
                // handle by simply going on with the normal lighting nor now
                // TODO to look somewhat realistic, this curently requires the material to be reflective
                //      but thats somewhat consistent with the phong lighting model idea: you're responsible for
                //      a realistic looking material, not the renderer
            }

        }

        // Regular materials: ambient + diffuse + specular + reflection
        Vec3 color(0.);

        // Ambient and diffuse
        color += ambient;  // ambient

        float_T kd = intersectionToShade.material->kd;
        for(const auto& light: scene.lights) {
            Vec3 L = (light.position - intersectionToShade.point).normalized();
            if(isInShadow(light, L))
                continue;

            Vec3 N = intersectionToShade.surfaceNormal;
            float_T diff = std::max(N.dot(L), (float_T) 0.);
            color += diffuse * diff * light.intensityPerColor * kd;  // diffuse
        }

        // Add specular for all materials
        color += calculateSpecularHighlights();

        // Handle reflection if material is reflective
        // TODO could do fresnel here and for refractivity
        if(intersectionToShade.material->reflectivity.has_value()) {
            Vec3 reflectedColor = scene.backgroundColor;
            Vec3 reflectionDir = intersectionToShade.incomingRay->direction - 
                intersectionToShade.surfaceNormal * 2 * 
                (intersectionToShade.incomingRay->direction.dot(intersectionToShade.surfaceNormal));
            Ray reflectionRay(intersectionToShade.point + reflectionDir * (10 * epsilon), reflectionDir);

            if (auto reflectionIntersection = traceRayToClosestSceneIntersection(reflectionRay)) {
                reflectedColor = blinnPhongShading(*reflectionIntersection, bounces + 1, currentIOR);
            }

            color = color.lerp(reflectedColor, *intersectionToShade.material->reflectivity);
        }

        return color;
    }

    /// does *not* clamp the color, this is done in writing the pixel to the buffer
    Vec3 linearToneMapping(Vec3 color){
        return color*scene.camera->exposure * 15.;
    }

    Vec3 gammaCorrect(Vec3 color){
        return Vec3(std::pow(color.x, 1./2.2), std::pow(color.y, 1./2.2), std::pow(color.z, 1./2.2));
    }

    template<bool useBVH = true>
    std::optional<Intersection> traceRayToClosestSceneIntersection(const Ray& ray){
        if constexpr(useBVH){
            return bvh.intersect(ray, scene.objects);
        }else{
            auto closestIntersection = std::optional<Intersection>();
            for(auto& object: scene.objects)
                if(auto intersection = object.intersect(ray))
                    if(!closestIntersection.has_value() || intersection->distance() < closestIntersection->distance())
                        closestIntersection = *intersection;

            return closestIntersection;
        }
    }

    template<RenderMode mode>
    void render(){
        std::println(stderr, "BVH num nodes: {}", bvh.numNodes());
        std::println(stderr, "num objects: {}", bvh.objectRange.size());

        if constexpr(mode == RenderMode::DEBUG_BVH)
            renderDebugBVHToBuffer();
        else if constexpr(mode == RenderMode::DEBUG_NORMALS)
            renderDebugNormalsToBuffer();
        else if constexpr(mode == RenderMode::PATHTRACE || mode == RenderMode::PATHTRACE_INCREMENTAL){
            // until user closes stdin (Ctrl+D)
            if constexpr(mode == RenderMode::PATHTRACE_INCREMENTAL)
                std::println(stderr, "Rendering incrementally. Press Ctrl+D to stop after next render");

            // TODO incremental and immediate still don't seem to have the exact same results for the same sample number

            renderPathtraceToBuffer();
            writeBufferToFile();

            uint32_t samplesPerPixelSoFar = scene.pathtracingSamplesPerPixel;
            std::vector<Vec3> previousBuffer = hdrPixelBuffer;

            std::atomic<bool> stopRendering = false;
            if(mode == RenderMode::PATHTRACE)
                stopRendering = true;

            // start a thread that stops the program when the user closes stdin
            std::jthread stopThread([&stopRendering]{
                char _[2];
                while(!stopRendering && std::fgets(_, sizeof(_), stdin) != nullptr);

                if(!stopRendering){
                    std::println(stderr, "Will stop rendering after this frame");
                    stopRendering = true;
                }
            });

            // TODO this doesnt really work yet
            while (!stopRendering) {
                std::println(stderr, "Frame rendered, {} samples per pixel so far", samplesPerPixelSoFar);

                renderPathtraceToBuffer();
                const uint32_t samplePixelsNow = scene.pathtracingSamplesPerPixel + samplesPerPixelSoFar;
                // average the results
                for(auto [pixel, previousPixel]: std::ranges::views::zip(hdrPixelBuffer, previousBuffer)){
                    pixel = scene.pathtracingSamplesPerPixel*pixel/samplePixelsNow + samplesPerPixelSoFar*previousPixel/samplePixelsNow;
                }
                writeBufferToFile();
                previousBuffer = hdrPixelBuffer;

                samplesPerPixelSoFar = samplePixelsNow;
            } 

            std::println(stderr, "Rendering stopped: Final image has {} samples per pixel", samplesPerPixelSoFar);
        }else{
            // cant try gpu openacc because nvcc doesnt support c++23 :(
            // openacc might not work at all with gcc here, is basically the same time as serial
            //#pragma acc parallel loop
#pragma omp parallel for
            for(uint32_t y = 0; y < scene.camera->heightPixels; y++){
                for(uint32_t x = 0; x < scene.camera->widthPixels; x++){
                    Ray cameraRay = scene.camera->generateRay(Vec2(x, y));

                    auto closestIntersection = traceRayToClosestSceneIntersection(cameraRay);

                    Vec3 pixelColor = scene.backgroundColor;
                    if(closestIntersection.has_value()){
                        if constexpr (mode == RenderMode::BINARY){
                            pixelColor = Vec3(1.0);
                        }else if constexpr (mode == RenderMode::PHONG){
                            pixelColor = blinnPhongShading(*closestIntersection);
                        }else{
                            static_assert(false, "Invalid render mode");
                        }
                    }

                    bufferSpecificPixel(Vec2(x, y), linearToneMapping(pixelColor));
                }
            }
        }
        writeBufferToFile();
    }

    void renderDebugBVHToBuffer() {
        // serially, so we dont have to use atomic accesses on the max intensity
        float_T maxIntensity = 0.;
        for(uint32_t y = 0; y < scene.camera->heightPixels; y++){
            for(uint32_t x = 0; x < scene.camera->widthPixels; x++){
                Ray cameraRay = scene.camera->generateRay(Vec2(x, y));

                std::vector<std::pair<BoundingBox, int>> intersected_boxes;
                bvh.recursivelyCollectIntersectedBoxes(cameraRay, intersected_boxes);

                // just write the size to the buffer for now, and keep track of the max
                float_T intensity = intersected_boxes.size();
                maxIntensity = std::max(maxIntensity, intensity);
                bufferSpecificPixel(Vec2(x, y), Vec3(intensity, intensity, intensity));
            }
        }

        // then go through all of them again and normalize them, mark areas close to max intensity as red
        static constexpr float_T redThreshhold = 0.9;
        for(auto& pixel: hdrPixelBuffer){
            float_T intensity = pixel.x;
            if(intensity > redThreshhold * maxIntensity)
                pixel = Vec3(1.0, 0.0, 0.0);
            else
                pixel = pixel / maxIntensity;
        }
    }

    void renderDebugNormalsToBuffer() {
        for(uint32_t y = 0; y < scene.camera->heightPixels; y++){
            for(uint32_t x = 0; x < scene.camera->widthPixels; x++){
                Ray cameraRay = scene.camera->generateRay(Vec2(x, y));

                if(auto closestIntersection = traceRayToClosestSceneIntersection(cameraRay)){
                    auto normalZeroOne = closestIntersection->surfaceNormal * 0.5 + Vec3(0.5);
                    bufferSpecificPixel(Vec2(x, y), normalZeroOne);
                }
            }
        }
    }

    /// generate a uniformly distributed random float in [0, 1)
    float_T randomFloat() {
        // initial generated by chatgpt, thread safe by me
        // make thread safe by ensuring each thread has its own generator
        thread_local static std::random_device rd;
        thread_local static std::mt19937 gen(rd());
        thread_local static std::uniform_real_distribution<float_T> dis(0., 1.);
        return dis(gen);
    }

    enum ImportanceSamplingTechnique{
        UNIFORM,
        COSINE_WEIGHTED_HEMISPHERE,
    };

    template<ImportanceSamplingTechnique technique>
    Vec3 sampleHemisphere(const Vec3& normal){
        if constexpr (technique == COSINE_WEIGHTED_HEMISPHERE){
            // TODO this isnt tested at all yet

            // cosine weighted hemisphere sampling, to eliminate the dot product from the rendering equation
            // basically the approximation of the integral already divides by cos(theta), so the multiplying by theta
            // in the normal rendering equation gets cancelled out.
            // To get the correct value of dividing by the PDF (cos(theta)/pi), we have to multiply by pi again

            // Generate two random numbers for disk sampling
            float_T r = sqrt(randomFloat());
            float_T theta = 2.0 * M_PI * randomFloat();

            // Convert uniform disk samples to hemisphere samples
            float_T x = r * cos(theta);
            float_T y = r * sin(theta);

            // Project up to hemisphere
            float_T z = sqrt(1.0 - x*x - y*y);

            // Create a coordinate system from the normal
            // TODO that z check is strange
            Vec3 up = (std::abs(normal.z) < (1 - 100 * epsilon)) ? Vec3(0, 0, 1) : Vec3(1, 0, 0);
            Vec3 tangent = up.cross(normal).normalized();
            Vec3 bitangent = normal.cross(tangent);

            // Transform the local hemisphere direction to world space
            // TODO hmmm, this will always give positive results though, right?
            return (tangent * x + bitangent * y + normal * z).normalized();
        }else if constexpr (technique == UNIFORM){
            float_T phi = 2.0 * M_PI * randomFloat();
            float_T z = randomFloat();
            float_T r = std::sqrt(1.0 - z*z);

            // Create basis vectors
            Vec3 up = normal;
            Vec3 right(1, 0, 0);
            if (std::abs(up.y) < std::abs(up.x)) {
                right = Vec3(0, 1, 0);
            }

            Vec3 tangent = up.cross(right);
            Vec3 bitangent = up.cross(tangent);

            Vec3 sample = tangent * (r * std::cos(phi)) + 
                bitangent * (r * std::sin(phi)) + 
                up * z;
            assert(sample == sample.normalized() && "Sample must be on the unit hemisphere");

            return sample;
        } else {
            static_assert(false, "Invalid importance sampling technique");
        }
    }

    template<ImportanceSamplingTechnique samplingTechnique = COSINE_WEIGHTED_HEMISPHERE>
    Vec3 shadePathtraced(const Intersection& intersection, uint32_t bounces = 1){
        if(bounces > scene.nBounces)
            return Vec3(scene.backgroundColor);

        const Vec3 emission = intersection.material->emissionColor.value_or(Vec3(0.));

        // add diffuse incoming light from all directions
        const Vec3 hemisphereSample = sampleHemisphere<samplingTechnique>(intersection.surfaceNormal);

        Ray incomingRay(intersection.point + hemisphereSample * (10 * epsilon), hemisphereSample);

        Vec3 incomingColor = scene.backgroundColor;
        if(auto incomingIntersection = traceRayToClosestSceneIntersection(incomingRay)){
            incomingColor = shadePathtraced<samplingTechnique>(*incomingIntersection, bounces + 1);
        }
        
        const auto brdf = intersection.material->pathtracingBRDF(intersection.textureCoords, incomingRay, *intersection.incomingRay, intersection.surfaceNormal);

        if constexpr(samplingTechnique == COSINE_WEIGHTED_HEMISPHERE){
            return emission + incomingColor * brdf;
        }else if(samplingTechnique == UNIFORM){
            // weight by the cosine of the angle between the normal and the incoming ray, this weighting is already present in the cosine weighted hemisphere sampling
            // the 2. factor comes from dividing by the PDF, which is 1/2pi for uniform sampling. The pi in the diffuse BRDF cancels out with the pi in the PDF
            return emission + 2. * incomingColor * brdf * intersection.surfaceNormal.dot(hemisphereSample);
        }
    }

    void renderPathtraceToBuffer(){
        // dynamic thread scheduling improves performance by ~10% on cornell box with 6 bounces (because some rays terminate early by bouncing into nothing)
#pragma omp parallel for collapse(2) schedule(dynamic, 64)
        for(uint32_t y = 0; y < scene.camera->heightPixels; y++){
            for(uint32_t x = 0; x < scene.camera->widthPixels; x++){
                const Vec2 pixelOrigin = Vec2(x, y);
                Vec3 colorSum = Vec3(0.);
                for(uint32_t sample = 0; sample < scene.pathtracingSamplesPerPixel; sample++){
                    // permute ray randomly/evenly (TODO importance sampling later)

                    // jittered sampling
                    Vec2 permutedPixel = pixelOrigin + Vec2(randomFloat(), randomFloat());

                    Ray cameraRay = scene.camera->generateRay(permutedPixel);


                    if(auto intersection = traceRayToClosestSceneIntersection(cameraRay)){
                        colorSum += shadePathtraced(*intersection);
                    }else{
                        // background color
                        colorSum += scene.backgroundColor;
                    }
                }

                bufferSpecificPixel(Vec2(x, y), linearToneMapping(gammaCorrect(colorSum / scene.pathtracingSamplesPerPixel)));
            }
        }
    }

};
