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

const float_t epsilon = 1e-6;

inline bool implies(bool a, bool b){
    return !a || b;
}

struct Vec2{
    float_t x,y;

    // see Vec3 for details on these operators

    Vec2 operator+(const Vec2& other) const{
        return Vec2(x+other.x, y+other.y);
    }

    Vec2 operator-(const Vec2& other) const{
        return Vec2(x-other.x, y-other.y);
    }

    Vec2 operator*(float_t scalar) const{
        return Vec2(x*scalar, y*scalar);
    }

    friend Vec2 operator*(float scalar, const Vec2& vec) {
        return vec * scalar;
    }
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

    // Friend function to overload for scalar * vector
    friend Vec3 operator*(float scalar, const Vec3& vec) {
        return vec * scalar;
    }

    Vec3 operator/(float_t scalar) const{
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
    Vec3 operator*=(float_t scalar){
        x *= scalar;
        y *= scalar;
        z *= scalar;
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

    Vec3 clamp(float_t min, float_t max){
        return Vec3(
            std::clamp(x, min, max),
            std::clamp(y, min, max),
            std::clamp(z, min, max)
        );
    }

    Vec3 lerp(const Vec3& other, float_t t) const{
        return *this * (1-t) + other * t;
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


    // TODO maybe make some kind of "prevent self intersection" constructor to dedupliate some code

    /// assumes that the direction is normalized!
    Ray(Vec3 origin, Vec3 direction)
        : origin(origin), direction(direction){
        assert(direction == direction.normalized() && "Ray direction must be normalized");
    }
};

// TODO could use CRTP later, but normal dynamic dispatch is fine for now

struct Camera{
    // TODO think about these
    Vec3 position;
    Vec3 direction;
    Vec3 down; // we're using a right-handed coordinate system, as PPM has (0,0) in the top left, so we want to go down not up
    Vec3 right;
    float_t width;
    uint64_t widthPixels;
    float_t height;
    uint64_t heightPixels;
    float_t exposure; // TODO use

    virtual ~Camera() = default; 

    // TODO maybe experiment with && and std::move to avoid some copies
    Camera(Vec3 position,
           Vec3 lookAt,
           Vec3 up,
           float_t width,
           float_t height,
           float_t exposure) : position(position), direction((lookAt - position).normalized()), down(-(up.normalized())), right(direction.cross(down).normalized()), width(width), widthPixels(std::round(width)), height(height), heightPixels(std::round(height)), exposure(exposure){
        const float_t aspectRatio = width / height;
        imagePlaneDimensions = Vec2(aspectRatio*imagePlaneHeight, imagePlaneHeight);
    }

    /// gets a pixel in pixel screen space, i.e. [0,width]x[0,height]
    /// outputs a ray in world space, i.e. adjusting for the camera's position
    virtual Ray generateRay(Vec2 pixelInScreenSpace) const = 0;

protected:
    float_t imagePlaneHeight = 1.0;
    Vec2 imagePlaneDimensions;

    /// gets a pixel in pixel screen space, i.e. [0,width]x[0,height]
    Vec3 pixelInWorldSpace(Vec2 pixelInScreenSpace) const {
        // so, the pixel is in the range [0,width]x[0,height]
        // we want to map this to [-0.5,0.5]x[-0.5,0.5] in the camera's space
        // which is then mapped to the image plane in world space

        constexpr float_t pixelWidthHeight = 1.0;

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

    OrthographicCamera(Vec3 position, Vec3 direction, Vec3 up, float_t width, float_t height, float_t exposure)
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
    float_t fovDegrees;

    PinholePerspectiveCamera(
        Vec3 position,
        Vec3 direction,
        Vec3 up,
        float_t fovDegrees,
        float_t width,
        float_t height,
        float_t exposure)
        : Camera(position, direction, up, width, height, exposure),
          fovDegrees(fovDegrees) {
        // TODO something about this is off I think, the image seems a little bit stretched
        // TODO I think part of it is the assumed 1 unit distance to the image plane
        // Calculate image plane height based on FOV and set image plane dimensions
        const float_t verticalFOVRad = fovDegrees * (M_PI / 180.0); // Convert FOV to radians
        imagePlaneHeight = 2.0f * tan(verticalFOVRad / 2.0f); // Distance to image plane is 1 unit
        const float_t aspectRatio = width / height;
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

    Vec3 colorAt(Vec2 textureCoords) const{
        // wrap around
        float_t x = std::fmod(textureCoords.x, 1.);
        float_t y = std::fmod(textureCoords.y, 1.);

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
    float_t ks,kd;
    uint64_t specularExponent;
    std::optional<float_t> reflectivity;
    std::optional<float_t> refractiveIndex;
    // share textures to reduce memory usage
    std::optional<std::shared_ptr<Texture>> texture;

    PhongMaterial(
            Vec3 diffuseColor,
            Vec3 specularColor,
            float_t ks,
            float_t kd,
            uint64_t specularExponent,
            std::optional<float_t> reflectivity,
            std::optional<float_t> refractiveIndex,
            std::optional<std::shared_ptr<Texture>> texture)
        : diffuseColor(diffuseColor), specularColor(specularColor), ks(ks), kd(kd), specularExponent(specularExponent), reflectivity(reflectivity), refractiveIndex(refractiveIndex), texture(texture){ }

    /// if the material has a texture, get the diffuse color at the given texture coordinates,
    /// otherwise just return the diffuse color
    /// The texture coordinates here need to be interpolated from the vertices of e.g. the triangle
    Vec3 diffuseColorAtTextureCoords(Vec2 textureCoords) const {
        if(texture.has_value())
            return (*texture)->colorAt(textureCoords);
        else
            return diffuseColor;
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
    Ray incomingRay;
    Vec3 point;
    Vec3 surfaceNormal;
    PhongMaterial material;
    Vec2 textureCoords;

    Intersection(Ray incomingray, Vec3 point, Vec3 surfaceNormal, PhongMaterial material, Vec2 textureCoords)
        : incomingRay(incomingray), point(point), surfaceNormal(surfaceNormal), material(material), textureCoords(textureCoords){
        assert(surfaceNormal == surfaceNormal.normalized() && "Surface normal must be normalized");
    }

    float_t distance() const{
        return (point - incomingRay.origin).length();
    }
};

inline PhongMaterial givenMaterialOrDefault(std::optional<PhongMaterial> material){
    if(material.has_value())
        return std::move(material.value());
    else
        // default material, because the json format allows for no material to be specified
        return PhongMaterial(Vec3(1,1,1), Vec3(1,1,1), 0.5, 0.5, 32, 0.0, 0.0, std::nullopt);
}

struct Sphere {
    Vec3 center;
    float_t radius;
    PhongMaterial material;

    Sphere(Vec3 center, float_t radius, std::optional<PhongMaterial> material)
        : center(center), radius(radius), material(givenMaterialOrDefault(std::move(material))){ }

    std::optional<Intersection> intersect(const Ray& ray) {
        // mostly generated by chatgpt

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

        // Calculate texture coordinates
        Vec3 localPoint = (intersectionPoint - center) / radius; // Normalize to unit sphere

        // Calculate spherical coordinates
        float theta = std::atan2(localPoint.z, localPoint.x); // Longitude
        float phi = std::acos(localPoint.y);                  // Latitude

        // Map spherical coordinates to texture coordinates (u, v)
        float u = (theta + M_PI) / (2 * M_PI);
        float v = phi / M_PI;

        return Intersection(ray, std::move(intersectionPoint), std::move(intersectionNormal), material, Vec2(u, v));
    }

};

struct Cylinder {
    Vec3 center;
    float_t radius;
    float_t eachSideHeight;
    Vec3 axis;
    PhongMaterial material;

    Cylinder(Vec3 center, float_t radius, float_t height, Vec3 axis, std::optional<PhongMaterial> material)
        : center(center), radius(radius), eachSideHeight(height), axis(axis), material(givenMaterialOrDefault(std::move(material))){ }

    std::optional<Intersection> intersect(const Ray& ray) {
        // mostly generated by chatgpt

        Vec3 d = ray.direction - axis * ray.direction.dot(axis);  // Projected ray direction onto the cylinder's plane
        Vec3 oc = ray.origin - center;
        Vec3 oc_proj = oc - axis * oc.dot(axis);                  // Projected ray origin onto the cylinder's plane

        float a = d.dot(d);
        float b = 2.0f * d.dot(oc_proj);
        float c = oc_proj.dot(oc_proj) - radius * radius;
        std::optional<Intersection> closestIntersection = std::nullopt;

        // Quadratic discriminant for side wall intersection
        float discriminant = b * b - 4 * a * c;
        if (discriminant >= 0) {
            float sqrtDiscriminant = std::sqrt(discriminant);
            for (float t : { (-b - sqrtDiscriminant) / (2.0 * a), (-b + sqrtDiscriminant) / (2.0 * a) }) {
                if (t < 0) continue;

                Vec3 point = ray.origin + ray.direction * t;
                Vec3 localPoint = point - center;
                float projectionOnAxis = localPoint.dot(axis);

                // Check if intersection point is within height limits of the cylinder
                if (projectionOnAxis >= -eachSideHeight && projectionOnAxis <= eachSideHeight) {
                    Vec3 normal = (localPoint - axis * projectionOnAxis).normalized();
                    Intersection intersection(ray, point, normal, material, textCoordsOfSideIntersection(point));


                    // Update closest intersection
                    if (!closestIntersection || t < (closestIntersection->point - ray.origin).length()) {
                        closestIntersection = intersection;
                    }
                }
            }
        }

        // Lambda to handle cap intersection
        auto checkCapIntersection = [&](const Vec3& capCenter, const Vec3& capNormal) -> std::optional<Intersection> {
            float denom = ray.direction.dot(capNormal);
            if (std::abs(denom) < 1e-6) return std::nullopt;

            float tCap = (capCenter - ray.origin).dot(capNormal) / denom;
            if (tCap < 0) return std::nullopt;

            Vec3 point = ray.origin + ray.direction * tCap;
            if ((point - capCenter).length() <= radius) {  // Check if within radius of cap
                Intersection intersection(ray, point, capNormal, material, textCoordsOfCapIntersection(point));
                return intersection;
            }
            return std::nullopt;
        };

        // Check intersections with the base and top caps
        for (auto& cap : { std::make_pair(center - axis * eachSideHeight, -axis), 
                std::make_pair(center + axis * eachSideHeight, axis) }) {
            if (auto capIntersection = checkCapIntersection(cap.first, cap.second); capIntersection) {
                float capDistance = (capIntersection->point - ray.origin).length();
                if (!closestIntersection || capDistance < (closestIntersection->point - ray.origin).length()) {
                    closestIntersection = capIntersection;
                }
            }
        }

        return closestIntersection;
    }

private:
    Vec2 textCoordsOfSideIntersection(const Vec3& intersectionPoint) const {
        Vec3 baseToIntersection = intersectionPoint - (center - axis * eachSideHeight);
        float vPosAlongAxis = baseToIntersection.dot(axis);        
        float v = vPosAlongAxis / (2 * eachSideHeight);  // Map height position to v in [0, 1]

        Vec3 circumferentialDir = (baseToIntersection - axis * vPosAlongAxis).normalized();
        float theta = std::atan2(circumferentialDir.z, circumferentialDir.x);        
        float u = (theta + M_PI) / (2 * M_PI);  // Map angle to u in [0, 1]

        return Vec2(u,v);
    }

    Vec2 textCoordsOfCapIntersection(const Vec3& intersectionPoint) const {
        // Determine which cap (top or bottom) we're on based on axis direction and height
        Vec3 capCenter = intersectionPoint.dot(axis) > 0 ? (center + axis * eachSideHeight) : (center - axis * eachSideHeight);
        Vec3 localCapPoint = intersectionPoint - capCenter;

        // Map `localCapPoint` to polar coordinates within the cap radius
        float r = localCapPoint.length() / radius;  // Distance from center mapped to [0, 1]
        float capTheta = std::atan2(localCapPoint.z, localCapPoint.x);

        float u = 0.5f + r * std::cos(capTheta) / 2;  // Map radial distance and angle to texture u
        float v = 0.5f + r * std::sin(capTheta) / 2;  // Map radial distance and angle to texture v

        return Vec2(u,v);
    }
};

struct Triangle {
    Vec3 v0,v1,v2;
    // TODO could try deduplicating materials for triangle objects later on, for big meshes that all have the same material
    PhongMaterial material;
    // these are only valid if the material has a texture
    Vec2 texCoordv0, texCoordv1, texCoordv2;

    Triangle(Vec3 v0, Vec3 v1, Vec3 v2, std::optional<PhongMaterial> material, Vec2 texCoordv0, Vec2 texCoordv1, Vec2 texCoordv2)
        : v0(v0), v1(v1), v2(v2), material(givenMaterialOrDefault(std::move(material))), texCoordv0(texCoordv0), texCoordv1(texCoordv1), texCoordv2(texCoordv2){ }

    Vec3 faceNormal() const{
        // TODO in the future, could interpolate this with the rest of the mesh
        return (v1-v0).cross(v2-v0);
    }

    // TODO normal vector interpolation at any point for smooth shading (requires knowledge of the rest of the mesh)

    std::optional<Intersection> intersect(const Ray& ray) {
        // Möller–Trumbore intersection
        // mostly generated by chatgpt
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        Vec3 h = ray.direction.cross(edge2);
        float a = edge1.dot(h);

        // If a is near zero, the ray is parallel to the triangle
        if (std::abs(a) < epsilon) return std::nullopt;

        float f = 1.0 / a;
        Vec3 s = ray.origin - v0;
        float u = f * s.dot(h);

        // Check if the intersection is outside the triangle
        if (u < 0.0 || u > 1.0) return std::nullopt;

        Vec3 q = s.cross(edge1);
        float v = f * ray.direction.dot(q);

        // Check if the intersection is outside the triangle
        if (v < 0.0 || u + v > 1.0) return std::nullopt;

        // Calculate the distance along the ray to the intersection point
        float t = f * edge2.dot(q);

        // Only accept intersections that are in front of the ray origin
        if (t > epsilon) {
            Vec3 intersectionPoint = ray.origin + ray.direction * t;
            Vec3 normal = faceNormal().normalized();  // Use the constant normal for the triangle

            // interpolate texture coordinates
            // calculate barycentric coordinate `w`
            float w = 1.0f - u - v;

            // interpolate the texture coordinates using barycentric weights
            Vec2 interpolatedTexCoord = texCoordv0 * w + texCoordv1 * u + texCoordv2 * v;

            return Intersection(ray, intersectionPoint, normal, material, interpolatedTexCoord);
        }

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
    Scene(uint32_t nBounces, RenderMode renderMode, std::unique_ptr<Camera> camera, Vec3 backgroundColor, std::vector<PointLight> lights, std::vector<std::variant<Triangle, Sphere, Cylinder>> objects)
        : nBounces(nBounces), renderMode(renderMode), camera(std::move(camera)), backgroundColor(backgroundColor), lights(lights), objects(objects){ }
};

struct Renderer{

    Scene scene;
    PPMWriter writer;
    // ordered the same way a PPM file is, row by row
    // use a buffer instead of writing to the file immediately to be able to do it in parallel
    std::vector<Vec3> pixelBuffer;

    Renderer(Scene&& scene, std::string_view outputFilePath)
        // TODO adjust writer
        : scene(std::move(scene)), writer(outputFilePath, this->scene.camera->width, this->scene.camera->height), pixelBuffer(this->scene.camera->widthPixels*this->scene.camera->heightPixels, Vec3(0.)){}

    void bufferSpecificPixel(Vec2 pixel, Vec3 color){
        assert(pixel.x >= 0 && pixel.x < scene.camera->width && pixel.y >= 0 && pixel.y < scene.camera->height && "Pixel out of range");
        assert(color == color.clamp(0.0f, 1.0f) && "Color must be clamped to [0,1]");
        assert(scene.camera->width * pixel.y + pixel.x < pixelBuffer.size() && "Pixel out of range");
        pixelBuffer[pixel.y * scene.camera->width + pixel.x] = color;
    }

    /// shades a single intersection point
    /// outputs an un-tonemapped color, not for immediate display
    Vec3 blinnPhongShading(const Intersection& intersectionToShade, uint32_t bounces = 1, float currentIOR = 1.0f) {
        if (bounces > scene.nBounces)
            return Vec3(0.0f);

        // material properties
        Vec3 diffuse = intersectionToShade.material.diffuseColorAtTextureCoords(intersectionToShade.textureCoords);
        float_t ambientIntensity = 0.25f;
        Vec3 ambient = diffuse * ambientIntensity;
        Vec3 specular = intersectionToShade.material.specularColor;
        float_t ks = intersectionToShade.material.ks;
        float specularExponentShinyness = intersectionToShade.material.specularExponent;

        auto isInShadow = [&](const Vec3& L) -> bool {
            Vec3 shadowRayOrigin = intersectionToShade.point + L * (100 * epsilon);
            Ray shadowRay(shadowRayOrigin, L);

            if (auto shadowIntersection = traceRayToClosestSceneIntersection(shadowRay)) {
                return (shadowIntersection->point - intersectionToShade.point).length() < (scene.lights[0].position - intersectionToShade.point).length();
            }
            return false;
        };

        // Helper function to calculate specular highlights
        auto calculateSpecularHighlights = [&]() -> Vec3 {
            Vec3 specularSum(0.0f);

            for(auto& light: scene.lights) {
                Vec3 L = (light.position - intersectionToShade.point).normalized();
                if(isInShadow(L))
                    continue;

                Vec3 V = -intersectionToShade.incomingRay.direction.normalized();
                Vec3 H = (L + V).normalized();
                float spec = std::pow(std::max(intersectionToShade.surfaceNormal.dot(H), 0.0f), 
                        specularExponentShinyness);
                specularSum += specular * spec * light.intensityPerColor * ks;
            }
            return specularSum;
        };

        // Handle transparent (refractive) materials differently
        if (intersectionToShade.material.refractiveIndex.has_value()) {
            // make sure to still have specular highlights on transparent objects
            // these would be weighted by the objects transmissiveness, but as that doesnt exist, just add them for now
            Vec3 finalColor = calculateSpecularHighlights();
            float materialIOR = *intersectionToShade.material.refractiveIndex;

            bool entering = intersectionToShade.incomingRay.direction.dot(intersectionToShade.surfaceNormal) < 0;
            Vec3 normal = entering ? intersectionToShade.surfaceNormal : -intersectionToShade.surfaceNormal;
            float etaRatio = entering ? currentIOR / materialIOR : materialIOR / currentIOR;

            float cosTheta_i = -normal.dot(intersectionToShade.incomingRay.direction);
            float sinTheta_t_squared = etaRatio * etaRatio * (1.0f - cosTheta_i * cosTheta_i);

            if (sinTheta_t_squared <= 1.0f) {
                float cosTheta_t = std::sqrt(1.0f - sinTheta_t_squared);
                Vec3 refractedDir = etaRatio * intersectionToShade.incomingRay.direction + 
                    (etaRatio * cosTheta_i - cosTheta_t) * normal;

                Ray refractedRay(intersectionToShade.point + refractedDir * (10 * epsilon), refractedDir);
                float nextIOR = entering ? materialIOR : 1.0f;

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
        Vec3 color(0.0f);

        // Ambient and diffuse
        color += ambient;  // ambient

        float_t kd = intersectionToShade.material.kd;
        for(auto& light: scene.lights) {
            Vec3 L = (light.position - intersectionToShade.point).normalized();
            if(isInShadow(L))
                continue;

            Vec3 N = intersectionToShade.surfaceNormal;
            float diff = std::max(N.dot(L), 0.0f);
            color += diffuse * diff * light.intensityPerColor * kd;  // diffuse
        }

        // Add specular for all materials
        color += calculateSpecularHighlights();

        // Handle reflection if material is reflective
        // TODO could do fresnel here and for refractivity
        if(intersectionToShade.material.reflectivity.has_value()) {
            Vec3 reflectedColor = scene.backgroundColor;
            Vec3 reflectionDir = intersectionToShade.incomingRay.direction - 
                intersectionToShade.surfaceNormal * 2 * 
                (intersectionToShade.incomingRay.direction.dot(intersectionToShade.surfaceNormal));
            Ray reflectionRay(intersectionToShade.point + reflectionDir * (10 * epsilon), reflectionDir);

            if (auto reflectionIntersection = traceRayToClosestSceneIntersection(reflectionRay)) {
                reflectedColor = blinnPhongShading(*reflectionIntersection, bounces + 1, currentIOR);
            }

            color = color.lerp(reflectedColor, *intersectionToShade.material.reflectivity);
        }

        return color;
    }

    Vec3 linearToneMapping(Vec3 color){
        return (color*scene.camera->exposure * 15.).clamp(0.0f, 1.0f);
    }

    Vec3 shadeIntersection(const Intersection& intersectionToShade){
        // "0th" bounce: decide on the shading model, and do tone mapping
        auto shadedHDRColor = blinnPhongShading(intersectionToShade);
        auto toneMapped = linearToneMapping(shadedHDRColor);
        return toneMapped;
    }


    std::optional<Intersection> traceRayToClosestSceneIntersection(const Ray& ray){
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
// cant try gpu openacc because nvcc doesnt support c++23 :(
// openacc might not work at all with gcc here, is basically the same time as serial
//#pragma acc parallel loop
#pragma omp parallel for
        for(uint32_t y = 0; y < scene.camera->heightPixels; y++){
            for(uint32_t x = 0; x < scene.camera->widthPixels; x++){
                Ray cameraRay = scene.camera->generateRay(Vec2(x, y));
                //std::println("Camera ray direction: ({},{},{}) from ({},{},{})", cameraRay.direction.x, cameraRay.direction.y, cameraRay.direction.z, cameraRay.origin.x, cameraRay.origin.y, cameraRay.origin.z);
                // TODO other objects etc
                // TODO get closest intersection, not just test one
                auto closestIntersection = traceRayToClosestSceneIntersection(cameraRay);

                Vec3 pixelColor = linearToneMapping(scene.backgroundColor);
                if(closestIntersection.has_value()){
                    if constexpr (mode == RenderMode::BINARY){
                        pixelColor = Vec3(1.0);
                    }else if constexpr (mode == RenderMode::PHONG){
                        pixelColor = shadeIntersection(*closestIntersection);
                    }else{
                        static_assert(false, "Invalid render mode");
                    }
                }

                bufferSpecificPixel(Vec2(x, y), pixelColor);
            }
        }

        // write buffer to file
        assert(pixelBuffer.size() == scene.camera->widthPixels * scene.camera->heightPixels && "Pixel buffer size mismatch");
        for(auto& pixel: pixelBuffer)
            writer.writePixel(pixel);
    }



};
