#pragma once
#include <cassert>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <fcntl.h>
#include <random>
#include <utility>
#include <string>
#include <print>
#include <format>

// single precision for now, ~15% faster than double, but defining the type here allows for easy switching
using float_T = float;

constexpr float_T epsilon = 1e-6;
constexpr float_T PI = M_PI;

inline bool implies(bool a, bool b){
    return !a || b;
}

/// generate a uniformly distributed random float in [0, 1)
inline float_T randomFloat() {
    // initially generated by ai, thread safe by me
    // make thread safe by ensuring each thread has its own generator
    thread_local static std::random_device rd;
    thread_local static std::mt19937 gen(rd());
    thread_local static std::uniform_real_distribution<float_T> dis(0., 1.);
    return dis(gen);
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

    Vec2 operator*(const Vec2& other) const{
        return Vec2(x*other.x, y*other.y);
    }

    Vec2 operator/(const Vec2& other) const{
        return Vec2(x/other.x, y/other.y);
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

    constexpr Vec3(float_T x, float_T y, float_T z) : x(x), y(y), z(z){ }

    constexpr explicit Vec3(float_T uniform) : x(uniform), y(uniform), z(uniform){ }

    constexpr Vec3() : Vec3(0.) {}

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
    Vec3& operator*=(float_T scalar){
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }
    Vec3& operator/=(float_T scalar){
        x /= scalar;
        y /= scalar;
        z /= scalar;
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

    Vec3 reflect(const Vec3& normal) const{
        const Vec3 incomingDirection = *this;
        return incomingDirection - normal * 2 * (incomingDirection.dot(normal));
    }

    // === static helper ===

    static std::pair</* tangent */ Vec3, /* bitangent */ Vec3> createOrthonormalBasis(const Vec3& N) {
        assert(N == N.normalized() && "N must be a normal vector");

        // First, pick a helper vector that's not parallel to N
        Vec3 helper = std::abs(N.y) < 0.999f ? Vec3(0, 1, 0) : Vec3(1, 0, 0);

        // Construct X (tangent/T) to be perpendicular to N using the helper
        Vec3 X = N.cross(helper).normalized();

        // Construct Y (bitangent/B) to be perpendicular to both N and X
        Vec3 Y = N.cross(X);  // NOTE: no need to normalize since N and X are unit vectors
                                     // and perpendicular to each other

        assert(X == X.normalized() && "X must be normalized");
        assert(Y == Y.normalized() && "Y must be normalized");
        assert(std::abs(X.dot(Y)) < epsilon && "X and Y must be orthogonal");
        assert(std::abs(N.dot(Y)) < epsilon && "N and Y must be orthogonal");
        assert(std::abs(N.dot(X)) < epsilon && "N and X must be orthogonal");

        return {X, Y};
    }

};

// overload std::format/std::println for Vec3
template <>
struct std::formatter<Vec3> : std::formatter<std::string> {
    auto format(const Vec3& v, format_context& ctx) const {
        return formatter<string>::format(std::format("[{}, {}, {}]", v.x, v.y, v.z), ctx);
    }
};

/// A ray, represented by its origin and direction
struct Ray{
    Vec3 origin;
    Vec3 direction;
    Vec3 invDirection;

    // constructors are not allowed for readability, createXyz functions should be used instead
private:
    /// assumes that the direction is normalized!
    Ray(Vec3 origin, Vec3 direction, Vec3 invDirection)
        : origin(origin), direction(direction), invDirection(invDirection){
        assert(direction == direction.normalized() && "Ray direction must be normalized");
        assert(invDirection == 1./direction && "Ray inverse direction must be the reciprocal of the direction");
    }

    Ray(Vec3 origin, Vec3 direction) : Ray(origin, direction, 1./direction){}

public:

    /// assumes that the direction is normalized!
    /// creates a ray exactly as specified
    static Ray createExact(Vec3 origin, Vec3 direction){
        return Ray(origin, direction);
    }

    /// Slightly offsets the ray from its origin in the direction to avoid self intersections
    /// with the object it might have originated from
    /// assumes that the direction is normalized!
    /// - epsilonFactor: how much the ray is offset from the origin in terms of the epsilon - the default of 10 is ususally fine
    static Ray createWithOffset(Vec3 origin, Vec3 direction, float_T epsilonFactor = 10.){
        return Ray(origin + direction * epsilonFactor * epsilon, direction);
    }
};

/// fresnel effect with Schlick's approximation
/// takes either an angle or something "similar", e.g. the dot product of the normal with the half vector (see PrincipledBRDFMaterial)
inline float_T schlickFresnelFactor(float_T angleEquivalent){
    return std::pow(1.0f - angleEquivalent, 5);
}
inline float_T addWeightedSchlickFresnel(float_T intensity, float_T angleEquivalent){
    return intensity + (1. - intensity) * schlickFresnelFactor(angleEquivalent);
}
inline Vec3 addWeightedSchlickFresnel(Vec3 color, float_T angleEquivalent){
    return Vec3(addWeightedSchlickFresnel(color.x, angleEquivalent), addWeightedSchlickFresnel(color.y, angleEquivalent), addWeightedSchlickFresnel(color.z, angleEquivalent));
}

// forward declaration, see materials.h

struct Material;

struct Intersection{
    // ray that caused the intersection
    const Ray* incomingRay;
    // NOTE: instead of storing the intersection point, could store the distance along the ray
    Vec3 point;
    Vec3 surfaceNormal;
    const Material* material;
    Vec2 textureCoords;

    Intersection(const Ray* incomingray, Vec3 point, Vec3 surfaceNormal, const Material* material, Vec2 textureCoords)
        : incomingRay(incomingray), point(point), surfaceNormal(surfaceNormal), material(material), textureCoords(textureCoords){
        assert(surfaceNormal == surfaceNormal.normalized() && "Surface normal must be normalized");
        assert(incomingRay && "Incoming ray must not be null");
        assert(material && "Material must not be null");
    }

    float_T distance() const{
        return (point - incomingRay->origin).length();
    }

};

