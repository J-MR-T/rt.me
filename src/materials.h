#pragma once

#include <memory>

#include "util.h"

struct Color8Bit{
    uint8_t r,g,b;
};

struct Texture{
private:
    uint32_t width, height;
    // store textures in 8-bit color format, saves 4x memory compared to float
    std::vector<Color8Bit> pixels;

public:
    Texture(uint32_t width, uint32_t height, std::vector<Color8Bit> pixels)
        : width(width), height(height), pixels(pixels){
        assert(width * height == pixels.size() && "Texture dimensions don't match pixel count");
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

struct MaterialBase{
    Vec3 diffuseBaseColor;
    // share textures to reduce memory usage
    std::optional<std::shared_ptr<Texture>> texture;

    /// if the material has a texture, get the diffuse color at the given texture coordinates,
    /// otherwise just return the diffuse color
    /// The texture coordinates here need to be interpolated from the vertices of e.g. the triangle
    Vec3 diffuseColorAtTextureCoords(const Vec2& textureCoords) const {
        if(texture.has_value())
            return (*texture)->colorAt(textureCoords);
        else
            return diffuseBaseColor;
    }
};

struct PhongMaterial : MaterialBase {
    Vec3 specularColor;
    float_T ks,kd;
    uint64_t specularExponent;
    std::optional<float_T> reflectivity;
    std::optional<float_T> refractiveIndex;

    static constexpr float_T ambientIntensity = 0.25;

    constexpr PhongMaterial(
            Vec3 diffuseColor,
            Vec3 specularColor,
            float_T ks,
            float_T kd,
            uint64_t specularExponent,
            std::optional<float_T> reflectivity,
            std::optional<float_T> refractiveIndex,
            std::optional<std::shared_ptr<Texture>> texture)
        : MaterialBase(diffuseColor, texture), specularColor(specularColor), ks(ks), kd(kd), specularExponent(specularExponent), reflectivity(reflectivity), refractiveIndex(refractiveIndex) { }

    /// if the material has a texture, get the diffuse color at the given texture coordinates,
    /// otherwise just return the diffuse color
    /// The texture coordinates here need to be interpolated from the vertices of e.g. the triangle
    Vec3 diffuseColorAtTextureCoords(const Vec2& textureCoords) const {
        if(texture.has_value())
            return (*texture)->colorAt(textureCoords);
        else
            return diffuseBaseColor;
    }

};

/// Disney-like Principled BRDF
struct PrincipledBRDFMaterial : MaterialBase{
    float_T emissiveness;

    float_T baseColorDiffuseIntensity;
    float_T metallic;
    float_T subsurface;
    float_T specular;
    float_T roughness;
    float_T specularTint;
    float_T anisotropic;
    float_T sheen;
    float_T sheenTint;
    float_T clearcoat;
    float_T clearcoatGloss;

    // TODO explain V, L, H, N, X, Y terminology

    constexpr PrincipledBRDFMaterial(
            Vec3 diffuseColor,
            std::optional<std::shared_ptr<Texture>> texture,
            float_T emissiveness,
            float_T baseColorIntensity,
            float_T metallic,
            float_T subsurface,
            float_T specular,
            float_T roughness,
            float_T specularTint,
            float_T anisotropic,
            float_T sheen,
            float_T sheenTint,
            float_T clearcoat,
            float_T clearcoatGloss)
        : MaterialBase(diffuseColor, texture), emissiveness(emissiveness), baseColorDiffuseIntensity(baseColorIntensity), metallic(metallic), subsurface(subsurface), specular(specular), roughness(roughness), specularTint(specularTint), anisotropic(anisotropic), sheen(sheen), sheenTint(sheenTint), clearcoat(clearcoat), clearcoatGloss(clearcoatGloss) {

            // Diffuse weight only depends on:
            // - metallic (metals don't have diffuse)
            // - clearcoat (reduces underlying diffuse)
            // - and the diffuse intensity itself
            diffuseWeight = (1.0f - metallic) * baseColorDiffuseIntensity * (1.0f - 0.5f * clearcoat * clearcoatGloss);

            // Specular weight depends on:
            // - metallic (increases specular)
            // - roughness (decreases specular more aggressively)
            // - specular parameter
            specularWeight = (1.0f + metallic) * specular * (1.0f - roughness * roughness);

            clearcoatWeight = clearcoat;

            // normalize them
            float_T total = diffuseWeight + specularWeight + clearcoatWeight;
            diffuseWeight   /= total;
            specularWeight  /= total;
            clearcoatWeight /= total;
        }

    Vec3 emissionColor(const Vec2& textureCoords) const {
        return emissiveness * diffuseColorAtTextureCoords(textureCoords);
    }

    // NOTE: the implementation is based off of generative AI, heavily refined by hand

private:
    // sampling weights for BRDF sampling components
    // these weights always add up to 1

    float_T diffuseWeight;
    float_T specularWeight;
    float_T clearcoatWeight;

    // this seems to be the standard value from the literature
    static constexpr float_T fixedClearcoatRoughness = 0.25;

    // === Utility functions ===

    // maths
    static float_T square(float_T x) { return x * x; }
    static float_T safe_sqrt(float_T x) { return std::sqrt(std::max(epsilon, x)); }

    // optics
    // (also see schlickFresnel)
    static float_T luminance(const Vec3& color) {
        return std::max(epsilon, color.dot(Vec3(0.3f, 0.6f, 0.1f)));
    }

    // Distribution functions
    static float_T GTR1(float_T NdotH, float_T a) {
        if (a >= 1.0f) return (float_T)(1.0 / PI);
        float_T a2 = std::max(epsilon, a * a);
        float_T t = 1.0 + (a2 - 1.0) * NdotH * NdotH;
        return (a2 - 1.0f) / (PI * std::log(a2) * t);
    }

    // GGX (Trowbridge-Reitz) Distribution Function
    static float_T D_GGX_aniso(const Vec3& H, const Vec3& N,
                              const Vec3& X, const Vec3& Y,
                              float_T ax, float_T ay) {
        float_T NdotH = std::max(epsilon, N.dot(H));

        // Early exit if normal and half vector are perpendicular
        if (NdotH <= 0) return 0;

        // Project H onto the tangent plane
        float_T HdotX = H.dot(X);
        float_T HdotY = H.dot(Y);

        // Calculate the squared slopes
        float_T ax2 = square(ax);
        float_T ay2 = square(ay);

        // Calculate the normalization factor
        float_T denom = (square(HdotX) / ax2 + square(HdotY) / ay2 + square(NdotH));

        return 1.0f / (PI * ax * ay * square(denom));
    }

    /// Calculate anisotropic roughness parameters
    static std::pair<float_T, float_T> calculateAnisotropicParams(float_T roughness, float_T anisotropic) {
        // TODO maybe get rid of parameters and make non-static?
        roughness = std::max(0.001f, roughness);
        
        // Modify how anisotropic affects the aspect ratio
        // Map anisotropic [0,1] to a more useful range for aspect ratio
        float_T t = anisotropic * 0.9f;  // Keep maximum anisotropy slightly below 1
        float_T ax = std::max(0.001f, roughness * (1.0f + t));
        float_T ay = std::max(0.001f, roughness * (1.0f - t));
        return {ax, ay};
    }

    Vec3 sampleGGX(float_T roughness, const Vec3& X, const Vec3& Y, const Vec3& N) const {
        float_T u1 = randomFloat();
        float_T u2 = randomFloat();

        auto [ax, ay] = calculateAnisotropicParams(roughness, anisotropic);

        // Transform V to tangent space
        // TODO what was this for?
        //Vec3 Vt = Vec3(V.dot(X), V.dot(Y), V.dot(N));

        // Sample visible normal distribution
        float_T phi = 2.0f * PI * u1;
        float_T theta = std::atan(roughness * std::sqrt(u2 / (1.0f - u2)));

        float_T cos_theta = std::cos(theta);
        float_T sin_theta = std::sin(theta);
        float_T cos_phi = std::cos(phi);
        float_T sin_phi = std::sin(phi);

        // Compute half vector in tangent space
        Vec3 Hlocal = Vec3(
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            cos_theta
        ).normalized();

        // Unstretch
        Hlocal = Vec3(Hlocal.x * ax, Hlocal.y * ay, Hlocal.z).normalized();

        // Transform back to world space
        return (X * Hlocal.x + Y * Hlocal.y + N * Hlocal.z).normalized();
    }

    // Geometric shadowing functions
    static float_T smithG_GGX_aniso(float_T NdotV, float_T VdotX, float_T VdotY, float_T NdotL, float_T LdotX, float_T LdotY,
            float_T ax, float_T ay) {
        float_T lambda_V = NdotV + safe_sqrt(square(VdotX*ax) + square(VdotY*ay) + square(NdotV));
        float_T lambda_L = NdotL + safe_sqrt(square(LdotX*ax) + square(LdotY*ay) + square(NdotL));

        return 2. / (lambda_V * lambda_L);
    }

    static float_T smithG(float_T NdotV, float_T alphaG) {
        float_T a = alphaG * alphaG;
        float_T b = NdotV * NdotV;
        return 1. / (NdotV + safe_sqrt(a + b - a * b));
    }

    // TODO clean up all of the clamps and epsilons


public:
    struct BRDFSample {
        Vec3 direction;  // Sampled direction in world space
        float_T pdf;    // Probability density of the sample
    };

    // Main sampling function
    BRDFSample sampleBRDF(const Vec3& V, const Vec3& N, const Vec3& X, const Vec3& Y) const {
        BRDFSample sample;

        float_T rand = randomFloat();
        
        // Sample direction based on weights
        if (rand < diffuseWeight) {
            // Cosine-weighted hemisphere sampling
            float_T r1 = randomFloat();
            float_T r2 = randomFloat();
            float_T phi = 2.0f * PI * r1;
            float_T cosTheta = std::sqrt(r2);
            float_T sinTheta = std::sqrt(1.0f - r2);
            
            sample.direction = (
                X * (std::cos(phi) * sinTheta) +
                Y * (std::sin(phi) * sinTheta) +
                N * cosTheta
            ).normalized();
        } else if (rand < diffuseWeight + specularWeight) {
            Vec3 H = sampleGGX(roughness * roughness, X, Y, N);
            sample.direction = (-V).reflect(H);
        } else {
            Vec3 H = sampleGGX(fixedClearcoatRoughness, X, Y, N);
            sample.direction = (-V).reflect(H);
        }

        const Vec3& L = sample.direction;

        // half vector: half-between V and L
        const Vec3& H = (V + L).normalized();

        // use lambdas to make clear what information each calculation is using (via their captures) - useful for readability and debugging

        float_T diffusePdf = [&N, &L] {
            // For cosine-weighted hemisphere sampling, the PDF is cos(theta)/pi
            // where theta is the angle between the normal and sampled direction
            float_T cosTheta = std::max(epsilon, N.dot(L));
            return cosTheta / PI;
        }();

        // needed for both specular and clearcoat PDFs
        float_T NdotH = std::max(epsilon, N.dot(H));
        float_T VdotH = std::max(epsilon, V.dot(H));

        float_T specularPdf = [&H, &N, &X, &Y, &NdotH, &VdotH, this]{
            auto [ax, ay] = calculateAnisotropicParams(roughness, anisotropic);

            // Calculate the GGX distribution term
            float_T D = D_GGX_aniso(H, N, X, Y, ax, ay);

            // The PDF for GGX importance sampling is:
            // pdf = D * NdotH / (4 * VdotH)
            // This comes from the Jacobian of the half-vector transformation

            if (VdotH < epsilon || NdotH < epsilon) {
                return 0.0f;
            }

            float_T pdf = (D * NdotH) / (4.0f * VdotH);

            return std::clamp(pdf, epsilon, 1e8f);
        }();

        float_T clearcoatPdf = [&VdotH, &NdotH, this]{
            // Clearcoat uses GTR1 distribution with fixed roughness
            // interpolated based on clearcoatGloss
            float_T alpha = std::lerp(0.1f, 0.001f, clearcoatGloss);

            // D_GTR1 term
            float_T D = GTR1(NdotH, alpha);

            if (VdotH < epsilon || NdotH < epsilon) {
                return 0.0f;
            }

            // Same jacobian as specular
            float_T pdf = D * NdotH / (4.0f * VdotH);
            return std::clamp(pdf, epsilon, 1e8f);
        }();

        // Combine PDFs using balance heuristic
        sample.pdf = diffuseWeight * diffusePdf + 
                    specularWeight * specularPdf + 
                    clearcoatWeight * clearcoatPdf;

        return sample;
    }

    // TODO rename/document params (also for sample)

    // Combined BRDF evaluation
    Vec3 evaluateBRDF(const Vec2& textureCoords, const Vec3& V, const Vec3& L, const Vec3& X, const Vec3& Y, const Vec3& N) const {
        float_T NdotL = N.dot(L);
        float_T NdotV = N.dot(V);
        
        if (NdotL <= 0.0f || NdotV <= 0.0f)
            return Vec3(0.0f);

        // round anything below epsilon to epsilon
        NdotL = std::max(epsilon, NdotL);
        NdotV = std::max(epsilon, NdotV);

        Vec3 H = (L + V).normalized();
        float_T LdotH = std::max(epsilon, L.dot(H));
        
        Vec3 baseColor = diffuseColorAtTextureCoords(textureCoords);

        auto diffuseBRDFContribution = [&]{
            float_T FL = std::pow(1.0f - NdotL, 5.0f);
            float_T FV = std::pow(1.0f - NdotV, 5.0f);
            float_T Rr = 2.0f * roughness * square(LdotH);

            float_T Fd90 = 0.5f + 2.0f * roughness * square(LdotH);
            float_T Fd = std::lerp(1.0f, Fd90, FL) * std::lerp(1.0f, Fd90, FV);

            float_T Fss90 = Rr;
            float_T Fss = std::lerp(1.0f, Fss90, FL) * std::lerp(1.0f, Fss90, FV);
            float_T ss = 1.25f * (Fss * (1.0f / (NdotL + NdotV) - 0.5f) + 0.5f);

            return (1.0f / PI) * std::lerp(Fd, ss, subsurface) * baseColor * baseColorDiffuseIntensity * (1.0f - metallic);
        }();

        auto specularBRDFContribution = [&] {
            // Calculate anisotropic roughness parameters
            auto [ax, ay] = calculateAnisotropicParams(roughness, anisotropic);

            // Calculate tint color
            Vec3 tint = (luminance(baseColor) > epsilon) ? 
                baseColor / luminance(baseColor) : Vec3(1.0f);

            // Calculate specular color with proper metallic workflow
            Vec3 specularColor = (Vec3(0.08)  * specular)
                .lerp(tint, specularTint)
                .lerp(baseColor, metallic);

            // Also calculate sheen contribution (disney sheen)
            Vec3 sheenColor = Vec3(1.).lerp(tint, sheenTint);
            Vec3 sheenContribution = sheen * sheenColor * schlickFresnelFactor(LdotH);

            // Fundamentally microfacet based:

            // Fresnel term
            Vec3 F = addWeightedSchlickFresnel(specularColor, LdotH);

            // Distribution term (GGX/Trowbridge-Reitz)
            float_T D = D_GGX_aniso(H, N, X, Y, ax, ay);

            // Geometric term
            float_T G = smithG_GGX_aniso(NdotV, V.dot(X), V.dot(Y),
                    NdotL, L.dot(X), L.dot(Y),
                    ax, ay);

            Vec3 specularBRDF = F * D * G;

            return specularBRDF + sheenContribution;
        }();

        auto clearcoatBRDFContribution = [&] {
            float_T NdotH = std::max(epsilon, N.dot(H));

            // Fixed clearcoat IOR of 1.5 gives F0 of 0.04 (glass-air interfaec)
            const float_T F0 = 0.04f;
            // Use the same Fresnel equation as specular but with fixed F0
            float_T Fr = addWeightedSchlickFresnel(F0, LdotH);

            // TODO decide

            // Option 1: Keep GTR1 with better gloss mapping
            float_T alpha = std::lerp(0.1f, 0.001f, clearcoatGloss);
            float_T Dr = GTR1(NdotH, alpha);

            // Option 2: Use GGX (like specular) with fixed roughness
            //float_T roughness = std::lerp(fixedClearcoatRoughness, 0.1f, clearcoatGloss);
            //float_T Dr = D_GGX_aniso(H, N, X, Y, roughness, roughness);

            // Use the same geometry term as specular but with fixed roughness
            float_T Gr = smithG(NdotL, fixedClearcoatRoughness) * smithG(NdotV, fixedClearcoatRoughness);

            return Vec3(clearcoat * Gr * Fr * Dr);
        }();
        
        return diffuseBRDFContribution + specularBRDFContribution + clearcoatBRDFContribution;
    }
};

/// for phong rendering, the material is always a PhongMaterial, for pathtracing, it is always a BRDFMaterial
struct Material {
    // In principle, we could introduce a generic `BRDFMaterial` (with `sampleBRDF` and `evaluateBRDF` functions), which could be different kinds of BRDFs.
    // But as the Disney BRDF is so versatile, there is no real need for more BRDF materials right now, so this distinction would just complicate things.
    std::variant<PhongMaterial, PrincipledBRDFMaterial> variant;

    const PhongMaterial& assumePhongMaterial() const {
        assert(std::holds_alternative<PhongMaterial>(variant) && "Material is not a Phong material");
        return std::get<PhongMaterial>(variant);
    }

    const PrincipledBRDFMaterial& assumePrincipledBRDFMaterial() const {
        assert(std::holds_alternative<PrincipledBRDFMaterial>(variant) && "Material is not a BRDF material");
        return std::get<PrincipledBRDFMaterial>(variant);
    }

    bool hasTexture() const {
        return std::visit([&](auto&& object){
            return object.texture.has_value();
        }, variant);
    }
};
