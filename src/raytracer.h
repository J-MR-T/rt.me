#pragma once

#include <atomic>
#include <ranges>
#include <thread>

#include "io.h"
#include "util.h"
#include "geometry.h"
#include "cameras.h"

struct PointLight{
    Vec3 position;
    // the json files seem to integrate intensity and color into one vector
    Vec3 intensityPerColor;
    /// 0 shadow softness means copletely hard shadows, higher means softer
    /// ONLY AFFECTS PATHTRACED SHADOWS
    float_T shadowSoftness;

    /// if falloff is 0, light will not fall off with distance and always be = intensityPerColor
    float_T falloff;

    Vec3 intensityAtDistance(float_T distance) const{
        Vec3 fallenOffInstensity = intensityPerColor / (distance * distance + 1);
        return intensityPerColor.lerp(fallenOffInstensity, falloff);
    }
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

enum class ToneMapMode{
    // local linear is the default, see `localLinearToneMapping` for why
    LOCAL_LINEAR,  // see `localLinearToneMapping`
    GLOBAL_LINEAR, // see `globalLinearToneMapping`
};

struct Scene{
    uint32_t nBounces;
    RenderMode renderMode;
    std::unique_ptr<Camera> camera;
    Vec3 backgroundColor;

    ToneMapMode toneMapMode;

    bool phongFresnel;

    std::vector<PointLight> pointLights;

    std::vector<SceneObject> objects;

    struct {
        uint32_t samplesPerPixel;
        uint32_t apertureSamplesPerPixelSample;
        uint32_t pointLightSamplesPerBounce;
        /// be aware, that these are the samples that will result in exponentially more rays
        /// there is almost no reason to use this higher than 1, more samples per pixel is the better monte carlo answer to this
        uint32_t hemisphereSamplesPerBounce;
    } pathtracingSamples;

    Scene(uint32_t nBounces,
            RenderMode renderMode,
            std::unique_ptr<Camera> camera,
            Vec3 backgroundColor,
            ToneMapMode toneMapMode,
            bool phongFresnel,
            std::vector<PointLight> lights,
            std::vector<SceneObject> objects,
            uint32_t pathtracingSamplesPerPixel,
            uint32_t pathtracingApertureSamplesPerPixelSample,
            uint32_t pathtracingPointLightSamplesPerBounce,
            uint32_t pathtracingHemisphereSamplesPerBounce)
        : nBounces(nBounces),
        renderMode(renderMode),
        camera(std::move(camera)),
        backgroundColor(backgroundColor),
        toneMapMode(toneMapMode),
        phongFresnel(phongFresnel),
        pointLights(std::move(lights)),
        objects(std::move(objects)),
        pathtracingSamples(
            pathtracingSamplesPerPixel,
            pathtracingApertureSamplesPerPixelSample,
            pathtracingPointLightSamplesPerBounce,
            pathtracingHemisphereSamplesPerBounce
        ){ }

    /// assumes that there is at least one point light in the scene
    const PointLight& randomPointLight() const {
        assert(!pointLights.empty() && "No point lights in scene");
        // std::rand() % pointLights.size() literally triples render time, so use this instead
        return pointLights[randomFloat() * pointLights.size()];
    }
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
          bvh(BVHNode(BVHNode::ObjectRange(0, this->scene.objects.size()), this->scene.objects))
    {}

    // all default values as an overview
    struct Defaults {
        static constexpr std::string outputFilePath       = "out.ppm";
        static constexpr uint32_t nBounces                = 4;
        static constexpr ToneMapMode toneMapMode          = ToneMapMode::LOCAL_LINEAR;
        static constexpr bool phongFresnel                = true;
        static constexpr float_T pointLightShadowSoftness = 0.;
        static constexpr float_T pointLightFalloff        = 0.; // dont let point lights fall off with distance by default
        static constexpr float_T emissiveness             = 0.;
        static constexpr PhongMaterial defaultPhongMaterial   = PhongMaterial(Vec3(1,1,1), Vec3(1,1,1), 0.5, 0.5, 32, 0.0, 0.0, std::nullopt);
        static constexpr PrincipledBRDFMaterial defaultPrincipledBRDFMaterial = PrincipledBRDFMaterial(Vec3(1.), std::nullopt, 0., 1., 0., 0., 0.5, 1., 0., 0., 0., 0., 0., 0.);

    };

    template<RenderMode mode>
    void render(){
        std::println(stderr, "BVH num nodes: {} (bvh memory usage: ~{:.4f} MB)", bvh.numNodes(), bvh.numNodes() * sizeof(BVHNode) * 1e-6);
        std::println(stderr, "num objects: {}", bvh.objectRange.size());

        if constexpr(mode == RenderMode::DEBUG_BVH)
            renderDebugBVHToBuffer();
        else if constexpr(mode == RenderMode::DEBUG_NORMALS)
            renderDebugNormalsToBuffer();
        else if constexpr(mode == RenderMode::PATHTRACE || mode == RenderMode::PATHTRACE_INCREMENTAL){
            // until user closes stdin (Ctrl+D)
            if constexpr(mode == RenderMode::PATHTRACE_INCREMENTAL)
                std::println(stderr, "Rendering incrementally. Press Ctrl+D to stop after next frame");

            // === stop when the user closes stdin or presses Ctrl+C, skip to next '===' comment ===

            volatile std::atomic<bool> stopRendering = false;
            static_assert(std::atomic<bool>::is_always_lock_free);
            if(mode == RenderMode::PATHTRACE)
                stopRendering = true;

            // start a thread that stops the program when the user closes stdin
            std::jthread stopThread([&stopRendering]{
                char _[2];
                while(!stopRendering && std::fgets(_, sizeof(_), stdin) != nullptr);

                stopRendering = true;
                std::println(stderr, "Will stop rendering after this frame");
            });

            // if the user presses Ctrl+C, abort rendering
            auto signalHandler = [](int){
                // c functions due to signal-safety (7)
                char buf[] = "Received SIGINT, aborting rendering (might result in broken image)\n";
                write(STDERR_FILENO, buf, sizeof(buf));
                _exit(1);
            };
            std::signal(SIGINT, signalHandler);

            // === actual rendering ===

            // always render once
            renderPathtraceToBuffer();
            writeBufferToFile();

            // keep track of rendered state to be able to average the results (no fancy sampling for incremental yet)
            auto samplesPerPixel = scene.pathtracingSamples.samplesPerPixel;
            auto samplesPerPixelSoFar = samplesPerPixel;
            std::vector<Vec3> previousBuffer = hdrPixelBuffer;

            while (!stopRendering) {
                std::println(stderr, "Frame rendered, {} samples per pixel so far", samplesPerPixelSoFar);

                renderPathtraceToBuffer();
                const auto samplePixelsNow = samplesPerPixel + samplesPerPixelSoFar;
                // average the results
                for(auto [pixel, previousPixel]: std::ranges::views::zip(hdrPixelBuffer, previousBuffer)){
                    pixel = samplesPerPixel*pixel/samplePixelsNow + samplesPerPixelSoFar*previousPixel/samplePixelsNow;
                }
                writeBufferToFile();
                previousBuffer = hdrPixelBuffer;

                samplesPerPixelSoFar = samplePixelsNow;
            } 

            std::println(stderr, "Rendering stopped: Final image has {} samples per pixel", samplesPerPixelSoFar);
        }else{
#pragma omp parallel for
            for(uint32_t y = 0; y < scene.camera->heightPixels; y++){
                for(uint32_t x = 0; x < scene.camera->widthPixels; x++){
                    Ray cameraRay = scene.camera->generateRay(Vec2(x, y));

                    Vec3 pixelColor = scene.backgroundColor;
                    if(auto closestIntersection = traceRayToClosestSceneIntersection(cameraRay)){
                        if constexpr (mode == RenderMode::BINARY){
                            bufferSpecificPixel(Vec2(x, y), Vec3(1.));
                            continue;
                        }else if constexpr (mode == RenderMode::PHONG){
                            pixelColor = shadeBlinnPhong(*closestIntersection);
                        }else{
                            static_assert(false, "Invalid render mode");
                        }
                    }

                    if(scene.toneMapMode == ToneMapMode::LOCAL_LINEAR)
                        bufferSpecificPixel(Vec2(x, y), localLinearToneMapping(pixelColor));
                    else if(scene.toneMapMode == ToneMapMode::GLOBAL_LINEAR)
                        // buffer raw color, tone map afterwards
                        bufferSpecificPixel(Vec2(x, y), pixelColor);
                    else
                        assert(false && "Invalid tone map mode");
                }
            }

            if(mode == RenderMode::PHONG && scene.toneMapMode == ToneMapMode::GLOBAL_LINEAR){
                // compute max brightness sequentially afterwards
                // if we comptued this while the initial image was rendered, the synchronization overhead would be tremendous
                float_T maxIntensity = 0.;
                for(auto& pixel: hdrPixelBuffer)
                    if(pixel.length() > maxIntensity)
                        maxIntensity = pixel.length();

                // apply global tone mapping
                for(auto& pixel: hdrPixelBuffer)
                    pixel = globalLinearToneMapping(pixel, maxIntensity);
            }
        }

        writeBufferToFile();
    }

private:

    void bufferSpecificPixel(Vec2 pixel, Vec3 color){
        assert(pixel.x >= 0 && pixel.x < scene.camera->width && pixel.y >= 0 && pixel.y < scene.camera->height && "Pixel out of range");
        assert(scene.camera->width * pixel.y + pixel.x < hdrPixelBuffer.size() && "Pixel out of range");
        hdrPixelBuffer[pixel.y * scene.camera->width + pixel.x] = color;
    }

    void writeBufferToFile(){
        // start where the pixel data starts
        writer.rewind();

        assert(hdrPixelBuffer.size() == scene.camera->widthPixels * scene.camera->heightPixels && "Pixel buffer size mismatch");
        for(auto& hdrPixel: hdrPixelBuffer){
            auto ldrPixel = hdrPixel.clamp(0., 1.);
            writer.writePixel(ldrPixel);
        }
        writer.flush();
    }


    bool isInShadow(const Intersection& intersection, const PointLight& light, const Vec3& L) {
        auto shadowRay = Ray::createWithOffset(intersection.point, L, 100.);
        return isInShadow(intersection, light, shadowRay);
    };

    bool isInShadow(const Intersection& intersection, const PointLight& light, const Ray& shadowRay) {
        if (auto shadowIntersection = traceRayToClosestSceneIntersection(shadowRay)) {
            // if the shadow intersection is closer to the original intersection than the light, it's in shadow, otherwise the shadow intersection is behind the light, so it does not obscure it
            return (shadowIntersection->point - intersection.point).length() < (light.position - intersection.point).length() - 100 * epsilon;
        }
        return false;
    }

    /// shades a single intersection point
    /// outputs an un-tonemapped color, not for immediate display
    Vec3 shadeBlinnPhong(const Intersection& intersectionToShade, uint32_t bounces = 1, float_T currentIOR = 1.) {
        if (bounces > scene.nBounces)
            // could also return the background color, but black makes it easier to see when a render has too few bounces
            return Vec3(0.);

        const PhongMaterial& material = intersectionToShade.material->assumePhongMaterial();

        // material properties
        Vec3 diffuseColor = material.diffuseColorAtTextureCoords(intersectionToShade.textureCoords);
        // ambient intensity is global constant for all materials
        float_T ambientIntensity = PhongMaterial::ambientIntensity;
        // to match the provided images, use the diffuse color for the ambient color
        Vec3 ambient = diffuseColor * ambientIntensity;
        Vec3 specularColor = material.specularColor;
        float_T ks = material.ks;
        float_T specularExponentShinyness = material.specularExponent;

        auto calculateSpecularHighlights = [&] -> Vec3 {
            Vec3 specularSum(0.);

            for(const auto& light: scene.pointLights) {
                // L: light vector
                Vec3 L = (light.position - intersectionToShade.point).normalized();
                if(isInShadow(intersectionToShade, light, L))
                    continue;

                // V: view vector (invert incoming ray so that it points outward)
                Vec3 V = -intersectionToShade.incomingRay->direction;
                // H: half vector half between L and V
                Vec3 H = (L + V).normalized();
                float_T specularWeight = std::pow(std::max(intersectionToShade.surfaceNormal.dot(H), (float_T) 0.), 
                        specularExponentShinyness);

                float_T lightDistance = (light.position - intersectionToShade.point).length();
                specularSum += specularColor * specularWeight * light.intensityAtDistance(lightDistance) * ks;
            }
            return specularSum;
        };

        auto calculateReflectedColor = [&] -> Vec3 {
            Vec3 reflectionDir = intersectionToShade.incomingRay->direction.reflect(intersectionToShade.surfaceNormal);
            auto reflectionRay = Ray::createWithOffset(intersectionToShade.point, reflectionDir);

            if (auto reflectionIntersection = traceRayToClosestSceneIntersection(reflectionRay)) {
                return shadeBlinnPhong(*reflectionIntersection, bounces + 1, currentIOR);
            }else{
                return scene.backgroundColor;
            }
        };

        // Handle transparent (refractive) materials differently
        if (material.refractiveIndex.has_value()) {
            float_T materialIOR = *material.refractiveIndex;

            bool entering = intersectionToShade.incomingRay->direction.dot(intersectionToShade.surfaceNormal) < 0;
            Vec3 normal = entering ? intersectionToShade.surfaceNormal : -intersectionToShade.surfaceNormal;
            float_T etaRatio = entering ? currentIOR / materialIOR : materialIOR / currentIOR;

            float_T cosTheta_i = -normal.dot(intersectionToShade.incomingRay->direction);
            float_T sinTheta_t_squared = etaRatio * etaRatio * (1. - cosTheta_i * cosTheta_i);

            if (sinTheta_t_squared <= 1.) {
                float_T cosTheta_t = std::sqrt(1. - sinTheta_t_squared);
                Vec3 refractedDir = etaRatio * intersectionToShade.incomingRay->direction + 
                    (etaRatio * cosTheta_i - cosTheta_t) * normal;

                auto refractedRay = Ray::createWithOffset(intersectionToShade.point, refractedDir);
                // assume that the next material is air when we're leaving the object.
                // this means that all objects that are placed in side one another are actually hollow.
                // this simplifies things and works reasonably well for most cases
                float_T nextIOR = entering ? materialIOR : 1.;

                Vec3 refractedColor = scene.backgroundColor;
                if (auto refractedIntersection = traceRayToClosestSceneIntersection(refractedRay)) {
                    refractedColor = shadeBlinnPhong(*refractedIntersection, bounces + 1, nextIOR);
                }

                // tint the refraction by the diffuse color, to be able to make e.g. red glass
                // NOTE: could add something like a density parameter to the material, to decide how much to tint (via lerp), but for now, thats just encoded in the diffuse color
                refractedColor = refractedColor * diffuseColor;

                // make sure to still have specular highlights on transparent objects, basically like a coat on top of the refraction
                refractedColor += calculateSpecularHighlights();

                // if fresnel is enabled, calculate the fresnel term
                if(scene.phongFresnel){
                    // this reflectivity is for looking straight at the surface, i.e. along the normal vector
                    float_T baseReflectivity = std::pow((etaRatio - 1) / (etaRatio + 1), 2);

                    float_T fresnelBlendAmount = addWeightedSchlickFresnel(baseReflectivity, cosTheta_i);

                    // blend between the reflected and refracted color based on fresnel amount
                    // but also give control to the user by utilizing the reflectivity: if reflectivity is zero, only refract
                    auto reflectedColor = refractedColor.lerp(calculateReflectedColor(), material.reflectivity.value_or(0.));
                    return refractedColor.lerp(reflectedColor, fresnelBlendAmount);
                }else{
                    return refractedColor;
                }
            }else{
                // in this case we have *total internal reflection*
                // handle by simply going on with the normal lighting for now
                // This looks realistic if the material is reflective
                // (basically, the designer is responible for making the material look good, just like with the rest of phong shading)
            }

        }

        // Regular materials: ambient + diffuse + specular highlights + reflection
        Vec3 color(0.);

        // Ambient and diffuse
        color += ambient;  // ambient

        // diffuse
        float_T kd = material.kd;
        for(const auto& light: scene.pointLights) {
            Vec3 L = (light.position - intersectionToShade.point).normalized();
            if(isInShadow(intersectionToShade, light, L))
                continue;

            Vec3 N = intersectionToShade.surfaceNormal;
            float_T diffuseWeight = std::max(N.dot(L), (float_T) 0.);

            float_T lightDistance = (light.position - intersectionToShade.point).length();

            color += diffuseColor * diffuseWeight * light.intensityAtDistance(lightDistance) * kd;
        }

        // Add specular for all materials
        color += calculateSpecularHighlights();

        // Handle reflection if material is reflective
        if(material.reflectivity.has_value()) {
            Vec3 reflectedColor = calculateReflectedColor();

            color = color.lerp(reflectedColor, *material.reflectivity);
        }

        return color;
    }

    /// performs local tone mapping, i.e. independent of the rest of the image
    /// this works nicely for the pathtracer, as it does not darken the entire image if a light is visible (like a simple global method would)
    /// it also looks good for 
    /// does *not* clamp the color yet, this is done in writing the buffer to the file
    Vec3 localLinearToneMapping(Vec3 color){
        // we cannot clamp the color here, because for incremental renders, we need the buffer to be in high-dynamic range,
        // i.e. if we clamped here, and then averaged with later samples, the image wouldn't be brightened up
        // (see PBR 3rd edition Figure 14.7 for more)
        return color * scene.camera->exposure * Camera::exposureCorrectionFactor;
    }

    /// UNUSED IN THE DEFAULT OPTIONS (enable via "tonemap: globalLinear")
    /// this can be used for phong shading, but it looks worse than the local tone mapping + clamping in my opinion.
    /// does *not* clamp the color yet, this is done in writing the buffer to the file
    Vec3 globalLinearToneMapping(Vec3 color, float_T maximumIntensity){
        return scene.camera->exposure * Camera::exposureCorrectionFactor * color / maximumIntensity;
    }

    /// only used for pathtraced shading, as the phong references don't seem to do gamma correction
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
                pixel = Vec3(intensity, 0.0, 0.0);
            else
                pixel = pixel / maxIntensity;
        }
    }

    void renderDebugNormalsToBuffer() {
        for(uint32_t y = 0; y < scene.camera->heightPixels; y++){
            for(uint32_t x = 0; x < scene.camera->widthPixels; x++){
                Ray cameraRay = scene.camera->generateRay(Vec2(x, y));

                if(auto closestIntersection = traceRayToClosestSceneIntersection(cameraRay)){
                    auto normalBetweenZeroOne = closestIntersection->surfaceNormal * 0.5 + Vec3(0.5);
                    bufferSpecificPixel(Vec2(x, y), normalBetweenZeroOne);
                }
            }
        }
    }

    Vec3 shadePathtraced(const Intersection& intersection, uint32_t bounces = 1){
        if(bounces > scene.nBounces)
            return Vec3(0.);

        const PrincipledBRDFMaterial& material = intersection.material->assumePrincipledBRDFMaterial();

        // start with emission
        Vec3 overallColor = material.emissionColor(intersection.textureCoords);

        Vec3 N = intersection.surfaceNormal.normalized();

        // view direction is the inverse of the incoming ray direction
        Vec3 V = (-intersection.incomingRay->direction).normalized();

        // create tangent space basis vectors (necessary for BRDF)
        auto [X, Y] = Vec3::createOrthonormalBasis(N);

        // TODO reintroduce comments from previous version

        unsigned actualSamplesTaken = 0;

        // Sample contribution for each hemisphere sample
        Vec3 accumulatedContributions(0.);
        for(unsigned hemisphereSampleNum = 0; 
                hemisphereSampleNum < scene.pathtracingSamples.hemisphereSamplesPerBounce; 
                hemisphereSampleNum++) {

            // Get multiple-importance-sample from BRDF
            auto sample = material.sampleBRDF(V, N, X, Y);

            // Skip invalid samples
            if (sample.pdf <= epsilon) {
                continue;
            }

            // in this case, we're going to actually use the sample
            actualSamplesTaken++;

            // create and trace ray in sampled direction
            auto incomingRay = Ray::createWithOffset(intersection.point, sample.direction);

            Vec3 incomingColor = scene.backgroundColor;
            if(auto incomingIntersection = traceRayToClosestSceneIntersection(incomingRay)) {
                incomingColor = shadePathtraced(*incomingIntersection, bounces + 1);
            }

            // evaluate BRDF for this direction
            Vec3 brdfValue = material.evaluateBRDF(
                intersection.textureCoords,
                V,
                sample.direction,
                X, Y, N
            );

            float_T cosTheta = std::max((float_T)0., N.dot(sample.direction));

            // rendering equation, adjusting for the monte carlo integration (divide by pdf)
            accumulatedContributions += incomingColor * brdfValue * cosTheta/sample.pdf;
        }

        if(actualSamplesTaken > 0)
            // Average the samples
            overallColor += accumulatedContributions / actualSamplesTaken;

        if(!scene.pointLights.empty()){
            // sample point lights explicitly, because they are infinitessimally small, they can never be hit by a random ray (and are thus also not part of the BVH)
            // luckily, the pdf of the dirac delta distribution representing these cancels out with the light intensity of the point light itself, so we can simply add it, if the light is not in shadow

            // optionally sample each light source multiple times
            // not strictly necessary because of monte carlo - we're sampling each pixel multiple times anyway
            // but this gives greater control, although it should be 1 in most cases
            Vec3 accumulatedContributions(0.);

            for(unsigned lightSampleNum = 0; lightSampleNum < scene.pathtracingSamples.pointLightSamplesPerBounce; lightSampleNum++){
                // we could just sample all point lights for every bounce, but thats a bit wasteful again for the later bounces
                // -> randomly pick one (pointLightSamplesPerBounce defaults to 1), then compensate for that choice by multiplying with the number of point lights
                // can adjust the convergence rate for point lights by changing the number of samples ber bounce
                const auto& light = scene.randomPointLight();

                // permute the origin randomly if the light has some amount of softness
                Vec3 intersectionOriginPlusJitter = intersection.point + Vec3(randomFloat() - 0.5, randomFloat() - 0.5, randomFloat() - 0.5) * light.shadowSoftness;
                // the jitter here is equal in all directions which is not ideal, but good enough. Could improve this by jittering around the tangent (X) and bitangent (Y)

                Vec3 L = (light.position - intersectionOriginPlusJitter).normalized();

                if(isInShadow(intersection, light, L))
                    continue;

                Vec3 brdf = material.evaluateBRDF(intersection.textureCoords, V, L, X, Y, N);

                float_T cosTheta = std::max(intersection.surfaceNormal.dot(L), (float_T) 0.);

                float_T lightDistance = (light.position - intersection.point).length();

                // rendering equation
                accumulatedContributions += brdf * cosTheta * light.intensityAtDistance(lightDistance);
            }

            // compensate for only sampling one light
            accumulatedContributions *= scene.pointLights.size();
            // compensate for the number of samples
            accumulatedContributions /= scene.pathtracingSamples.pointLightSamplesPerBounce;

            overallColor += accumulatedContributions;
        }

        return overallColor;
    }

    void renderPathtraceToBuffer(){
        // dynamic thread scheduling improves performance by ~10% on cornell box with 6 bounces (because some rays terminate early by bouncing into nothing)
#pragma omp parallel for collapse(2) schedule(dynamic, 64)
        for(uint32_t y = 0; y < scene.camera->heightPixels; y++){
            for(uint32_t x = 0; x < scene.camera->widthPixels; x++){
                const Vec2 pixelOrigin = Vec2(x, y);
                Vec3 colorSum = Vec3(0.);
                for(uint32_t sample = 0; sample < scene.pathtracingSamples.samplesPerPixel; sample++){
                    // permute ray randomly/evenly for jittered sampling
                    Vec2 permutedPixel = pixelOrigin + Vec2(randomFloat() - 0.5, randomFloat() - 0.5);

                    Ray cameraRay = scene.camera->generateRay(permutedPixel);

                    if(auto intersection = traceRayToClosestSceneIntersection(cameraRay)){
                        colorSum += shadePathtraced(*intersection);
                    }else{
                        // background color
                        colorSum += scene.backgroundColor;
                    }
                }

                bufferSpecificPixel(Vec2(x, y), localLinearToneMapping(gammaCorrect(colorSum / scene.pathtracingSamples.samplesPerPixel)));
            }
        }
    }

};
