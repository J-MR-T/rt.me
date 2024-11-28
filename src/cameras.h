#pragma once

#include "util.h"

struct Camera{
    Vec3 position;
    Vec3 direction;
    Vec3 down; // PPM has (0,0) in the top left, so we want to go down not up
    Vec3 right;
    float_T width;
    uint64_t widthPixels;
    float_T height;
    uint64_t heightPixels;
    float_T exposure;

    // to make the images at the provided exposure look more like the reference
    static constexpr float_T exposureCorrectionFactor = 15.;

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

        // scale to world space scale (no translation yet) by multiplying by the image plane dimensions
        Vec2 pixelScaledByWorldSpace = Vec2(pixelCenterInCameraSpace.x * imagePlaneDimensions.x, pixelCenterInCameraSpace.y * imagePlaneDimensions.y);

        // now after scaling to world space, translate to world space (i.e. the cameras position), and add the camera's right/down directions
        Vec3 pixelOrigin = position + right*pixelScaledByWorldSpace.x + down*pixelScaledByWorldSpace.y;
        return pixelOrigin;
    }

    void setImagePlaneDimensionsFromFOV(float_T fovDegrees){
        // Convert FOV to radians
        const float_T verticalFOVRad = fovDegrees * (PI / 180.0);
        imagePlaneHeight = 2. * tan(verticalFOVRad / 2.); // Distance to image plane is 1 unit
        // calculate image plane width from pixel screen aspect ratio
        const float_T aspectRatio = width / height;
        imagePlaneDimensions = Vec2(imagePlaneHeight * aspectRatio, imagePlaneHeight);
    }
};

struct OrthographicCamera : public Camera{

    OrthographicCamera(Vec3 position, Vec3 lookAt, Vec3 up, float_T width, float_T height, float_T exposure)
        : Camera(position, lookAt, up, width, height, exposure){ }

    virtual Ray generateRay(Vec2 pixelInScreenSpace) const override{
        // for an orthographic camera, basically just shoot a ray in the look direction, through the pixel center
        return Ray::createExact(
            pixelInWorldSpace(pixelInScreenSpace),
            direction
        );
    }
};

struct PinholePerspectiveCamera : public Camera{
    PinholePerspectiveCamera(
        Vec3 position,
        Vec3 lookAt,
        Vec3 up,
        float_T fovDegrees,
        float_T width,
        float_T height,
        float_T exposure)
        : Camera(position, lookAt, up, width, height, exposure) {
        // Calculate image plane height based on FOV and set image plane dimensions
        setImagePlaneDimensionsFromFOV(fovDegrees);
    }

    /// gets a pixel in pixel screen space, i.e. [0,width]x[0,height]
    /// outputs a ray in world space, i.e. adjusting for the camera's position
    /// Generates a ray from the camera position through the specified pixel
    virtual Ray generateRay(Vec2 pixelInScreenSpace) const override {
        // Use pixelInWorldSpace to get the point on the image plane in world space
        Vec3 pointOnImagePlane = pixelInWorldSpace(pixelInScreenSpace) + /* place image plane 1 unit away from camera */ direction;

        // Calculate ray direction from camera position to point on image plane
        Vec3 rayDirection = (pointOnImagePlane - position).normalized();

        return Ray::createExact(position, rayDirection);
    }
};

/// thin lens camera with depth of field.
/// aperture does not affect the amound of light let-in, only the depth of field,
/// as its easier to adjust the amount of light via the exposure
struct SimplifiedThinLensCamera : public Camera {
    float_T focalLength;
    float_T apertureRadius;
    // Distance to focal plane
    float_T focusDistance; 

    /// takes focal length in mm, focal distance in meters
    SimplifiedThinLensCamera(
        Vec3 position,
        Vec3 direction,
        Vec3 up,
        float_T fovDegrees,
        float_T width,
        float_T height,
        float_T exposure,
        float_T fStop,
        float_T focalLength,       // in mm
        float_T focusDistance)     // in meters
        : Camera(position, direction, up, width, height, exposure),
          focalLength(0.001 * focalLength), // convert mm to meter
          apertureRadius((0.001 * focalLength / fStop) /* = diameter, so halve it to get radius */ * 0.5),
          focusDistance(focusDistance) {
        // NOTE: for a proper (i.e. not simplified) thin lens camera, should use the focal length to calculate the image plane dimensions, and place it behind the lens
        setImagePlaneDimensionsFromFOV(fovDegrees);
    }

    virtual Ray generateRay(Vec2 pixelInScreenSpace) const override {
        // generated by ai

        // get the point on the image plane in world space
        // NOTE for a proper (i.e. not simplified) thin lens camera, should use the focal length as distance to image plane, not just 1 (and place image plane/sensor behind the lens)
        Vec3 pointOnImagePlane = pixelInWorldSpace(pixelInScreenSpace) + direction;

        // calculate the direction from the camera position to the point on the image plane
        Vec3 rayDirection = (pointOnImagePlane - position).normalized();

        // sample a point on the lens aperture
        Vec2 lensSample = sampleAperture();
        Vec3 lensPoint = position + right * lensSample.x + down * lensSample.y;

        // calculate the point on the focal plane
        float_T t = focusDistance / rayDirection.dot(direction);
        Vec3 focalPoint = position + rayDirection * t;

        // calculate the new ray direction from the lens point to the focal point
        Vec3 newRayDirection = (focalPoint - lensPoint).normalized();

        return Ray::createExact(lensPoint, newRayDirection);
    }

private:
    Vec2 sampleAperture() const {
        // uniform random point on the unit disk, times the aperture radius
        float_T theta = 2.0 * PI * randomFloat();
        float_T r = sqrt(randomFloat()) * apertureRadius;
        // Convert from local space to world space using image plane dimensions per pixel
        // for the purposes of the aperture, we need to assume the image plane has its "correct" dimensions, i.e. scale it down by the focal length
        // 1000 to compensate for double mm vs meter conversion in both apertureRadius and focalLength
        return Vec2(r * cos(theta), r * sin(theta)) * 1000. * focalLength;
    }

};

