#pragma once

#include <fstream>
#include <memory>

#include "util.h"

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
        file.seekp(pixelDataStart);
    }

    void flush(){
        file.flush();
    }
};

struct Texture;

std::shared_ptr<Texture> readPPMTexture(std::string_view path);

struct Renderer;

std::unique_ptr<Renderer> jsonFileToRenderer(std::string_view path);
