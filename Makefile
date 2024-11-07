# standard cpp makefile
# g++ 14.2.1

CXX=g++
OUT=raytracer

CXXFLAGS=-Wall -Wextra -Wpedantic -O3 -std=c++2b -I./include
DEBUGFLAGS=-fsanitize=address -fsanitize=undefined -fsanitize=leak -O0 -g

SOURCES=$(wildcard src/*.cpp) raytracer.cpp
OBJECTS=$($(filter-out raytracer.cpp,$(SOURCES)):src/%.cpp=build/%.o) build/raytracer.o

.PHONY: all debug clean

all: setup $(SOURCES)
	export CXXFLAGS="$(CXXFLAGS) -DNDEBUG"
	$(MAKE) $(OUT)

debug: setup $(SOURCES)
	export CXXFLAGS="$(CXXFLAGS) $(DEBUGFLAGS)"
	$(MAKE) $(OUT)

setup:
	mkdir -p build

build/raytracer.o: raytracer.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@
build/%.o: src/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@


# link
$(OUT): $(OBJECTS)
	$(CXX) $^ $(CXXFLAGS) -o $@

clean:
	rm -rf build
	rm -f $(OUT)
