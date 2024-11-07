# standard cpp makefile

CXX=g++
OUT=render

CXXFLAGS=-Wall -Wextra -Wpedantic -O3 -std=c++2b -fno-rtti
DEBUGFLAGS=-fsanitize=address -fsanitize=undefined -fsanitize=leak -O0 -g

SOURCES=$(wildcard src/*.cpp)
OBJECTS=$(SOURCES:src/%.cpp=build/%.o)

.PHONY: all debug clean

all: setup $(SOURCES)
	export CXXFLAGS="$(CXXFLAGS) -DNDEBUG"
	$(MAKE) $(OUT)

debug: setup $(SOURCES)
	export CXXFLAGS="$(CXXFLAGS) $(DEBUGFLAGS)"
	$(MAKE) $(OUT)

setup:
	mkdir -p build

build/%.o: src/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

# link
$(OUT): $(OBJECTS)
	$(CXX) $^ $(CXXFLAGS) -o $@

clean:
	rm -rf build
	rm -f $(OUT)
