# standard cpp makefile
# g++ 14.2.1

CXX=g++
OUT=raytracer

CXXFLAGS=-Wall -Wextra -Wpedantic -O3 -std=c++2b -I./include -fopenmp
DEBUGFLAGS=-fsanitize=address -fsanitize=undefined -fsanitize=leak -O0 -g

SOURCES=$(wildcard src/*.cpp)
DEPS=$(wildcard src/*.h)
OBJECTS=$(SOURCES:src/%.cpp=build/%.o)

.PHONY: all debug clean

all: setup $(SOURCES)
	# for some reason, export CXXFLAGS+=... doesn't work
	$(MAKE) $(OUT) CXXFLAGS="$(CXXFLAGS) -DNDEBUG"

debug: setup $(SOURCES)
	$(MAKE) $(OUT) CXXFLAGS="$(CXXFLAGS) $(DEBUGFLAGS)"


setup:
	mkdir -p build

build/%.o: src/%.cpp $(DEPS)
	$(CXX) $< $(CXXFLAGS) -c -o $@


# link
$(OUT): $(OBJECTS)
	$(CXX) $^ $(CXXFLAGS) -o $@

clean:
	rm -rf build
	rm -f $(OUT)
