# tested with gcc 14.2.1

CXX=g++
OUT=raytracer

# -Ofast is ~10% faster than -O3, and we dont care so much about fp precision
CXXFLAGS=-Wall -Wextra -Wpedantic -Wunused-result -Ofast -std=c++2b -I./include -fopenmp -fopenacc -fno-exceptions -fno-rtti
DEBUGFLAGS=-fsanitize=address -fsanitize=undefined -fsanitize=leak -O0 -g -fexceptions

SOURCES=$(wildcard src/*.cpp)
DEPS=$(wildcard src/*.h)
OBJECTS=$(SOURCES:src/%.cpp=build/%.o)

.PHONY: release relWithDebInfo debug clean pgo-generate pgo-utilize setup clang-tidy

release: setup $(SOURCES)
	[ -f build/isRelease ] || $(MAKE) clean && $(MAKE) setup
	touch build/isRelease
	$(MAKE) $(OUT) CXXFLAGS="$(CXXFLAGS) -DNDEBUG"

relWithDebInfo: setup $(SOURCES)
	[ -f build/isRelWithDebInfo ] || $(MAKE) clean && $(MAKE) setup
	touch build/isRelWithDebInfo
	$(MAKE) $(OUT) CXXFLAGS="$(CXXFLAGS) -g -DNDEBUG"

debug: setup $(SOURCES)
	[ -f build/isDebug ] || $(MAKE) clean && $(MAKE) setup
	touch build/isDebug
	$(MAKE) $(OUT) CXXFLAGS="$(CXXFLAGS) $(DEBUGFLAGS)"

# PGO results in ~ 10% speedup for cornell box (when trained on cornell box)
pgo-generate: setup $(SOURCES)
	$(MAKE) $(OUT) CXXFLAGS="$(CXXFLAGS) -DNDEBUG -fprofile-generate"

pgo-utilize: setup $(SOURCES)
	$(MAKE) $(OUT) CXXFLAGS="$(CXXFLAGS) -DNDEBUG -fprofile-correction -fprofile-use"


clang-tidy:
	# bugprone-unchecked-optional-access check crashes clang
	clang-tidy -checks=*,-bugprone-unchecked-optional-access,-readability-identifier-length,-*narrowing*,-llvmlibc-callee-namespace src/*.cpp src/*.h -- -x c++ $(CXXFLAGS)

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
