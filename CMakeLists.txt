cmake_minimum_required(VERSION 3.16)
project(TitledNNUE CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Native/AVX2 build for fast bitboard + (later) NNUE SIMD. Override with
# -DUSE_NATIVE=OFF for a portable binary.
option(USE_NATIVE "Build for the host CPU (-march=native)" ON)
if(USE_NATIVE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
  if(HAS_MARCH_NATIVE)
    add_compile_options(-march=native)
  endif()
endif()

add_executable(titled-nnue
  src/main.cpp
  src/uci.cpp
  src/search.cpp
  src/eval.cpp
  src/tt.cpp
  src/nnue/nnue.cpp
)

find_package(Threads REQUIRED)
target_link_libraries(titled-nnue PRIVATE Threads::Threads)

target_include_directories(titled-nnue PRIVATE
  ${CMAKE_SOURCE_DIR}/third_party/chess-library
)

# Data-extraction tool: PGN -> filtered, WDL-labeled training positions.
add_executable(extract
  extract/extract.cpp
  extract/filter.cpp
)
target_include_directories(extract PRIVATE
  ${CMAKE_SOURCE_DIR}/third_party/chess-library
)
