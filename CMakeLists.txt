cmake_minimum_required(VERSION 3.28...3.31)
# An empty cached Release flag string suppresses CMake's compiler defaults.
# Repair that state before compiler initialization; preserve nonempty overrides.
foreach(language C CXX)
  set(release_flags "CMAKE_${language}_FLAGS_RELEASE")
  if(DEFINED CACHE{${release_flags}} AND "$CACHE{${release_flags}}" STREQUAL "")
    message(STATUS "Restoring compiler defaults for empty ${release_flags}")
    unset(${release_flags} CACHE)
  endif()
endforeach()
project(skellyforge_native LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(SKELLYFORGE_NATIVE_TESTS "Build Forge native tests" OFF)
include(cpp/cmake/dependencies.cmake)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_package(pybind11 CONFIG REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -c
  "from importlib.metadata import distribution; d=distribution('pybind11'); print(next(d.locate_file(p) for p in d.files if str(p).endswith('dist-info/licenses/LICENSE')))"
  OUTPUT_VARIABLE PYBIND11_LICENSE OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
install(FILES "${PYBIND11_LICENSE}" DESTINATION skellyforge/licenses RENAME PYBIND11_LICENSE COMPONENT python)
if(SKELLYFORGE_NATIVE_TESTS)
  enable_testing()
endif()
add_subdirectory(cpp)
