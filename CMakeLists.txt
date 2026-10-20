cmake_minimum_required(VERSION 3.16)
project(ptx-forge VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PTXFORGE_BUILD_TESTS "Build the unit tests" ON)
option(PTXFORGE_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

function(ptxforge_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive-)
    if(PTXFORGE_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    if(PTXFORGE_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  endif()
endfunction()

# ---------------------------------------------------------------------------
# Library: front end, IR, passes, backend. The CUDA driver is loaded at run time
# (nvcuda.dll / libcuda.so.1), so neither the CUDA SDK nor a GPU is needed to build.
# ---------------------------------------------------------------------------
add_library(ptxforge STATIC
  src/frontend/Lexer.cpp
  src/frontend/Parser.cpp
  src/ir/PTXInstruction.cpp
  src/ir/ControlFlowGraph.cpp
  src/ir/PTXPrinter.cpp
  src/passes/LivenessAnalysis.cpp
  src/passes/DeadCodeElim.cpp
  src/passes/RegAllocChaitin.cpp
  src/backend/ElfCubinParser.cpp
  src/backend/CubinProvider.cpp
  src/backend/DriverJITLinker.cpp
  src/Pipeline.cpp)
target_include_directories(ptxforge PUBLIC include)
target_link_libraries(ptxforge PUBLIC ${CMAKE_DL_LIBS})
ptxforge_warnings(ptxforge)

add_executable(ptx-forge tools/ptx-forge-cli.cpp)
target_link_libraries(ptx-forge PRIVATE ptxforge)
if(MINGW)
  target_link_options(ptx-forge PRIVATE -static)  # no MinGW runtime DLLs needed on PATH
endif()
ptxforge_warnings(ptx-forge)

# ---------------------------------------------------------------------------
# Tests: GoogleTest if installed, otherwise the bundled single-header shim.
# ---------------------------------------------------------------------------
if(PTXFORGE_BUILD_TESTS)
  enable_testing()
  find_package(GTest QUIET)
  add_executable(ptx-forge-tests
    test/unit/FrontendTests.cpp
    test/unit/PassTests.cpp
    test/unit/BackendTests.cpp)
  target_include_directories(ptx-forge-tests PRIVATE test/unit)
  target_compile_definitions(ptx-forge-tests PRIVATE
    PTXFORGE_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test")
  target_link_libraries(ptx-forge-tests PRIVATE ptxforge)
  if(GTest_FOUND)
    target_compile_definitions(ptx-forge-tests PRIVATE PTXFORGE_USE_GTEST)
    target_link_libraries(ptx-forge-tests PRIVATE GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(ptx-forge-tests)
  else()
    message(STATUS "GoogleTest not found: using the built-in test shim")
    target_sources(ptx-forge-tests PRIVATE test/unit/TestMain.cpp)
    add_test(NAME ptx-forge-tests COMMAND ptx-forge-tests)
  endif()
endif()

install(TARGETS ptx-forge RUNTIME DESTINATION bin)
