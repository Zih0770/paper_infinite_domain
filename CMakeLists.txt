# Generate compile_commands.json for clangd/IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

cmake_minimum_required(VERSION 3.5)
project(paper_infinite_domain LANGUAGES CXX)

# Disallow in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Make a build/ dir and run CMake from there.")
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")

find_package(MPI REQUIRED CXX)

# ---- Output directories (set BEFORE any targets) ----
# Libraries still go under build/, executables go to project_root/bin
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Optional: extra module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# ---- Core dependencies ----
# MFEM + friends
set(MFEM_LIBRARIES
  "/home/sssou/local/mfem/lib/libmfem.so"
  "/home/sssou/source/petsc/arch-opt/lib/libmpi.so"
  "/home/sssou/source/hypre/build/lib/libHYPRE.so"
  "/home/sssou/source/mfemElasticity/installed/lib/libmfemElasticity.a"
)
set(MFEM_INCLUDE
  "/home/sssou/local/mfem/include"
  "/home/sssou/source/mfemElasticity/installed/include"
)
include_directories(${MFEM_INCLUDE})

# CUDA
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# RAJA
find_path(RAJA_INCLUDE_DIR RAJA/RAJA.hpp HINTS /home/sssou/source/raja/installed/include)
find_library(RAJA_LIBRARY RAJA HINTS /home/sssou/source/raja/installed/lib)
include_directories(${RAJA_INCLUDE_DIR})

# AMGX
find_path(AMGX_INCLUDE_DIR amgx_c.h HINTS /home/sssou/source/amgx/installed/include)
find_library(AMGX_LIBRARY amgxsh HINTS /home/sssou/source/amgx/installed/lib)
include_directories(${AMGX_INCLUDE_DIR})

# SUNDIALS
set(SUNDIALS_DIR "/home/sssou/source/sundials/installed")
find_path(SUNDIALS_INCLUDE_DIR sundials/sundials_config.h HINTS ${SUNDIALS_DIR}/include)
find_library(SUNDIALS_LIBRARY sundials_cvode HINTS ${SUNDIALS_DIR}/lib)
include_directories(${SUNDIALS_INCLUDE_DIR})

# Ginkgo
set(GINKGO_DIR "/home/sssou/source/ginkgo/installed")
find_path(GINKGO_INCLUDE_DIR ginkgo/ginkgo.hpp HINTS ${GINKGO_DIR}/include)
find_library(GINKGO_LIBRARY ginkgo HINTS ${GINKGO_DIR}/lib)
include_directories(${GINKGO_INCLUDE_DIR})

# PETSc/MPI (using PETSc's MPI here)
set(PETSC_DIR "/home/sssou/source/petsc")
set(PETSC_ARCH "arch-opt")
include_directories("${PETSC_DIR}/${PETSC_ARCH}/include")
include_directories("${PETSC_DIR}/include")
link_directories("${PETSC_DIR}/${PETSC_ARCH}/lib")
find_package(Boost REQUIRED)
#set(MPI_CXX_LIBRARIES "${PETSC_DIR}/${PETSC_ARCH}/lib/libmpi.so")
#set(MPI_CXX_INCLUDE_PATH "${PETSC_DIR}/${PETSC_ARCH}/include")
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

# ---- Prebuilt giafem from the other project ----
set(GIAFEM_INCLUDE_DIR "/home/sssou/giafem/include")
set(GIAFEM_LIBRARY     "/home/sssou/giafem/build/libgiafem_lib.a")

add_library(giafem::giafem UNKNOWN IMPORTED)
set_target_properties(giafem::giafem PROPERTIES
  IMPORTED_LOCATION "${GIAFEM_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${GIAFEM_INCLUDE_DIR}"
)
# If libgiafem_lib.a needs MFEM/MPI at link time, propagate them:
set_property(TARGET giafem::giafem APPEND PROPERTY
  INTERFACE_LINK_LIBRARIES "${MFEM_LIBRARIES};${MPI_CXX_LIBRARIES}"
)

# ---- Gmsh (header + library), target-scoped wiring ----
# gmsh.h is directly under /home/sssou/local/gmsh/include
set(GMSH_INCLUDE_DIR "/home/sssou/local/gmsh/include")
find_library(GMSH_LIBRARY NAMES gmsh HINTS /home/sssou/local/gmsh/lib /usr/local/lib /usr/lib)
if(NOT GMSH_LIBRARY)
  message(FATAL_ERROR "Could not find libgmsh in /home/sssou/local/gmsh/lib")
endif()

# ---- Mesh geometry helper sources (definitions for createCircle/createSphere) ----
set(MESH_GEOM_SOURCES
  ${CMAKE_SOURCE_DIR}/mesh/common.cpp
  ${CMAKE_SOURCE_DIR}/mesh/CircularMesh.cpp
)
add_library(meshgeom STATIC ${MESH_GEOM_SOURCES})
# Headers for your mesh code + Gmsh header path (because these .cpp include <gmsh.h>)
target_include_directories(meshgeom PUBLIC
  ${CMAKE_SOURCE_DIR}/mesh
  ${GMSH_INCLUDE_DIR}
)

# ---- Executables ----
# (No 'test' target since src/test.cpp is empty / no main())
add_executable(Poisson_big_domain src/Poisson_big_domain.cpp)
target_link_libraries(Poisson_big_domain PRIVATE
  ${MFEM_LIBRARIES}
  MPI::MPI_CXX
)

# Mesh generators that use Gmsh and the helper lib
set(MESH_GENERATORS
  mesh/offset_2d.cpp
  mesh/offset_3d.cpp
)

foreach(mesh_src ${MESH_GENERATORS})
  get_filename_component(exec_name ${mesh_src} NAME_WE)
  add_executable(${exec_name} ${mesh_src})
  target_include_directories(${exec_name} PRIVATE
    ${GMSH_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/mesh
  )
  target_link_libraries(${exec_name} PRIVATE
    ${GMSH_LIBRARY}
    meshgeom
  )
  # If any executable also uses giafem symbols/headers, add:
  # target_link_libraries(${exec_name} PRIVATE giafem::giafem)
endforeach()

