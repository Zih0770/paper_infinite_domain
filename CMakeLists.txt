cmake_minimum_required(VERSION 3.16)
project(paper_infinite_domain LANGUAGES C CXX)

# Safer builds
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "In-source builds are not allowed. Create a build/ dir and run CMake from there.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")

# ---- Output directories ----
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# ---- MPI ----
find_package(MPI REQUIRED COMPONENTS C CXX)
if(TARGET MPI::MPI_CXX)
  message(STATUS "Found MPI::MPI_CXX")
else()
  message(FATAL_ERROR "MPI::MPI_CXX not found.")
endif()

# ---- MFEM ----
# Preferred: system package config via CMAKE_PREFIX_PATH
find_package(mfem CONFIG QUIET)

if(mfem_FOUND)
  message(STATUS "Using MFEM via package config.")
  # MFEM::mfem provides include dirs + link deps (HYPRE, MPI, etc.)
  set(MFEM_TARGET MFEM::mfem)
else()
  message(STATUS "MFEM package not found; using variables MFEM_INCLUDE / MFEM_LIBRARIES.")
  if(NOT MFEM_INCLUDE OR NOT MFEM_LIBRARIES)
    message(FATAL_ERROR
      "MFEM not found.\n"
      "Either install MFEM with CMake package config (so 'find_package(mfem CONFIG)' works)\n"
      "or set MFEM_INCLUDE and MFEM_LIBRARIES via a preset.")
  endif()
  add_library(mfem_external INTERFACE)
  target_include_directories(mfem_external SYSTEM INTERFACE ${MFEM_INCLUDE})
  target_link_libraries(mfem_external INTERFACE ${MFEM_LIBRARIES} MPI::MPI_CXX)
  set(MFEM_TARGET mfem_external)
endif()

# ---- Optional: headers for MFEM’s enabled backends (no linking) ----
# Provide a semicolon-separated list of prefixes, e.g.
#   /usr/local/cuda;/path/to/raja/installed;/path/to/sundials/installed
# For each prefix we try <prefix>/include and a few CUDA include layouts.
set(EXTRA_INCLUDE_PREFIXES "" CACHE STRING "Prefixes to search for optional headers (RAJA, AMGX, SUNDIALS, Ginkgo, CUDA, etc.).")
if(EXTRA_INCLUDE_PREFIXES)
  foreach(_p IN LISTS EXTRA_INCLUDE_PREFIXES)
    foreach(_cand IN ITEMS
      "${_p}/include"
      "${_p}/targets/x86_64-linux/include"      # CUDA 12 default
      "${_p}/targets/x86_64-linux-gnu/include"  # some CUDA layouts
    )
      if(EXISTS "${_cand}")
        include_directories(SYSTEM "${_cand}")
      endif()
    endforeach()
  endforeach()
endif()

# ---- Optional Gmsh (for mesh tools only) ----
option(ENABLE_GMSH "Build the Gmsh-based mesh tools" OFF)
if(ENABLE_GMSH)
  set(GMSH_INCLUDE_DIR "" CACHE PATH "Path to gmsh include dir containing gmsh.h")
  set(GMSH_LIBRARY "" CACHE FILEPATH "Path to libgmsh.so (or .a)")
  if(NOT (GMSH_INCLUDE_DIR AND GMSH_LIBRARY))
    message(FATAL_ERROR "ENABLE_GMSH=ON but gmsh was not found. Set GMSH_INCLUDE_DIR and GMSH_LIBRARY in your preset.")
  endif()
  message(STATUS "Gmsh includes: ${GMSH_INCLUDE_DIR}")
  message(STATUS "Gmsh library : ${GMSH_LIBRARY}")

  # Mesh helper lib used only by the gmsh tools
  set(MESH_GEOM_SOURCES
    ${CMAKE_SOURCE_DIR}/mesh/common.cpp
    ${CMAKE_SOURCE_DIR}/mesh/CircularMesh.cpp
  )
  add_library(meshgeom STATIC ${MESH_GEOM_SOURCES})
  target_include_directories(meshgeom PUBLIC
    ${CMAKE_SOURCE_DIR}/mesh
    ${GMSH_INCLUDE_DIR}
  )
endif()

# ---- Solver ----
#add_executable(Poisson_big_domain src/Poisson_big_domain.cpp)
#target_include_directories(Poisson_big_domain PRIVATE ${CMAKE_SOURCE_DIR}/src)
#target_link_libraries(Poisson_big_domain PRIVATE ${MFEM_TARGET} MPI::MPI_CXX)

add_executable(ex2p src/ex2p.cpp)
target_include_directories(ex2p PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(ex2p PRIVATE ${MFEM_TARGET} MPI::MPI_CXX)

add_executable(ex2p_ext src/ex2p_ext.cpp)
target_include_directories(ex2p_ext PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(ex2p_ext PRIVATE ${MFEM_TARGET} MPI::MPI_CXX)

add_executable(ex2p_0 src/ex2p_0.cpp)
target_include_directories(ex2p_0 PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(ex2p_0 PRIVATE ${MFEM_TARGET} MPI::MPI_CXX)

#add_executable(Poisson_DtN src/Poisson_DtN.cpp)
#target_include_directories(Poisson_DtN PRIVATE ${CMAKE_SOURCE_DIR}/src)
#target_link_libraries(Poisson_DtN PRIVATE ${MFEM_TARGET} MPI::MPI_CXX)

#add_executable(Poisson_DtN_bp src/Poisson_DtN_bp.cpp)
#target_include_directories(Poisson_DtN_bp PRIVATE ${CMAKE_SOURCE_DIR}/src)
#target_link_libraries(Poisson_DtN_bp PRIVATE ${MFEM_TARGET} MPI::MPI_CXX)

#add_executable(Poisson_DtN_test src/Poisson_DtN_test.cpp)
#target_include_directories(Poisson_DtN_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
#target_link_libraries(Poisson_DtN_test PRIVATE ${MFEM_TARGET} MPI::MPI_CXX)

#add_executable(Poisson_multipole src/Poisson_multipole.cpp)
#target_include_directories(Poisson_multipole PRIVATE ${CMAKE_SOURCE_DIR}/src)
#target_link_libraries(Poisson_multipole PRIVATE ${MFEM_TARGET} MPI::MPI_CXX)

# ---- Gmsh tools (only if enabled) ----
if(ENABLE_GMSH)
  set(MESH_GENERATORS
    mesh/offset_2d.cpp
    mesh/offset_3d.cpp
    mesh/simple_2d.cpp
    mesh/simple_3d.cpp
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
  endforeach()
endif()

