cmake_minimum_required(VERSION 3.1)
project(example)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option(LIBIGL_USE_PREBUILT_LIBRARIES "Use prebuilt libraries"       ON)
option(LIBIGL_USE_STATIC_LIBRARY     "Use libigl as static library" ${LIBIGL_USE_PREBUILT_LIBRARIES})
option(LIBIGL_WITH_CGAL              "Use CGAL"                     OFF)
option(LIBIGL_WITH_COMISO            "Use CoMiso"                   OFF)
option(LIBIGL_WITH_CORK              "Use Cork"                     OFF)
option(LIBIGL_WITH_EMBREE            "Use Embree"                   OFF)
option(LIBIGL_WITH_MATLAB            "Use Matlab"                   OFF)
option(LIBIGL_WITH_MOSEK             "Use MOSEK"                    OFF)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"                   ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"                     ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"                    OFF)
option(LIBIGL_WITH_PNG               "Use PNG"                      OFF)
option(LIBIGL_WITH_TETGEN            "Use Tetgen"                   OFF)
option(LIBIGL_WITH_TRIANGLE          "Use Triangle"                 OFF)
option(LIBIGL_WITH_PREDICATES        "Use exact predicates"         OFF)
option(LIBIGL_WITH_XML               "Use XML"                      OFF)
find_package(LIBIGL REQUIRED QUIET)



# Define executable here
file(GLOB SRCFILES *.cpp)
add_executable(${PROJECT_NAME} ${SRCFILES})

# fix mosek
IF(APPLE)
  INCLUDE(cmake/OSXFixDylibReferences.cmake)
  OSX_FIX_DYLIB_REFERENCES(${PROJECT_NAME} "${MOSEK_LIBRARIES}")
ENDIF()

# For each libigl module, link to target (and flip definition flag)
target_link_libraries(${PROJECT_NAME} igl::core)
list(APPEND modules "cgal" "comiso" "cork" "embree" "matlab" "mosek" "opengl" "opengl_glfw" "opengl_glfw_imgui" "png" "tetgen" "triangle" "predicates" "xml")
foreach(module_name IN LISTS modules)
  string(TOUPPER "LIBIGL_WITH_${module_name}" option_string)
  set(option_variable ${${option_string}})
  if(option_variable)
    target_compile_definitions(${PROJECT_NAME} PRIVATE -D${option_string})
    target_link_libraries(${PROJECT_NAME} "igl::${module_name}")
  endif()
endforeach()

if(USE_MKL)
  find_package(MKL REQUIRED)
  target_compile_definitions(${PROJECT_NAME} PUBLIC -DUSE_MKL)
  target_compile_definitions(${PROJECT_NAME} PUBLIC -DEIGEN_USE_MKL_ALL)
  include_directories(${MKL_INCLUDE_DIR})
  target_link_libraries(${PROJECT_NAME} ${MKL_LIBRARIES})
endif()

if(USE_CHOLMOD)
  message(STATUS "Using fastest solver: Cholmod")
  target_compile_definitions(${PROJECT_NAME} PUBLIC -DUSE_CHOLMOD)
  # heavy handed way of including and linking to SuiteSparse, surely there's a
  # more modern cmake way. This still builds umfpack, spqr and klu though those
  # are not used.
  set(WITH_FORTRAN OFF CACHE BOOL "")
  set(WITH_CUDA    OFF CACHE BOOL "")
  set(WITH_OPENMP  OFF CACHE BOOL "")
  set(WITH_PARTITION OFF CACHE BOOL "")
  set(WITH_DEMOS OFF CACHE BOOL "")
  set(BUILD_CXSPARSE OFF CACHE BOOL "")
  add_subdirectory("SuiteSparse/")
  target_link_libraries(${PROJECT_NAME} cholmod)
endif()
