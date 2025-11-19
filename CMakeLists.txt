cmake_minimum_required(VERSION 3.28)
project(MLEngineBenchmarking LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)


# -----------------------
# Compile flags
# -----------------------
set(RELEASE_FLAGS_CXX -march=native -mtune=native -DNDEBUG -fno-lto -Ofast -O3 -lopenblas -fopenmp -w)


# -----------------------
# Link libraries
# -----------------------
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

# -----------------------
# Optionally enable ccache
# -----------------------
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()


# -----------------------
# Build binaries
# -----------------------
file(GLOB CHILDREN RELATIVE "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/src/*")
foreach(CH ${CHILDREN})
    if(IS_DIRECTORY "${CMAKE_SOURCE_DIR}/src/${CH}")
        # Grab source files from the directory
        file(GLOB_RECURSE SRCS CONFIGURE_DEPENDS
            "${CMAKE_SOURCE_DIR}/src/${CH}/*.cpp"
        )

        if(SRCS)
            add_executable(${CH} ${SRCS})
            set_target_properties(${CH} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin"
                OUTPUT_NAME "${CH}"
            )

            target_compile_options(${CH} PRIVATE ${RELEASE_FLAGS_CXX})
            target_link_libraries(${CH} PRIVATE OpenMP::OpenMP_CXX ${BLAS_LIBRARIES})
        endif()
    endif()
endforeach()