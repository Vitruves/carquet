cmake_minimum_required(VERSION 3.16)
project(carquet VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# macOS deployment target - use current system version to avoid library version mismatch
if(APPLE)
    # Get the current macOS version and use it as deployment target
    execute_process(
        COMMAND sw_vers -productVersion
        OUTPUT_VARIABLE MACOS_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # Extract major version (e.g., "15" from "15.2")
    string(REGEX MATCH "^[0-9]+" MACOS_MAJOR_VERSION "${MACOS_VERSION}")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "${MACOS_MAJOR_VERSION}.0" CACHE STRING "Minimum macOS version" FORCE)

    # Suppress linker warnings about library version mismatches from Homebrew LLVM
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -w")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -w")
endif()

# Build options
option(CARQUET_BUILD_TESTS "Build tests" ON)
option(CARQUET_BUILD_SHARED "Build shared library" OFF)
option(CARQUET_ENABLE_SSE "Enable SSE optimizations" ON)
option(CARQUET_ENABLE_AVX2 "Enable AVX2 optimizations" ON)
option(CARQUET_ENABLE_AVX512 "Enable AVX-512 optimizations" ON)
option(CARQUET_ENABLE_NEON "Enable NEON optimizations" ON)
option(CARQUET_ENABLE_SVE "Enable SVE optimizations" OFF)

# Compiler detection
if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    set(CARQUET_COMPILER_GCC_LIKE TRUE)
elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
    set(CARQUET_COMPILER_MSVC TRUE)
endif()

# Architecture detection
include(CheckCSourceCompiles)
check_c_source_compiles("
    #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    int main() { return 0; }
    #else
    #error Not x86
    #endif
" CARQUET_ARCH_X86)

check_c_source_compiles("
    #if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
    int main() { return 0; }
    #else
    #error Not ARM
    #endif
" CARQUET_ARCH_ARM)

# Compiler flags
if(CARQUET_COMPILER_GCC_LIKE)
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

    # Debug/Release flags
    set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DCARQUET_DEBUG")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
endif()

# Source files
set(CARQUET_CORE_SOURCES
    src/core/arena.c
    src/core/buffer.c
    src/core/bitpack.c
    src/core/endian.c
    src/core/error.c
)

set(CARQUET_THRIFT_SOURCES
    src/thrift/thrift_decode.c
    src/thrift/thrift_encode.c
    src/thrift/parquet_types.c
)

set(CARQUET_ENCODING_SOURCES
    src/encoding/plain.c
    src/encoding/rle.c
    src/encoding/delta.c
    src/encoding/delta_length.c
    src/encoding/delta_strings.c
    src/encoding/dictionary.c
    src/encoding/byte_stream_split.c
)

set(CARQUET_COMPRESSION_SOURCES
    src/compression/lz4.c
    src/compression/snappy.c
    src/compression/zstd.c
    src/compression/gzip.c
)

set(CARQUET_SIMD_SOURCES
    src/simd/detect.c
    src/simd/dispatch.c
)

set(CARQUET_READER_SOURCES
    src/reader/file_reader.c
    src/reader/row_group_reader.c
    src/reader/column_reader.c
    src/reader/page_reader.c
    src/reader/batch_reader.c
    src/reader/statistics.c
    src/reader/mmap_reader.c
)

set(CARQUET_WRITER_SOURCES
    src/writer/file_writer.c
    src/writer/row_group_writer.c
    src/writer/column_writer.c
    src/writer/page_writer.c
)

set(CARQUET_METADATA_SOURCES
    src/metadata/schema.c
    src/metadata/statistics.c
    src/metadata/bloom_filter.c
    src/metadata/page_index.c
)

set(CARQUET_UTIL_SOURCES
    src/util/crc32.c
    src/util/xxhash.c
)

# x86 SIMD sources (conditionally compiled)
if(CARQUET_ARCH_X86)
    set(CARQUET_SIMD_X86_SOURCES
        src/simd/x86/sse_ops.c
        src/simd/x86/avx2_ops.c
        src/simd/x86/avx512_ops.c
    )

    # Set architecture-specific compiler flags
    if(CARQUET_COMPILER_GCC_LIKE)
        set_source_files_properties(src/simd/x86/sse_ops.c PROPERTIES
            COMPILE_FLAGS "-msse4.2")
        set_source_files_properties(src/simd/x86/avx2_ops.c PROPERTIES
            COMPILE_FLAGS "-mavx2 -mbmi2")
        set_source_files_properties(src/simd/x86/avx512_ops.c PROPERTIES
            COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512vl")
    endif()
endif()

# ARM SIMD sources (conditionally compiled)
if(CARQUET_ARCH_ARM)
    set(CARQUET_SIMD_ARM_SOURCES
        src/simd/arm/neon_ops.c
        src/simd/arm/sve_ops.c
    )

    if(CARQUET_COMPILER_GCC_LIKE AND CARQUET_ENABLE_SVE)
        set_source_files_properties(src/simd/arm/sve_ops.c PROPERTIES
            COMPILE_FLAGS "-march=armv8-a+sve")
    endif()
endif()

# Combine all sources
set(CARQUET_SOURCES
    ${CARQUET_CORE_SOURCES}
    ${CARQUET_THRIFT_SOURCES}
    ${CARQUET_ENCODING_SOURCES}
    ${CARQUET_COMPRESSION_SOURCES}
    ${CARQUET_SIMD_SOURCES}
    ${CARQUET_READER_SOURCES}
    ${CARQUET_WRITER_SOURCES}
    ${CARQUET_METADATA_SOURCES}
    ${CARQUET_UTIL_SOURCES}
)

if(CARQUET_ARCH_X86)
    list(APPEND CARQUET_SOURCES ${CARQUET_SIMD_X86_SOURCES})
endif()
if(CARQUET_ARCH_ARM)
    list(APPEND CARQUET_SOURCES ${CARQUET_SIMD_ARM_SOURCES})
endif()

# Create library
if(CARQUET_BUILD_SHARED)
    add_library(carquet SHARED ${CARQUET_SOURCES})
    target_compile_definitions(carquet PRIVATE CARQUET_BUILD_SHARED)
else()
    add_library(carquet STATIC ${CARQUET_SOURCES})
endif()

target_include_directories(carquet
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Architecture definitions
if(CARQUET_ARCH_X86)
    target_compile_definitions(carquet PRIVATE CARQUET_ARCH_X86)
    if(CARQUET_ENABLE_SSE)
        target_compile_definitions(carquet PRIVATE CARQUET_ENABLE_SSE)
    endif()
    if(CARQUET_ENABLE_AVX2)
        target_compile_definitions(carquet PRIVATE CARQUET_ENABLE_AVX2)
    endif()
    if(CARQUET_ENABLE_AVX512)
        target_compile_definitions(carquet PRIVATE CARQUET_ENABLE_AVX512)
    endif()
endif()

if(CARQUET_ARCH_ARM)
    target_compile_definitions(carquet PRIVATE CARQUET_ARCH_ARM)
    if(CARQUET_ENABLE_NEON)
        target_compile_definitions(carquet PRIVATE CARQUET_ENABLE_NEON)
    endif()
    if(CARQUET_ENABLE_SVE)
        target_compile_definitions(carquet PRIVATE CARQUET_ENABLE_SVE)
    endif()
endif()

# Tests
if(CARQUET_BUILD_TESTS)
    enable_testing()

    # Helper function to configure test targets
    function(add_carquet_test TEST_NAME TEST_SOURCE)
        add_executable(${TEST_NAME} ${TEST_SOURCE})
        target_link_libraries(${TEST_NAME} PRIVATE carquet)
        target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        if(CARQUET_COMPILER_GCC_LIKE)
            target_compile_options(${TEST_NAME} PRIVATE -Wno-unused-result)
        endif()
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endfunction()

    add_carquet_test(test_core tests/test_core.c)
    add_carquet_test(test_thrift tests/test_thrift.c)
    add_carquet_test(test_encodings tests/test_encodings.c)
    add_carquet_test(test_reader tests/test_reader.c)
    add_carquet_test(test_encodings_extended tests/test_encodings_extended.c)
    add_carquet_test(test_compression tests/test_compression.c)
    add_carquet_test(test_utils tests/test_utils.c)
    add_carquet_test(test_production tests/test_production.c)
    add_carquet_test(test_maturity tests/test_maturity.c)
endif()

# Examples
option(CARQUET_BUILD_EXAMPLES "Build examples" ON)
if(CARQUET_BUILD_EXAMPLES)
    # Helper function to configure example targets
    function(add_carquet_example EXAMPLE_NAME EXAMPLE_SOURCE)
        add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
        target_link_libraries(${EXAMPLE_NAME} PRIVATE carquet)
        if(CARQUET_COMPILER_GCC_LIKE)
            target_compile_options(${EXAMPLE_NAME} PRIVATE -Wno-unused-result)
        endif()
    endfunction()

    add_carquet_example(example_basic_write_read examples/basic_write_read.c)
    add_carquet_example(example_data_types examples/data_types.c)
    add_carquet_example(example_compression examples/compression_codecs.c)
    add_carquet_example(example_nullable examples/nullable_columns.c)
endif()

# Benchmarks
option(CARQUET_BUILD_BENCHMARKS "Build benchmarks" ON)
if(CARQUET_BUILD_BENCHMARKS)
    add_executable(benchmark_carquet benchmark/benchmark_carquet.c)
    target_link_libraries(benchmark_carquet PRIVATE carquet)
    if(CARQUET_COMPILER_GCC_LIKE)
        target_compile_options(benchmark_carquet PRIVATE -Wno-unused-result)
    endif()
endif()

# Installation
include(GNUInstallDirs)
install(TARGETS carquet
    EXPORT carquetTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/carquet
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT carquetTargets
    FILE carquetTargets.cmake
    NAMESPACE carquet::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/carquet
)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/carquetConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/carquetConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/carquetConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/carquet
)
