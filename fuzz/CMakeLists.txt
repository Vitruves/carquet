# Fuzzing targets for carquet
#
# Build with:
#   cmake -B build-fuzz -DCMAKE_C_COMPILER=clang -DCARQUET_BUILD_FUZZ=ON \
#         -DCMAKE_BUILD_TYPE=Debug
#   cmake --build build-fuzz
#
# Or for AFL++:
#   CC=afl-clang-fast cmake -B build-afl -DCARQUET_BUILD_FUZZ=ON \
#         -DCARQUET_FUZZ_ENGINE=AFL -DCMAKE_BUILD_TYPE=Debug
#   cmake --build build-afl

cmake_minimum_required(VERSION 3.16)

# Fuzzing options
option(CARQUET_FUZZ_SANITIZERS "Enable sanitizers for fuzzing" ON)
set(CARQUET_FUZZ_ENGINE "libFuzzer" CACHE STRING "Fuzzing engine: libFuzzer or AFL")

# Sanitizer flags
if(CARQUET_FUZZ_SANITIZERS)
    set(FUZZ_SANITIZER_FLAGS "-fsanitize=address" "-fsanitize=undefined" "-fno-omit-frame-pointer")
endif()

# Function to add a fuzz target
function(add_fuzz_target TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    target_link_libraries(${TARGET_NAME} PRIVATE carquet)
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
    )

    if(CARQUET_FUZZ_ENGINE STREQUAL "libFuzzer")
        target_compile_options(${TARGET_NAME} PRIVATE -fsanitize=fuzzer ${FUZZ_SANITIZER_FLAGS})
        target_link_options(${TARGET_NAME} PRIVATE -fsanitize=fuzzer ${FUZZ_SANITIZER_FLAGS})
    elseif(CARQUET_FUZZ_ENGINE STREQUAL "AFL")
        target_compile_definitions(${TARGET_NAME} PRIVATE AFL_MAIN)
        if(CARQUET_FUZZ_SANITIZERS)
            target_compile_options(${TARGET_NAME} PRIVATE ${FUZZ_SANITIZER_FLAGS})
            target_link_options(${TARGET_NAME} PRIVATE ${FUZZ_SANITIZER_FLAGS})
        endif()
    endif()
endfunction()

# Add fuzz targets
# Note: fuzz_reader covers all code paths through the public API
# fuzz_compression and fuzz_thrift test internal decoders directly
add_fuzz_target(fuzz_reader fuzz_reader.c)
add_fuzz_target(fuzz_compression fuzz_compression.c)
# Encodings fuzzer tests RLE, delta, and dictionary decoders
add_fuzz_target(fuzz_encodings fuzz_encodings.c)

# Thrift fuzzer tests the Thrift compact protocol decoder
add_fuzz_target(fuzz_thrift fuzz_thrift.c)

# Roundtrip fuzzer tests encode->decode consistency
add_fuzz_target(fuzz_roundtrip fuzz_roundtrip.c)

# Writer fuzzer tests schema creation and data writing
add_fuzz_target(fuzz_writer fuzz_writer.c)

# Create corpus directories
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/corpus_reader)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/corpus_compression)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/corpus_encodings)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/corpus_thrift)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/corpus_roundtrip)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/corpus_writer)
