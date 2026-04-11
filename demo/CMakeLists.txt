cmake_minimum_required(VERSION 3.10)

add_library(demo_executor STATIC
    executor/run_config.h
    executor/memory_tracker.h
    executor/memory_tracker.cpp
    executor/io_utils.h
    executor/io_utils.cpp
    executor/ncnn_executor.h
    executor/ncnn_executor.cpp
)

target_include_directories(demo_executor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/test
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}/src
)

target_link_libraries(demo_executor PRIVATE ncnn)

set_target_properties(demo_executor PROPERTIES FOLDER "demo")
set_target_properties(demo_executor PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(demo_inference_test
    test/inference_test_main.cpp
    test/inference_test_case.h
    test/inference_test_case.cpp
)

target_link_libraries(demo_inference_test PRIVATE demo_executor ncnn)

target_include_directories(demo_inference_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/test
)

set_target_properties(demo_inference_test PROPERTIES FOLDER "demo")

add_dependencies(demo_inference_test demo_executor)
