include_directories(../include)

set(CPP_SOURCES_AND_H_HEADERS_TEST_UTILS
    		              include/ForwardModelMock.h)

add_library(test_utils_library ${CPP_SOURCES_AND_H_HEADERS_TEST_UTILS})

target_include_directories(test_utils_library PUBLIC ./include ${CMAKE_SOURCE_DIR}/libraries/io/include PRIVATE ${CMAKE_SOURCE_DIR}/libraries/io/include PRIVATE ${CMAKE_SOURCE_DIR}/libraries/utilities/include)

target_link_libraries(test_utils_library Eigen3::Eigen)
target_link_libraries(test_utils_library core_library)
