set(CPP_SOURCES_AND_H_HEADERS 
	cpuClock.cpp    			    ../include/cpuClock.h)

add_library(performance_library ${CPP_SOURCES_AND_H_HEADERS})
target_include_directories(performance_library PRIVATE ${CMAKE_SOURCE_DIR}/libraries/performance/include PRIVATE ${CMAKE_SOURCE_DIR}/libraries/io/include)

target_link_libraries(performance_library PRIVATE Eigen3::Eigen)
