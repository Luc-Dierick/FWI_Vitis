add_library(integral_forward_model_library
    integralForwardModel.cpp                    ../include/integralForwardModel.h
                                                ../include/integralForwardModelInput.h
    integralForwardModelInputCardReader.cpp     ../include/integralForwardModelInputCardReader.h)

set( LIBRARY_INCLUDE_DIRS_INTEGRAL_FORWARD_MODEL ${CMAKE_SOURCE_DIR}/libraries/forwardModel/integral/include CACHE INTERNAL "" )
target_include_directories(integral_forward_model_library 
PUBLIC ${LIBRARY_INCLUDE_DIRS_INTEGRAL_FORWARD_MODEL}
PRIVATE ${CMAKE_SOURCE_DIR}/libraries/io/include
PRIVATE ${CMAKE_SOURCE_DIR}/libraries/forwardModel/interface/include
PRIVATE ${CMAKE_SOURCE_DIR}/libraries/utilities/include)

target_link_libraries(integral_forward_model_library core_library)
