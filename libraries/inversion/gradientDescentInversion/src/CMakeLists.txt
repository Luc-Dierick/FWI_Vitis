add_library(gradient_descent_inversion_library
    gradientDescentInversion.cpp                    ../include/gradientDescentInversion.h
                                                    ../include/gradientDescentInversionInput.h
    gradientDescentInversionInputCardReader.cpp     ../include/gradientDescentInversionInputCardReader.h)

set( LIBRARY_INCLUDE_DIRS_GRADIENT_DESCENT_INVERSION ${CMAKE_SOURCE_DIR}/libraries/inversion/gradientDescentInversion/include CACHE INTERNAL "" )
target_include_directories(gradient_descent_inversion_library PUBLIC ${LIBRARY_INCLUDE_DIRS_GRADIENT_DESCENT_INVERSION} PRIVATE ${CMAKE_SOURCE_DIR}/libraries/io/include)

target_link_libraries(
    gradient_descent_inversion_library
    core_library
    integral_forward_model_library)
