add_library(conjugate_gradient_inversion_library
    conjugateGradientInversion.cpp                      ../include/conjugateGradientInversion.h
    conjugateGradientOpenMPinversion.cpp                ../include/conjugateGradientOpenMPinversion.h
    conjugateGradientMPIInversion.cpp                   ../include/conjugateGradientMPIInversion.h
    conjugateGradientDataParallelCppInversion.cpp       ../include/conjugateGradientDataParallelCppInversion.h


                                                        ../include/conjugateGradientInversionInput.h
    conjugateGradientInversionInputCardReader.cpp       ../include/conjugateGradientInversionInputCardReader.h
                                                        ../include/deltaAmplification.h
                                                        ../include/iter1.h
                                                        ../include/regularization.h)

set( LIBRARY_INCLUDE_DIRS_CONJUGATE_GRADIENT_INVERSION ${CMAKE_SOURCE_DIR}/libraries/inversion/conjugateGradientInversion/include CACHE INTERNAL "" )
target_include_directories(conjugate_gradient_inversion_library PUBLIC ${LIBRARY_INCLUDE_DIRS_CONJUGATE_GRADIENT_INVERSION} ${CMAKE_SOURCE_DIR}/libraries/io/include)

target_link_libraries(
    conjugate_gradient_inversion_library
    core_library
    integral_forward_model_library
    io_library)
