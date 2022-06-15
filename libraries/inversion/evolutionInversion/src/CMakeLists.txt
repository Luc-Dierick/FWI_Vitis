
add_library(evolution_inversion_library
    evolutionInversion.cpp                      ../include/evolutionInversion.h
                                                ../include/evolutionInversionInput.h
    evolutionInversionInputCardReader.cpp       ../include/evolutionInversionInputCardReader.h)

set( LIBRARY_INCLUDE_DIRS_EVOLUTION_INVERSION ${CMAKE_SOURCE_DIR}/libraries/inversion/evolutionInversion/include CACHE INTERNAL "" )
target_include_directories(evolution_inversion_library PUBLIC ${LIBRARY_INCLUDE_DIRS_EVOLUTION_INVERSION} PRIVATE ${CMAKE_SOURCE_DIR}/libraries/io/include)

target_link_libraries(
    evolution_inversion_library
    core_library
    integral_forward_model_library)
