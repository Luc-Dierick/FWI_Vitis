set(CPP_SOURCES_AND_H_HEADERS 
    createChiCSV.cpp       		        ../include/createChiCSV.h
									    ../include/genericInput.h
									    ../include/json.h
	log.cpp								../include/log.h
    ReadJsonHelper.cpp                  ../include/ReadJsonHelper.h
    genericInputCardReader.cpp          ../include/genericInputCardReader.h
    inputCardReader.cpp                 ../include/inputCardReader.h
	chiIntegerVisualisation.cpp		    ../include/chiIntegerVisualisation.h
	csvReader.cpp     			        ../include/csvReader.h
	progressBar.cpp       			    ../include/progressBar.h
	argumentReader.cpp       			../include/argumentReader.h)

add_library(io_library ${CPP_SOURCES_AND_H_HEADERS})

set( LIBRARY_INCLUDE_DIRS_CORE ${CMAKE_SOURCE_DIR}/libraries/core/include CACHE INTERNAL "" )
target_include_directories(io_library PUBLIC ${LIBRARY_INCLUDE_DIRS_CORE} PRIVATE ${CMAKE_SOURCE_DIR}/libraries/io/include)
