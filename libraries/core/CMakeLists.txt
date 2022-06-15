add_subdirectory(src)
if(NOT IntelSYCL_FOUND)
add_subdirectory(test)
endif()
