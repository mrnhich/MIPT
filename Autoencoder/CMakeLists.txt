set(TARGET_NAME Autoencoder)
set(Autoencoder_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

set(Autoencoder_SOURCES
    ${Autoencoder_SOURCE_DIR}/Autoencoder.cpp
)

set(Autoencoder_HEADERS
    ${Autoencoder_SOURCE_DIR}/Autoencoder.h
)

#include_directories(
#)

add_library(${TARGET_NAME} ${LIBRARY_TYPE} ${Autoencoder_SOURCES} ${Autoencoder_HEADERS})
#set_target_properties(${TARGET_NAME} PROPERTIES DEFINE_SYMBOL Autoencoder_EXPORTS)
add_executable(${TARGET_NAME}-app ${Autoencoder_SOURCE_DIR}/main.cpp)
target_link_libraries(${TARGET_NAME}-app ${TARGET_NAME})
ADD_DEFINITIONS("-Wall" "-fPIC")
