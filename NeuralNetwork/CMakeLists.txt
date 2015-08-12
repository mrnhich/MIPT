set(TARGET_NAME NeuralNetwork)
set(NeuralNetwork_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

set(NeuralNetwork_SOURCES
    ${NeuralNetwork_SOURCE_DIR}/NeuralNetwork.cpp
)

set(NeuralNetwork_HEADERS
    ${NeuralNetwork_SOURCE_DIR}/NeuralNetwork.h
)

#include_directories(
#)

add_library(${TARGET_NAME} ${LIBRARY_TYPE} ${NeuralNetwork_SOURCES} ${NeuralNetwork_HEADERS})
#set_target_properties(${TARGET_NAME} PROPERTIES DEFINE_SYMBOL NeuralNetwork_EXPORTS)
add_executable(${TARGET_NAME}-app ${NeuralNetwork_SOURCE_DIR}/main.cpp)
target_link_libraries(${TARGET_NAME}-app ${TARGET_NAME})
ADD_DEFINITIONS("-Wall" "-fPIC")
