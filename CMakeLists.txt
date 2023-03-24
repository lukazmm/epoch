cmake_minimum_required(VERSION 3.10.0)

#############################
## Basic Vars and Options ###
#############################

#############################
## Project ##################
#############################

project(epoch CXX C)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#############################
## Packages #################
#############################

## Dealii

set(DEALII_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/packages/build/dealii)

find_package(deal.II 9.4.0 CONFIG REQUIRED PATHS ${DEALII_BUILD_DIR})
deal_ii_initialize_cached_variables()

#########################
## Epoch Target #########
#########################

set(EPOCH_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(EPOCH_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(epoch STATIC)
deal_ii_setup_target(epoch)

# Public
target_include_directories(epoch PUBLIC ${EPOCH_INC_DIR})
target_sources(epoch PUBLIC 
    ${EPOCH_INC_DIR}/epoch/dummy.hpp
)

# Private
target_include_directories(epoch PRIVATE ${EPOCH_SRC_DIR})
target_sources(epoch PRIVATE 
    ${EPOCH_SRC_DIR}/dummy.cpp
)
#########################
## Bins #################
#########################

add_subdirectory(bins/scalar_field)

#####################
## Custom Targets ###
#####################

add_custom_target(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
)

add_custom_target(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
)

add_custom_target(build
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Build Project"
)

# add_custom_target(run COMMAND eon
#   COMMENT "Run with ${CMAKE_BUILD_TYPE} configuration"
# )