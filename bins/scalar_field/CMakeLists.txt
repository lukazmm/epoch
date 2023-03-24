#########################
## Scalar Field Target ##
#########################

set(SCALAR_FIELD_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(epoch_scalar_field)
deal_ii_setup_target(epoch_scalar_field)

# Private
target_include_directories(epoch_scalar_field PRIVATE ${SCALAR_FIELD_SRC_DIR})
target_sources(epoch_scalar_field PRIVATE 
    ${SCALAR_FIELD_SRC_DIR}/domain.hpp
    ${SCALAR_FIELD_SRC_DIR}/field.hpp
    ${SCALAR_FIELD_SRC_DIR}/main.cpp
)

add_custom_target(run_scalar_field COMMAND epoch_scalar_field
  COMMENT "Run with ${CMAKE_BUILD_TYPE} configuration"
)