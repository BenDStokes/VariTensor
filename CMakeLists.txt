cmake_minimum_required(VERSION 3.25)
project(VariTensor
  VERSION 1.1.0
  LANGUAGES CXX
  DESCRIPTION "Ergonomic tensors with variance."
)

add_library(varitensor
  src/Tensor.cxx
  src/pretty_print.cxx
  src/impl/ExpressionIteratorBase.cxx
  src/impl/Index.cxx
  src/impl/LinkedOp.cxx
  src/impl/pre_defined.cxx
  src/impl/ProductOp.cxx
  src/impl/View.cxx
  src/impl/bulk.cxx
  src/impl/Preparatory.cxx
)

add_library(varitensor::varitensor ALIAS varitensor)

target_include_directories(varitensor
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(varitensor PUBLIC cxx_std_23)

# ------------------------------------------------------------------------------------ INSTALLATION

include(GNUInstallDirs)

install(TARGETS varitensor
  EXPORT varitensorTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT varitensorTargets
  FILE varitensorTargets.cmake
  NAMESPACE varitensor::
  DESTINATION lib/cmake/varitensor
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/varitensorConfig.cmake"
  INSTALL_DESTINATION lib/cmake/varitensor
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/varitensorConfig.cmake"
  DESTINATION lib/cmake/varitensor
)

# ------------------------------------------------------------------------------------------- TESTS

add_executable(varitensor_tests
  test/run_tests.cxx
  src/Tensor.cxx
  src/pretty_print.cxx
  src/impl/ExpressionIteratorBase.cxx
  src/impl/Index.cxx
  src/impl/LinkedOp.cxx
  src/impl/pre_defined.cxx
  src/impl/ProductOp.cxx
  src/impl/View.cxx
  src/impl/bulk.cxx
  src/impl/Preparatory.cxx
)

target_include_directories(varitensor_tests
  PRIVATE
    test
    include
)

target_compile_features(varitensor_tests PUBLIC cxx_std_23)
