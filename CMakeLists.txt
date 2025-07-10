cmake_minimum_required(VERSION 3.25)
project(varitensor_tests)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -pedantic")

add_executable(run_tests test/run_tests.cxx)
target_include_directories(run_tests
  PRIVATE
    test
    include
)
