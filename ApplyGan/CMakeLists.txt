#cmake_minimum_required(VERSION 2.8)

#include(CheckCXXCompilerFlag)
#CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
#CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
#
#if(COMPILER_SUPPORTS_CXX11)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has  c++11 support")
##set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gun++11")
#elseif(COMPILER_SUPPORTS_CXX0X)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
#message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has c++0x support")
#else()
#message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
#endif()

add_definitions(-DGLIBCXX_USE_CXX11_ABI=0)

set(main_src ./src/apply/main.cpp ./src/apply/SLCIORdr.cpp ./src/utils/my_utils.cpp)
set(main_inc ./src/apply/ ./src/utils/)

set(main_src1 ./src/check/main.cpp ./src/check/SLCIORdr.cpp ./src/utils/my_utils.cpp)
set(main_inc1 ./src/check/ ./src/utils/)

set(tf_inc /junofs/users/lint/tensorflow-c-api/dev-from-scratch/v1.14.0/include)
set(tf_lib /junofs/users/lint/tensorflow-c-api/dev-from-scratch/v1.14.0/lib)
#set(lcio_lib /junofs/users/wxfang/CEPC/LCIO-02-12-01/build/lib)
#set(lcio_inc /junofs/users/wxfang/CEPC/LCIO-02-12-01/src/cpp/include /junofs/users/wxfang/CEPC/LCIO-02-12-01/src/cpp/include/pre-generated/)
set(lcio_lib /junofs/users/wxfang/CEPC/LCIO/build/lib)
set(lcio_inc /junofs/users/wxfang/CEPC/LCIO/src/cpp/include /junofs/users/wxfang/CEPC/LCIO/src/cpp/include/pre-generated/)

#SET(CMAKE_MODULE_PATH /cvmfs/cepc.ihep.ac.cn/software/cepcsoft/x86_64-sl6-gcc49/external/ROOT/5.34.18/install/etc/cmake)

find_package(LCIO)
#find_package(ROOT)

#if(ROOT_FOUND)
#    message("found ROOT: ${ROOT_INCLUDE_DIR} ${ROOT_LIBRARIES}")
#endif(ROOT_FOUND)

if(LCIO_FOUND)
    message("found LCIO: ${LCIO_INCLUDE_DIRS} ${LCIO_LIBRARIES}")
endif(LCIO_FOUND)

INCLUDE_DIRECTORIES(${main_inc})
INCLUDE_DIRECTORIES(${main_inc1})
INCLUDE_DIRECTORIES(${tf_inc})
INCLUDE_DIRECTORIES(${lcio_inc})
#INCLUDE_DIRECTORIES(${LCIO_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${ROOT_INCLUDE_DIR})
#LINK_DIRECTORIES(${tf_lib})


ADD_EXECUTABLE(Gan ${main_src})
#TARGET_LINK_LIBRARIES(Gan ${LCIO_LIBRARIES} ${ROOT_LIBRARIES} ${tf_lib}/libtensorflow_framework.so ${tf_lib}/libtensorflow.so )
TARGET_LINK_LIBRARIES(Gan ${lcio_lib}/liblcio.so ${lcio_lib}/libsio.so ${ROOT_LIBRARIES} ${tf_lib}/libtensorflow_framework.so ${tf_lib}/libtensorflow.so )


ADD_EXECUTABLE(Check ${main_src1})
TARGET_LINK_LIBRARIES(Check ${lcio_lib}/liblcio.so ${lcio_lib}/libsio.so ${ROOT_LIBRARIES} ${tf_lib}/libtensorflow_framework.so ${tf_lib}/libtensorflow.so )

