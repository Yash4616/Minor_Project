cmake_minimum_required(VERSION 3.18)

project(Change LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CHANGE_FORCE_GLIBCXX_ABI0 "Force -D_GLIBCXX_USE_CXX11_ABI=0 for LibTorch compatibility" OFF)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(Change
	src/main.cpp
	src/data_gen.cpp
	src/train_detector.cpp
	src/train_unet.cpp
	src/evaluate.cpp
	src/visualize.cpp
	src/report.cpp
)

target_include_directories(Change
	PRIVATE
		${PROJECT_SOURCE_DIR}/include
		${OpenCV_INCLUDE_DIRS}
)

if(MSVC)
	target_compile_options(Change PRIVATE /W4 /permissive-)
else()
	target_compile_options(Change PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(NOT MSVC AND CHANGE_FORCE_GLIBCXX_ABI0)
	target_compile_definitions(Change PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
endif()

target_link_libraries(Change PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})

if(WIN32)
	file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
	add_custom_command(TARGET Change POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${TORCH_DLLS}
			$<TARGET_FILE_DIR:Change>
	)
endif()

