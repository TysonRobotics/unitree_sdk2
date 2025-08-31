add_executable(g1_audio_play_wav g1_audio_play_wav.cpp)
target_link_libraries(g1_audio_play_wav unitree_sdk2)
# For wav.hpp
target_include_directories(g1_audio_play_wav PRIVATE ${CMAKE_SOURCE_DIR}/example/g1/audio)



