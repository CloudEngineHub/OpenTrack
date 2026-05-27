# add_executable(state_machine_example main.cpp)
aux_source_directory(. SRC_LIST)
add_executable(state_machine_example ${SRC_LIST})
target_link_libraries(state_machine_example unitree_sdk2)

if(ENABLE_DANCE_TORQUE_PROJECTION)
	target_compile_definitions(state_machine_example PRIVATE ENABLE_DANCE_TORQUE_PROJECTION=1)
else()
	target_compile_definitions(state_machine_example PRIVATE ENABLE_DANCE_TORQUE_PROJECTION=0)
endif()

# Allow running from build/bin without manually exporting LD_LIBRARY_PATH.
# build/bin -> ../../thirdparty/lib/<arch>
set_target_properties(state_machine_example PROPERTIES
	BUILD_RPATH "$ORIGIN/../../thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}"
	INSTALL_RPATH "$ORIGIN/../../thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}"
)