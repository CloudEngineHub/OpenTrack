#pragma once

#include <array>
#include <vector>
#include <filesystem>

#include <fstream>
#include <string>

#include "robot_interface.hpp"
#include "gamepad.hpp"
#include "cfg.hpp"

namespace fs = std::filesystem;

namespace unitree::common
{
    class BasicUserController
    {
    public:
        BasicUserController() {}

        virtual void LoadParam(fs::path &param_folder) = 0;

        virtual void Reset() = 0;

        virtual void GetInput(RobotInterface &robot_interface, Gamepad &gamepad) = 0;

        virtual void Calculate() = 0;

        virtual std::vector<float> GetLog() = 0;

        float infer_dt = 0.02f;
        float total_time = 2.0f; // 总运动时间2秒

        bool dance_done_flag = false;

        float dt;
        std::array<float, 29> kp;
        std::array<float, 29> kd;
        std::array<float, 29> init_pos;
        std::array<float, 29> jpos_des;
    };
}
