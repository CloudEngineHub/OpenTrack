#pragma once

#include "fsm_basic_controller.hpp"

#include <eigen3/Eigen/Dense> // 线性代数计算

namespace fs = std::filesystem;

namespace unitree::common
{
    class FsmStandController : public BasicUserController
    {
    public:
        FsmStandController() {}

        virtual void LoadParam(fs::path &param_folder) override;

        virtual void GetInput(RobotInterface &robot_interface, Gamepad &gamepad) override;

        virtual void Reset() override;

        virtual void Calculate() override;

        virtual std::vector<float> GetLog() override;

    private:
        int step_ = 0;
        std::array<float, 29> jpos;
    };
} // namespace unitree::common
