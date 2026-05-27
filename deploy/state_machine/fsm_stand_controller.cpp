
#include "fsm_stand_controller.hpp"

namespace fs = std::filesystem;

namespace unitree::common
{

    void FsmStandController::LoadParam(fs::path &param_folder)
    {
        // do nothing
    }

    void FsmStandController::GetInput(RobotInterface &robot_interface, Gamepad &gamepad)
    {
        const std::shared_ptr<const MotorState> ms = robot_interface.motor_state_buffer_.GetData();
        jpos = ms->jpos;
    }

    void FsmStandController::Reset()
    {
        std::string path = "../../state_machine/params/";
        fs::path param_folder = fs::current_path() / path;
        std::ifstream cfg_file(param_folder / "stand.json");
        std::stringstream ss;
        ss << cfg_file.rdbuf();

        ExampleCfg cfg;
        FromJsonString(ss.str(), cfg);

        dt = cfg.dt;
        kp = cfg.kp;
        kd = cfg.kd;
        for (int i = 0; i < 29; ++i)
        {
            init_pos.at(i) = cfg.init_pos.at(i);
        }
        step_ = 0;
    }

    void FsmStandController::Calculate()
    {
        Eigen::VectorXf cur_pos(29);
        for (int i = 0; i < 29; i++)
        {
            cur_pos[i] = jpos[i];
        }
        // 从当前姿态，插值到默认姿态
        std::array<float, 29> target_pos;
        target_pos = init_pos;
        // 把target_pos从数组转为向量
        Eigen::Map<Eigen::VectorXf> target_pos_vec(target_pos.data(), 29);

        ///////////////////////////////////////////插值处理
        const float infer_dt = 0.02f;
        const float total_time = 2.0f; // 总运动时间2秒
        const int num_steps = static_cast<int>(total_time / infer_dt);

        VLOG(2) << "stand step:" << step_;
        if (step_ > num_steps)
        {
            return;
        }

        const float alpha = static_cast<float>(step_) / num_steps;

        // 线性插值
        Eigen::VectorXf tar_qpos = cur_pos * (1.0f - alpha) + target_pos_vec * alpha;

        // 计算当前帧的目标角度，写入jpos_des
        for (int i = 0; i < 29; i++)
        {
            jpos_des.at(i) = tar_qpos[i];
        }

        step_++;
    }

    std::vector<float> FsmStandController::GetLog()
    {
        std::vector<float> log;
        return log;
    }

} // namespace unitree::common
