#pragma once

#include "fsm_basic_controller.hpp"

#include <onnxruntime_cxx_api.h>    // ONNX推理库
#include <eigen3/Eigen/Dense>       // 线性代数计算

namespace fs = std::filesystem;

namespace unitree::common
{

    class FsmLocoController : public BasicUserController
    {
    public:
        FsmLocoController()
            : env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime"), // 初始化列表
            session(env, ONNX_PATH.c_str(), session_options),
            last_action(Eigen::VectorXf::Zero(12)) // 使用初始化列表
        {
            // 初始化需要顺序控制的变量
            input_shape = {1, 73};
            
            // 初始化输入/输出名称
            input_names.push_back(session.GetInputNameAllocated(0, allocator).release());
            output_names.push_back(session.GetOutputNameAllocated(0, allocator).release());
            
            // 其他初始化
            _phase = {0.0f, static_cast<float>(M_PI)};
            _last_command = {0.0f, 0.0f, 0.0f};
            infer_dt = 0.02f;
            obs_joint_ids = OBS_JOINT_IDS;
            _default_qpos = DEFAULT_QPOS;
            _foot_height = 0.05f;
            action_scale=0.5f;
            gait_freq = 1.2f;
            phase_dt = 2 * M_PI * infer_dt * gait_freq;
        }

        void LoadParam(fs::path &param_folder);

        // 读取遥控器控制信息
        void GetInput(RobotInterface &robot_interface, Gamepad &gamepad);

        void Reset();

        // 遥控器控制走路，计算关节的目标角度
        void Calculate();

        // 欧拉角 转为 旋转向量
        Eigen::Vector3f rpy2rotvec(const std::array<float, 3>& rpy);

        void updatePhase(const Eigen::Vector3f& command);

        std::vector<float> GetLog();

        // cfg
        ExampleCfg cfg;

        // state
        std::array<float, 3> rpy, gyro;
        std::array<float, 3> cmd;
        std::array<float, 29> jpos_processed;
        std::array<float, 29> jvel_processed;
        std::array<float, 29> jpos, jvel;


        private:
            // 声明成员变量（不初始化）
            const std::string ONNX_PATH = "../../storage/policy/G1-Walk.onnx";
            Ort::Env env;
            Ort::SessionOptions session_options;
            Ort::Session session;
            Ort::AllocatorWithDefaultOptions allocator;
            std::vector<int64_t> input_shape;
            std::vector<const char*> input_names;
            std::vector<const char*> output_names;
            std::vector<float> _phase;
            std::vector<float> _last_command;
            float infer_dt;
            const std::vector<int> OBS_JOINT_IDS = {
                0, 1, 2, 3, 4, 5,
                6, 7, 8, 9, 10, 11,
                12, 13, 14,
                15, 16, 17, 18,
                22, 23, 24, 25
            };
            const std::vector<float> DEFAULT_QPOS = {
                -0.1, 0, 0, 0.3, -0.2, 0,
                -0.1, 0, 0, 0.3, -0.2, 0,
                0, 0, 0,
                0.2, 0.3, 0, 1.28, 0, 0, 0,
                0.2, -0.3, 0, 1.28, 0, 0, 0
            };
            std::vector<int> obs_joint_ids;
            std::vector<float> _default_qpos;
            std::vector<int> actuators_ids_active = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
            float _foot_height;
            float action_scale;
            float gait_freq;
            float phase_dt ;
            std::array<float, 2> _stance_phase = {0.0f, 0.0f};
            int _last_move_flag = 0;
            std::array<float, 2> _init_phase = {0.0f, M_PI};
            Eigen::VectorXf last_action;

    };
} // namespace unitree::common
