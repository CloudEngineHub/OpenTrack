#include <glog/logging.h>
#include "fsm_loco_controller.hpp"

namespace fs = std::filesystem;

namespace unitree::common
{

    void FsmLocoController::LoadParam(fs::path &param_folder)
    {
    }

    // 读取遥控器控制信息
    void FsmLocoController::GetInput(RobotInterface &robot_interface, Gamepad &gamepad)
    {
        const std::shared_ptr<const MotorState> ms = robot_interface.motor_state_buffer_.GetData();
        const std::shared_ptr<const ImuState> imus = robot_interface.imu_state_buffer_.GetData();
        // save necessary data from input
        rpy = imus->rpy;
        gyro = imus->gyro;
        jpos = ms->jpos;
        jvel = ms->jvel;
        // record command
        cmd.at(0) = gamepad.ly;
        cmd.at(1) = -gamepad.lx;
        cmd.at(2) = -gamepad.rx;

        // record robot state
        for (int i = 0; i < 29; ++i)
        {
            jpos_processed.at(i) = ms->jpos.at(i) - init_pos.at(i);
            jvel_processed.at(i) = ms->jvel.at(i) / 2.0;
        }
    }

    void FsmLocoController::Reset()
    {
        std::string path = "../../state_machine/params/";
        fs::path param_folder = fs::current_path() / path;
        // load param file
        std::ifstream cfg_file(param_folder / "loco.json");
        VLOG(1) << "[loco] reading params from " << (param_folder / "loco.json");
        std::stringstream ss;
        ss << cfg_file.rdbuf();
        FromJsonString(ss.str(), cfg);

        // get data from json
        dt = cfg.dt;
        for (int i = 0; i < 29; ++i)
        {
            init_pos.at(i) = cfg.init_pos.at(i);
            kp.at(i) = cfg.kp[i];
            kd.at(i) = cfg.kd[i];
        }
    }

    // 遥控器控制走路，计算关节的目标角度
    void FsmLocoController::Calculate()
    {
        // 构建步态相位特征
        std::vector<float> gait_phase;
        // 先计算所有cos值
        for (float p : _phase)
        {
            gait_phase.push_back(std::cos(p));
        }
        // 再计算所有sin值
        for (float p : _phase)
        {
            gait_phase.push_back(std::sin(p));
        }

        // 3. 计算移动标志
        std::vector<float> obs_command = {
            cmd.at(0) * 0.5f,
            cmd.at(1) * (-0.5f),
            cmd.at(2) * (-0.5f)};
        // 3. 计算变化量并限幅
        for (size_t i = 0; i < obs_command.size(); i++)
        {
            // 计算变化量
            float delta = obs_command[i] - _last_command[i];
            // 限幅 (±infer_dt)
            float clipped_delta = std::clamp(delta, -infer_dt, infer_dt);
            // 应用限幅后的变化
            obs_command[i] = _last_command[i] + clipped_delta;
        }
        _last_command = obs_command;

        Eigen::Vector3f cmd_vel(obs_command[0], obs_command[1], obs_command[2]);
        bool cmd_move_flag = cmd_vel.norm() > 0.2f;
        // 4. 构建指令特征
        std::vector<float> obs_cmd = {
            cmd_move_flag ? 1.0f : 0.0f,
            obs_command[0], -obs_command[1], -obs_command[2]};

        // 5. 构建传感器状态向量 (64维)
        Eigen::VectorXf state_sensor(64);
        int index = 0;
        // 计算当前姿态的重力向量
        Eigen::Vector3f pelvis_gvec = rpy2rotvec(rpy);
        for (int i = 0; i < 3; ++i)
            state_sensor[index++] = gyro[i];

        // gvec_pelvis (3)
        for (int i = 0; i < 3; ++i)
            state_sensor[index++] = pelvis_gvec[i];

        // joint position differences (23)
        for (int id : obs_joint_ids)
        {
            state_sensor[index++] = jpos[id] - init_pos[id];
        }

        // joint velocities (23)
        for (int id : obs_joint_ids)
        {
            state_sensor[index++] = jvel[id];
        }

        // last action (12)
        for (int j = 0; j < last_action.size(); ++j)
        {
            state_sensor[index++] = last_action[j];
        }

        // 6. 构建指令状态向量 (9维)
        Eigen::VectorXf state_command(9);
        index = 0;

        for (float val : obs_cmd)
            state_command[index++] = val;

        // foot_height (1)
        state_command[index++] = _foot_height;

        // gait_phase (4)
        for (float val : gait_phase)
            state_command[index++] = val;

        Eigen::VectorXf state(state_sensor.size() + state_command.size());
        state << state_sensor, state_command;
        Eigen::VectorXf nn_obs(state_sensor.size() + state_command.size());
        nn_obs = state.transpose();

        std::vector<Ort::Value> input_tensors;
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            nn_obs.data(),
            nn_obs.size(),
            input_shape.data(),
            input_shape.size()));

        // 执行推理
        // auto start = std::chrono::high_resolution_clock::now();
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            input_tensors.data(),
            input_names.size(),
            output_names.data(),
            output_names.size());

        // GetTensorMutableData：获取执行张量数据的原始指针，返回指向张量数据缓冲区起始位置的指针
        float *nn_action_data = output_tensors[0].GetTensorMutableData<float>();
        std::vector<float> nn_action(nn_action_data,
                                     nn_action_data + output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount());

        // 计算目标关节位置
        std::vector<float> motor_targets = _default_qpos;
        for (size_t i = 0; i < actuators_ids_active.size(); i++)
        {
            int idx = actuators_ids_active[i];
            motor_targets[idx] = _default_qpos[idx] + nn_action[i] * action_scale;
        }

        int action_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        last_action = Eigen::Map<Eigen::VectorXf>(nn_action_data, action_size);

        // 更新phase
        updatePhase(cmd_vel);

        for (int i = 0; i < 29; i++)
        {
            jpos_des.at(i) = motor_targets[i];
        }
    }

    // 欧拉角 转为 旋转向量
    Eigen::Vector3f FsmLocoController::rpy2rotvec(const std::array<float, 3> &rpy)
    {
        float roll = rpy[0], pitch = rpy[1], yaw = rpy[2];

        // 计算半角三角函数
        float cy = std::cos(yaw * 0.5f);
        float sy = std::sin(yaw * 0.5f);
        float cp = std::cos(pitch * 0.5f);
        float sp = std::sin(pitch * 0.5f);
        float cr = std::cos(roll * 0.5f);
        float sr = std::sin(roll * 0.5f);

        // 通过旋转顺序ZYX计算四元数
        float w = cr * cp * cy + sr * sp * sy;
        float x = sr * cp * cy - cr * sp * sy;
        float y = cr * sp * cy + sr * cp * sy;
        float z = cr * cp * sy - sr * sp * cy;

        // 调用四元数转重力向量
        float gx = -2 * (x * z - y * w);
        float gy = -2 * (y * z + x * w);
        float gz = -1 + 2 * (x * x + y * y);

        return Eigen::Vector3f(gx, gy, gz);
    }

    void FsmLocoController::updatePhase(const Eigen::Vector3f &command)
    {
        // 计算移动标志 (简化实现)
        float move_flag = command.norm() > 0.2f;
        // 计算新相位
        std::vector<float> new_phase(_phase.size());
        for (size_t i = 0; i < _phase.size(); i++)
        {
            // 相位增量
            new_phase[i] = _phase[i] + phase_dt;

            // 相位规范化 [-π, π]
            new_phase[i] = std::fmod(new_phase[i] + M_PI, 2 * M_PI) - M_PI;

            // 根据移动标志调整相位
            if (move_flag != 1.0f)
            {
                new_phase[i] = _stance_phase[i];
            }

            // 从静止到移动的过渡
            if (_last_move_flag == 0.0f && move_flag == 1.0f)
            {
                new_phase[i] = _init_phase[i];
            }
        }

        // 更新相位状态
        _phase = new_phase;
        _last_move_flag = move_flag;
    }

    std::vector<float> FsmLocoController::GetLog()
    {
        // record input, output and other info into a vector
        std::vector<float> log;
        for (int i = 0; i < 3; ++i)
        {
            log.push_back(cmd.at(i));
        }
        for (int i = 0; i < 29; ++i)
        {
            log.push_back(jpos_processed.at(i));
        }
        for (int i = 0; i < 29; ++i)
        {
            log.push_back(jvel_processed.at(i));
        }
        for (int i = 0; i < 29; ++i)
        {
            log.push_back(jpos_des.at(i));
        }

        return log;
    }

} // namespace unitree::common
