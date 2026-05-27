#pragma once

// Officialized logging utility shared by main.cpp / robot_controller.hpp /
// FSM controllers. See G1_deploy/README_officialize.md for the design.
//
// Three channels:
//   - main.log      : glog stderr (mirrored by start_*.sh tee). Same content
//                     as stdout. Carries [STARTUP], [SELFCHECK], [WAIT_R2],
//                     [FSM], [MODE], [EXIT] and a few errors.
//   - selfcheck.log : key=value snapshot written once at startup.
//   - events.log    : one-line audit trail for FSM transitions, dance
//                     start/end, mode-voice, torque-clip events.
//   - dance_<i>_<k>.csv + .meta.json : opened only while in DANCE state,
//                     one row per 50Hz control cycle with the state we
//                     received and the action we sent.

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <glog/logging.h>

namespace unitree::common
{

namespace fs = std::filesystem;

class OfficialLogger
{
public:
    static OfficialLogger &Instance()
    {
        static OfficialLogger inst;
        return inst;
    }

    void Init(const fs::path &log_dir)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        log_dir_ = log_dir;
        selfcheck_ofs_.open(log_dir_ / "selfcheck.log", std::ios::out | std::ios::trunc);
        events_ofs_.open(log_dir_ / "events.log", std::ios::out | std::ios::app);
        current_section_.clear();
    }

    const fs::path &LogDir() const { return log_dir_; }

    // ------------------------- selfcheck.log -------------------------
    // Open a new "[section]" header in selfcheck.log.
    void SelfcheckSection(const std::string &name)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!selfcheck_ofs_.is_open()) return;
        if (current_section_ == name) return;
        if (!current_section_.empty()) selfcheck_ofs_ << "\n";
        selfcheck_ofs_ << "[" << name << "]\n";
        selfcheck_ofs_.flush();
        current_section_ = name;
    }

    // Write a single key=value line into the current selfcheck section and
    // mirror a compact [SELFCHECK] line into glog/main.log for grepping.
    void Selfcheck(const std::string &key, const std::string &value)
    {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (selfcheck_ofs_.is_open())
            {
                selfcheck_ofs_ << key << "=" << value << "\n";
                selfcheck_ofs_.flush();
            }
        }
        LOG(INFO) << "[SELFCHECK] " << current_section_ << "." << key << "=" << value;
    }

    void SelfcheckSummary(bool pass, const std::string &reason = "")
    {
        if (pass)
        {
            LOG(INFO) << "[SELFCHECK] PASS";
        }
        else
        {
            LOG(WARNING) << "[SELFCHECK] FAIL reason=" << reason;
        }
    }

    // ------------------------- events.log + main.log -------------------------
    void Event(const std::string &event_type, const std::string &event_text)
    {
        const uint64_t ts_ms = NowMs();
        const std::string line = std::to_string(ts_ms) + " [" + event_type + "] " + event_text;
        LOG(INFO) << line;
        std::lock_guard<std::mutex> lk(mtx_);
        if (events_ofs_.is_open())
        {
            events_ofs_ << line << "\n";
            events_ofs_.flush();
        }
    }

    // ------------------------- dance_*.csv -------------------------
    struct DanceMeta
    {
        int mode_index = -1;
        int slot = -1;
        std::string policy_name;
        std::string checkpoint_dir;
        std::string checkpoint_onnx;
        std::string motion_name;
        std::string projection_compile = "OFF";
        std::string projection_runtime = "OFF";
        int action_dim = 29;
        int control_hz = 50;
    };

    void DanceOpen(const DanceMeta &meta)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        DanceCloseLocked("auto_close_before_reopen");

        dance_open_wallclock_ = std::chrono::system_clock::now();
        dance_open_steady_ = std::chrono::steady_clock::now();
        dance_cycle_ = 0;
        dance_meta_ = meta;

        std::ostringstream base;
        base << "dance_" << meta.mode_index << "_" << meta.slot;
        // Disambiguate when the same (mode, slot) is re-entered in one session.
        std::string stem = base.str();
        int suffix = 0;
        while (fs::exists(log_dir_ / (stem + ".csv")))
        {
            ++suffix;
            stem = base.str() + "_" + std::to_string(suffix);
        }
        dance_csv_path_ = log_dir_ / (stem + ".csv");
        dance_meta_path_ = log_dir_ / (stem + ".meta.json");

        dance_csv_ofs_.open(dance_csv_path_, std::ios::out | std::ios::trunc);
        if (dance_csv_ofs_.is_open())
        {
            WriteDanceHeader();
        }

        std::ofstream meta_ofs(dance_meta_path_, std::ios::out | std::ios::trunc);
        if (meta_ofs.is_open())
        {
            meta_ofs << "{\n";
            meta_ofs << "  \"timestamp_start\": \"" << TimestampString(dance_open_wallclock_) << "\",\n";
            meta_ofs << "  \"mode_index\": " << meta.mode_index << ",\n";
            meta_ofs << "  \"slot\": " << meta.slot << ",\n";
            meta_ofs << "  \"policy_name\": \"" << meta.policy_name << "\",\n";
            meta_ofs << "  \"checkpoint_dir\": \"" << meta.checkpoint_dir << "\",\n";
            meta_ofs << "  \"checkpoint_onnx\": \"" << meta.checkpoint_onnx << "\",\n";
            meta_ofs << "  \"motion_name\": \"" << meta.motion_name << "\",\n";
            meta_ofs << "  \"projection_compile\": \"" << meta.projection_compile << "\",\n";
            meta_ofs << "  \"projection_runtime\": \"" << meta.projection_runtime << "\",\n";
            meta_ofs << "  \"control_hz\": " << meta.control_hz << ",\n";
            meta_ofs << "  \"action_dim\": " << meta.action_dim << ",\n";
            meta_ofs << "  \"csv\": \"" << dance_csv_path_.filename().string() << "\"\n";
            meta_ofs << "}\n";
        }

        dance_open_ = true;
    }

    void DanceClose(const std::string &reason)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        DanceCloseLocked(reason);
    }

    bool DanceIsOpen() const
    {
        return dance_open_;
    }

    // Write one row. q/dq are the received joint state (29), quat is wxyz (4),
    // omega is body angular velocity (3), remote23 are the two button bytes,
    // action is the raw policy output (29) before projection, jpos_des/kp/kd
    // are what we will actually publish, tau_clip_count is per-cycle clip count.
    void DanceRow(const std::array<float, 29> &q,
                  const std::array<float, 29> &dq,
                  const std::array<float, 4> &quat,
                  const std::array<float, 3> &omega,
                  uint8_t remote2,
                  uint8_t remote3,
                  const std::array<float, 29> &action_pre_projection,
                  const std::array<float, 29> &jpos_des,
                  const std::array<float, 29> &kp,
                  const std::array<float, 29> &kd,
                  int tau_clip_count)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!dance_open_ || !dance_csv_ofs_.is_open()) return;
        const auto now = std::chrono::steady_clock::now();
        const auto t_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - dance_open_steady_).count();

        std::ostringstream os;
        os << dance_cycle_ << "," << t_ms;
        AppendCsv(os, q);
        AppendCsv(os, dq);
        os << "," << quat[0] << "," << quat[1] << "," << quat[2] << "," << quat[3];
        os << "," << omega[0] << "," << omega[1] << "," << omega[2];
        os << "," << static_cast<int>(remote2) << "," << static_cast<int>(remote3);
        AppendCsv(os, action_pre_projection);
        AppendCsv(os, jpos_des);
        AppendCsv(os, kp);
        AppendCsv(os, kd);
        os << "," << tau_clip_count;
        dance_csv_ofs_ << os.str() << "\n";
        ++dance_cycle_;
    }

private:
    OfficialLogger() = default;

    static uint64_t NowMs()
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
    }

    static std::string TimestampString(std::chrono::system_clock::time_point tp)
    {
        const auto t = std::chrono::system_clock::to_time_t(tp);
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            tp.time_since_epoch())
                            .count() %
                        1000;
        std::ostringstream os;
        os << std::put_time(std::localtime(&t), "%FT%T");
        os << "." << std::setw(3) << std::setfill('0') << ms;
        return os.str();
    }

    void WriteDanceHeader()
    {
        std::ostringstream os;
        os << "cycle,t_ms";
        for (int i = 0; i < 29; ++i) os << ",jpos_obs_" << i;
        for (int i = 0; i < 29; ++i) os << ",jvel_obs_" << i;
        os << ",imu_quat_w,imu_quat_x,imu_quat_y,imu_quat_z";
        os << ",imu_omega_x,imu_omega_y,imu_omega_z";
        os << ",remote_2,remote_3";
        for (int i = 0; i < 29; ++i) os << ",action_" << i;
        for (int i = 0; i < 29; ++i) os << ",jpos_des_" << i;
        for (int i = 0; i < 29; ++i) os << ",kp_" << i;
        for (int i = 0; i < 29; ++i) os << ",kd_" << i;
        os << ",tau_clip_count";
        dance_csv_ofs_ << os.str() << "\n";
    }

    template <std::size_t N>
    static void AppendCsv(std::ostringstream &os, const std::array<float, N> &arr)
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            os << "," << arr[i];
        }
    }

    void DanceCloseLocked(const std::string &reason)
    {
        if (!dance_open_) return;
        const uint64_t cycles = dance_cycle_;
        if (dance_csv_ofs_.is_open())
        {
            dance_csv_ofs_.flush();
            dance_csv_ofs_.close();
        }
        dance_open_ = false;
        // Emit DANCE_END event (mirror to main.log + events.log). We unlock
        // briefly so the Event() call can re-acquire the mutex.
        const int mi = dance_meta_.mode_index;
        const int sl = dance_meta_.slot;
        mtx_.unlock();
        Event("DANCE_END",
              "mode_index=" + std::to_string(mi) +
                  ", slot=" + std::to_string(sl) +
                  ", cycles=" + std::to_string(cycles) +
                  ", reason=" + reason);
        mtx_.lock();
    }

    std::mutex mtx_;
    fs::path log_dir_;
    std::ofstream selfcheck_ofs_;
    std::ofstream events_ofs_;
    std::string current_section_;

    bool dance_open_ = false;
    std::ofstream dance_csv_ofs_;
    fs::path dance_csv_path_;
    fs::path dance_meta_path_;
    DanceMeta dance_meta_;
    std::chrono::steady_clock::time_point dance_open_steady_{};
    std::chrono::system_clock::time_point dance_open_wallclock_{};
    uint64_t dance_cycle_ = 0;
};

}  // namespace unitree::common
