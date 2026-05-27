#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <optional>
#include <chrono>
#include <iomanip>
#include <glog/logging.h>

#include "officialize_logger.hpp"
#include "robot_controller.hpp"
#include "robot_interface.hpp"

using namespace unitree::common;
using namespace unitree::robot;

namespace fs = std::filesystem;

int main(int argc, char const *argv[])
{
    std::string param_folder;
    std::string network_interface;
    std::string legacy_positional_iface;
    std::string positional_iface_notice;

    // parse command line params
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "--param" && i + 1 < argc)
        {
            param_folder = argv[i + 1];
            ++i;
            continue;
        }

        if ((arg == "--iface" || arg == "--network-interface") && i + 1 < argc)
        {
            network_interface = argv[i + 1];
            ++i;
            continue;
        }

        // Legacy positional interface argument is deprecated and ignored by default.
        if (!arg.empty() && arg[0] != '-' && legacy_positional_iface.empty())
        {
            legacy_positional_iface = arg;
        }
    }

    if (network_interface.empty() && !legacy_positional_iface.empty())
    {
        const char *allow_legacy_iface = std::getenv("G1_ALLOW_LEGACY_POSITIONAL_IFACE");
        if (allow_legacy_iface != nullptr && std::string(allow_legacy_iface) == "1")
        {
            network_interface = legacy_positional_iface;
            positional_iface_notice =
                "Using deprecated positional iface argument: " + legacy_positional_iface;
        }
        else
        {
            positional_iface_notice =
                "Ignoring deprecated positional iface argument: " + legacy_positional_iface +
                ". Use --iface <name> to force DDS interface binding.";
        }
    }

    fs::path param = fs::current_path() / param_folder;

    // Resolve log directory. Caller (start_*.sh) pins it via G1_LOG_DIR;
    // otherwise fall back to ./logs/<timestamp> under the current cwd.
    fs::path log_folder;
    std::string log_dir_source = "default";
    if (const char *env_log_dir = std::getenv("G1_LOG_DIR"))
    {
        if (*env_log_dir != '\0')
        {
            log_folder = fs::path(env_log_dir);
            log_dir_source = "env";
        }
    }
    if (log_folder.empty())
    {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y_%m_%d_%H_%M_%S");
        log_folder = fs::current_path() / "logs" / ss.str();
    }
    std::filesystem::create_directories(log_folder);

    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;
    FLAGS_log_dir = log_folder;
    FLAGS_stderrthreshold = google::INFO;

    OfficialLogger::Instance().Init(log_folder);

    if (!positional_iface_notice.empty())
    {
        LOG(WARNING) << positional_iface_notice;
    }

    std::ofstream cfg_file(log_folder / "cfg.txt");
    cfg_file << "param_folder: " << param << std::endl;
    cfg_file << "network_interface: "
             << (network_interface.empty() ? "<auto>" : network_interface)
             << std::endl;
    cfg_file << "legacy_positional_iface: "
             << (legacy_positional_iface.empty() ? "<none>" : legacy_positional_iface)
             << std::endl;
    cfg_file.close();

    // -------- [STARTUP] (channel A) + [build]/[dds] selfcheck --------
#if ENABLE_DANCE_TORQUE_PROJECTION
    const char *startup_projection_compile = "ON";
#else
    const char *startup_projection_compile = "OFF";
#endif
    std::string startup_projection_runtime = "OFF";
    if (const char *env = std::getenv("G1_ENABLE_DANCE_TORQUE_PROJECTION"))
    {
        std::string value(env);
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (value == "1" || value == "true" || value == "on" || value == "yes")
        {
            startup_projection_runtime = "ON";
        }
    }
    LOG(INFO) << "[STARTUP] projection_compile=" << startup_projection_compile
              << " projection_runtime=" << startup_projection_runtime
              << " iface=" << (network_interface.empty() ? "<auto>" : network_interface)
              << " log_dir=" << log_folder.string()
              << " log_dir_source=" << log_dir_source;

    auto &lg = OfficialLogger::Instance();
    lg.SelfcheckSection("build");
    lg.Selfcheck("projection_compile", startup_projection_compile);
    lg.Selfcheck("projection_runtime", startup_projection_runtime);
    lg.SelfcheckSection("dds");
    lg.Selfcheck("iface", network_interface.empty() ? "<auto>" : network_interface);
    lg.Selfcheck("iface_source", network_interface.empty() ? "default" : "argv");

    fs::path log_file_name = log_folder / "log.txt";

    RobotController robot_controller(log_file_name);
    robot_controller.LoadParam(param);

    robot_controller.InitDdsModel(network_interface);

    lg.SelfcheckSummary(true);

    robot_controller.StartControl();

    return 0;
}