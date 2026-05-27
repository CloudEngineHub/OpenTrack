import os
import sys
import time
import threading
from copy import deepcopy
from pathlib import Path

current_dir = Path(__file__).resolve().parent


def _sim_verbose():
    return os.getenv("SIM_VERBOSE", "").strip().lower() in ("1", "true", "on", "yes")


_SIM_VERBOSE = _sim_verbose()

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
    unitree_hg_msg_dds__MotorState_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_, IMUState_, MotorState_, MotorCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

try:
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_server import MotionSwitcherServer
except ModuleNotFoundError:
    class MotionSwitcherServer:  # type: ignore[no-redef]
        def Init(self):
            print("[WARN] MotionSwitcherServer not available, running without mode switcher service")

        def Start(self):
            print("[WARN] MotionSwitcherServer disabled")

from viewer_dds import SimState_
from wireless_controller import UnitreeRemoteController

DEFAULT_QPOS = np.float32(
    [
        0,
        0,
        0.8,
        1,
        0,
        0,
        0,
        -0.1,
        0,
        0,
        0.3,
        -0.2,
        0,
        -0.1,
        0,
        0,
        0.3,
        -0.2,
        0,
        0,
        0,
        0,
        0.2,
        0.3,
        0,
        1.28,
        0,
        0,
        0,
        0.2,
        -0.3,
        0,
        1.28,
        0,
        0,
        0,
    ]
)

KPs = np.float32(
    [
        100,
        100,
        100,
        200,
        80,
        20,
        100,
        100,
        100,
        200,
        80,
        20,
        300,
        300,
        300,
        90,
        60,
        20,
        60,
        20,
        20,
        20,
        90,
        60,
        20,
        60,
        20,
        20,
        20,
    ]
)

KDs = np.float32(
    [
        2,
        2,
        2,
        4,
        2,
        1,
        2,
        2,
        2,
        4,
        2,
        1,
        10,
        10,
        10,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
    ]
)


class MockG1Robot:
    def __init__(self, sim_dt=0.002, viewer_dt=0.02):
        self.sim_dt = sim_dt
        self.viewer_dt = viewer_dt
        self.lowcmd = unitree_hg_msg_dds__LowCmd_()
        self.start_time = time.time()

        # ---- runtime diagnostics / controls (env-driven) ----
        # IMPORTANT: by user request the physics is ONLY enabled by a manual
        # SimStart key press. We do NOT auto-enable when lowcmd starts to
        # flow (otherwise pressing R2 to release the controller's safety
        # would immediately drop the robot before it has a chance to settle).
        self.auto_start_on_lowcmd = os.getenv("SIM_AUTO_START_ON_LOWCMD", "0") == "1"
        try:
            self.lowcmd_timeout_sec = float(os.getenv("SIM_LOWCMD_TIMEOUT_SEC", "0.35"))
        except ValueError:
            self.lowcmd_timeout_sec = 0.35
        try:
            self.diag_period = int(os.getenv("SIM_DIAG_PERIOD", "500"))
        except ValueError:
            self.diag_period = 500
        self.lowcmd_rx_count = 0
        self.lowstate_tx_count = 0
        self.sim_step_count = 0
        self.hold_step_count = 0
        self._last_lowcmd_time = 0.0
        self._last_diag_log_time = time.monotonic()

        # Optional qpos trace for debugging. Writes one CSV row per
        # simulation step. Set SIM_QPOS_TRACE_PATH to enable.
        self._qpos_trace_path = os.getenv("SIM_QPOS_TRACE_PATH", "").strip()
        self._qpos_trace_fp = None
        if self._qpos_trace_path:
            try:
                self._qpos_trace_fp = open(self._qpos_trace_path, "w", buffering=1)
                self._qpos_trace_fp.write(
                    "t_ms,sim_enabled,qpos_x,qpos_y,qpos_z,qw,qx,qy,qz\n"
                )
                print(f"[SIM] qpos trace -> {self._qpos_trace_path}")
            except OSError as e:
                print(f"[SIM] WARN: cannot open qpos trace '{self._qpos_trace_path}': {e}")
                self._qpos_trace_fp = None
        # default standing pos
        for i in range(29):
            self.lowcmd.motor_cmd[i].mode = 1
            self.lowcmd.motor_cmd[i].kp = KPs[i]
            self.lowcmd.motor_cmd[i].kd = KDs[i]
            self.lowcmd.motor_cmd[i].q = DEFAULT_QPOS[i + 7]
            self.lowcmd.motor_cmd[i].dq = 0
            self.lowcmd.motor_cmd[i].tau = 0

        self.lowstate = unitree_hg_msg_dds__LowState_()
        self.crc = CRC()

        self.xml_path = str(current_dir / "mjcf/scene_mjx_flat_terrain.xml")
        self.mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.mj_model.opt.timestep = sim_dt
        self.mj_data = mujoco.MjData(self.mj_model)
        self.qpos = DEFAULT_QPOS.copy()
        self.qvel = np.zeros(35)
        # Start in hold mode: publish lowstate but keep robot pinned at default pose
        # until SimStart key is pressed.
        self.simulation_enabled = False

        self.lowcmd_lock = threading.Lock()
        self.simulation_lock = threading.Lock()
        self.simulation_thread = RecurrentThread(
            interval=sim_dt,
            target=self._simulation_step,
            name="simulation_step",
        )
        self.viewer_thread = RecurrentThread(
            interval=viewer_dt,
            target=self._viewer_step,
            name="viewer_step",
        )

        # publish state in a high frequency (e.g. 500Hz)
        self.lowstate_publisher = ChannelPublisher("rt/lowstate", LowState_)
        self.lowstate_publisher.Init()

        # publish sim state in a low frequency (e.g. 50Hz)
        self.simstate_publisher = ChannelPublisher("rt/simstate", SimState_)
        self.simstate_publisher.Init()

        # receive lowcmd in event triggered mode
        self.lowcmd_subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.lowcmd_subscriber.Init(self._receive_lowcmd, 10)

        # motion switcher
        self.mss = MotionSwitcherServer()
        self.mss.Init()

        # controller with keyboard
        self.remote_controller = UnitreeRemoteController()

    def start(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:] = DEFAULT_QPOS
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        self.simulation_thread.Start()
        print("Simulation thread started")
        self.viewer_thread.Start()
        print("Viewer thread started")
        self.mss.Start()
        print("Motion switcher server started")
        self.remote_controller.listen_keyboard()
        print("Keyboard listener started")
        print(
            "[SIM] Hold mode active (robot pinned at DEFAULT_QPOS). "
            "Press 'F5' (or KP_9 / backtick `) to enable physics stepping. "
            f"auto_start_on_lowcmd={self.auto_start_on_lowcmd}"
        )

    def _publish_default_hold_state(self):
        # Keep a deterministic standing pose while still publishing lowstate.
        self.mj_data.qpos[:] = DEFAULT_QPOS
        self.mj_data.qvel[:] = 0
        self.mj_data.ctrl[:] = 0
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._prepare_lowstate()
        self.lowstate_publisher.Write(self.lowstate)

    def _simulation_step(self):
        with self.lowcmd_lock:
            motor_cmd: MotorCmd_ = deepcopy(self.lowcmd.motor_cmd)
            last_lowcmd_time = self._last_lowcmd_time

        with self.simulation_lock:
            if not self.simulation_enabled:
                lowcmd_fresh = (
                    self.auto_start_on_lowcmd
                    and last_lowcmd_time > 0.0
                    and (time.monotonic() - last_lowcmd_time) <= self.lowcmd_timeout_sec
                )
                if self.remote_controller.is_button_pressed("SimStart") or lowcmd_fresh:
                    self.simulation_enabled = True
                    reason = "SimStart" if self.remote_controller.is_button_pressed("SimStart") else "lowcmd_fresh"
                    print(
                        f"[SIM] >>> physics ENABLED (reason={reason}, "
                        f"sim_steps_so_far={self.sim_step_count}, "
                        f"hold_steps_so_far={self.hold_step_count})"
                    )
                else:
                    # Hold mode: pin robot at default pose and keep publishing lowstate.
                    # This snap-back is the ONLY path that overwrites qpos to DEFAULT;
                    # once physics is enabled we never come back here.
                    self._publish_default_hold_state()
                    self.hold_step_count += 1
                    self._maybe_log_diag()
                    self._maybe_trace_qpos()
                    return

            # NOTE: we intentionally do NOT auto-revert to hold mode on stale
            # lowcmd. A previous version did, but that caused the robot's
            # qpos to be reset to DEFAULT_QPOS every time the lowcmd stream
            # hiccupped (>SIM_LOWCMD_TIMEOUT_SEC gap), which looked like a
            # "periodic mujoco reset" to the user. If you really need this
            # safety, set SIM_REVERT_ON_STALE_LOWCMD=1 explicitly.
            if os.getenv("SIM_REVERT_ON_STALE_LOWCMD", "0") == "1" and last_lowcmd_time > 0.0:
                if (time.monotonic() - last_lowcmd_time) > self.lowcmd_timeout_sec:
                    self.simulation_enabled = False
                    print("[SIM] lowcmd stale, reverting to hold mode (qpos will snap to DEFAULT)")
                    self._publish_default_hold_state()
                    self.hold_step_count += 1
                    self._maybe_log_diag()
                    self._maybe_trace_qpos()
                    return

            if self.remote_controller.is_button_pressed("Reset"):
                mujoco.mj_resetData(self.mj_model, self.mj_data)
                self.mj_data.qpos[:] = DEFAULT_QPOS
                mujoco.mj_forward(self.mj_model, self.mj_data)
                print("[SIM] manual Reset triggered (qpos reset to DEFAULT)")

            for i in range(29):
                if motor_cmd[i].mode == 0:
                    self.mj_data.ctrl[i] = 0
                else:
                    self.mj_data.ctrl[i] = (
                        motor_cmd[i].tau
                        + motor_cmd[i].kp * (motor_cmd[i].q - self.mj_data.qpos[i + 7])
                        + motor_cmd[i].kd * (motor_cmd[i].dq - self.mj_data.qvel[i + 6])
                    )
            mujoco.mj_step(self.mj_model, self.mj_data)
            self._prepare_lowstate()
            self.lowstate_publisher.Write(self.lowstate)
            self.lowstate_tx_count += 1
            self.sim_step_count += 1
            self._maybe_log_diag()
            self._maybe_trace_qpos()

    def _viewer_step(self):
        with self.simulation_lock:
            self.qpos = self.mj_data.qpos.copy()
            self.qvel = self.mj_data.qvel.copy()
            self.simstate_publisher.Write(SimState_(qpos=self.qpos, qvel=self.qvel, tick=int((time.time() - self.start_time) * 1000)))

    def _receive_lowcmd(self, msg: LowCmd_):
        with self.lowcmd_lock:
            self.lowcmd = msg
            self._last_lowcmd_time = time.monotonic()
            self.lowcmd_rx_count += 1
            cnt = self.lowcmd_rx_count
        if cnt == 1 or (_SIM_VERBOSE and self.diag_period > 0 and cnt % self.diag_period == 0):
            try:
                m0 = msg.motor_cmd[0]
                print(
                    f"[SIM_RX_LOWCMD] cnt={cnt}, m0(mode={m0.mode}, q={m0.q:.4f}, "
                    f"kp={m0.kp:.2f}, kd={m0.kd:.2f}, tau={m0.tau:.4f})"
                )
            except Exception:
                pass

    def _maybe_trace_qpos(self):
        fp = self._qpos_trace_fp
        if fp is None:
            return
        try:
            t_ms = int((time.time() - self.start_time) * 1000.0)
            q = self.mj_data.qpos
            fp.write(
                f"{t_ms},{int(self.simulation_enabled)},"
                f"{q[0]:.5f},{q[1]:.5f},{q[2]:.5f},"
                f"{q[3]:.5f},{q[4]:.5f},{q[5]:.5f},{q[6]:.5f}\n"
            )
        except Exception:
            pass

    def _maybe_log_diag(self):
        if not _SIM_VERBOSE:
            return
        if self.diag_period <= 0:
            return
        now = time.monotonic()
        if (now - self._last_diag_log_time) < 1.0:
            return
        self._last_diag_log_time = now
        age_ms = -1.0
        if self._last_lowcmd_time > 0.0:
            age_ms = (now - self._last_lowcmd_time) * 1000.0
        try:
            remote = self.lowstate.wireless_remote
            remote_str = f"({int(remote[2])},{int(remote[3])})"
        except Exception:
            remote_str = "NA"
        print(
            f"[SIM_DIAG] sim_enabled={self.simulation_enabled}, "
            f"sim_steps={self.sim_step_count}, hold_steps={self.hold_step_count}, "
            f"lowcmd_rx={self.lowcmd_rx_count}, lowstate_tx={self.lowstate_tx_count}, "
            f"lowcmd_age_ms={age_ms:.1f}, remote23={remote_str}"
        )

    def _prepare_imustate(self):
        rotation = R.from_quat(self.mj_data.qpos[[4, 5, 6, 3]].copy())
        imu_state = IMUState_(
            quaternion=self.mj_data.qpos[3:7].copy(),
            gyroscope=self.mj_data.qvel[3:6].copy(),
            accelerometer=self.mj_data.qacc[3:6].copy(),
            rpy=rotation.as_euler("xyz", degrees=False),
            temperature=0,  # not defined
        )
        return imu_state

    def _prepare_motorstate(self):
        motor_state = []
        for i in range(29):
            motor_state.append(
                MotorState_(
                    mode=1,
                    q=self.mj_data.qpos[i + 7].copy(),
                    dq=self.mj_data.qvel[i + 6].copy(),
                    ddq=self.mj_data.qacc[i + 6].copy(),
                    tau_est=self.mj_data.qfrc_constraint[i + 6].copy(),
                    temperature=[0, 0],  # not defined
                    vol=0.0,  # not defined
                    sensor=[0, 0],  # not defined
                    motorstate=0,  # not defined
                    reserve=[0, 0, 0, 0],  # not defined
                )
            )
        for i in range(6):
            motor_state.append(unitree_hg_msg_dds__MotorState_())
        return motor_state

    def _prepare_lowstate(self):
        # NOTE: version, mode_pr, mode_machine and reserve are not defined
        wireless_remote = self.remote_controller.encode_botton()
        self.lowstate = LowState_(
            version=[0, 0],
            mode_pr=0,
            mode_machine=0,
            tick=int((time.time() - self.start_time) * 1000),
            imu_state=self._prepare_imustate(),
            motor_state=self._prepare_motorstate(),
            wireless_remote=wireless_remote,
            reserve=[0, 0, 0, 0],
            crc=0,
        )
        
        self.lowstate.crc = self.crc.Crc(self.lowstate)


if __name__ == "__main__":
    iface = os.getenv("SIM_IFACE", "").strip()
    if len(sys.argv) > 1 and sys.argv[1]:
        ChannelFactoryInitialize(0, sys.argv[1])
        print(f"[SIM] DDS iface={sys.argv[1]} (argv)")
    elif iface:
        ChannelFactoryInitialize(0, iface)
        print(f"[SIM] DDS iface={iface} (env)")
    else:
        ChannelFactoryInitialize(0)
        print("[SIM] DDS iface=<default>")

    mock_g1_robot = MockG1Robot()
    mock_g1_robot.start()

    while True:
        time.sleep(0.1)
