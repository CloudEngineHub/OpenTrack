import sys
import time
import threading
from pathlib import Path

current_dir = Path(__file__).resolve().parent

import mujoco
import mujoco.viewer

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from viewer_dds import SimState_


class MockG1RobotViewer:
    def __init__(self, sim_dt=0.002, viewer_dt=0.02):
        self.sim_dt = sim_dt
        self.viewer_dt = viewer_dt
        self.start_time = time.time()

        self.xml_path = str(current_dir / "mjcf/scene_mjx_flat_terrain.xml")
        self.mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.mj_model.opt.timestep = sim_dt
        self.mj_data = mujoco.MjData(self.mj_model)
        self.qpos = self.mj_data.qpos.copy()
        self.qvel = self.mj_data.qvel.copy()

        self.simstate_lock = threading.Lock()
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        # receive lowcmd in a low frequency and event triggered
        self.simstate_subscriber = ChannelSubscriber("rt/simstate", SimState_)
        self.simstate_subscriber.Init(self._receive_simstate, 10)

    def _receive_simstate(self, msg: SimState_):
        with self.simstate_lock:
            self.qpos = msg.qpos
            self.qvel = msg.qvel

    def viewer_loop(self):
        with self.simstate_lock:
            self.mj_data.qpos[:] = self.qpos.copy()
            self.mj_data.qvel[:] = self.qvel.copy()

        mujoco.mj_forward(self.mj_model, self.mj_data)
        if self.viewer.is_running():
            self.viewer.sync()


if __name__ == "__main__":
    import os
    iface = os.getenv("SIM_IFACE", "").strip()
    if len(sys.argv) > 1 and sys.argv[1]:
        ChannelFactoryInitialize(0, sys.argv[1])
    elif iface:
        ChannelFactoryInitialize(0, iface)
    else:
        ChannelFactoryInitialize(0)

    mock_g1_robot_viewer = MockG1RobotViewer()

    while True:
        start_time = time.perf_counter()
        mock_g1_robot_viewer.viewer_loop()
        end_time = time.perf_counter()
        time.sleep(max(0, mock_g1_robot_viewer.viewer_dt - (end_time - start_time)))
