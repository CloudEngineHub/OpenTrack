# sim_interface 使用说明

## 1. 这是什么

`sim_interface` 是给 `G1_deploy` 准备的一层 Mujoco 仿真适配器。

它的目标不是重写一套控制器，而是尽量复用真机部署链路已经使用的 DDS API：

- 订阅 `rt/lowcmd`，把 `G1_deploy` 发出的关节目标应用到 Mujoco。
- 发布 `rt/lowstate`，让 `G1_deploy` 以为自己在读真机低层状态。
- 发布一个额外的 `rt/simstate`，只给本地可视化进程用。
- 尝试启动一个 `MotionSwitcherServer`；如果当前 `unitree_sdk2py` 版本里没有该模块，会自动降级为 no-op，不阻塞主链路。

可以把它理解成：

```text
G1_deploy/state_machine 负责控制逻辑
sim_interface            负责伪造真机侧 DDS 接口和 Mujoco 物理环境
```

只要 DDS 话题和消息类型保持一致，`G1_deploy` 的主体代码就不需要为了仿真单独改一份接口。

---

## 2. 目录结构

当前目录下几个关键文件的职责如下：

| 文件 | 作用 |
| --- | --- |
| `main.py` | 拉起两个子进程：`robot_entry.py` 和 `viewer_entry.py`，并负责统一退出清理；会持续监控子进程状态，任一子进程异常退出时主进程会报错并触发清理。 |
| `robot_entry.py` | 仿真主进程。加载 Mujoco 模型，订阅 `rt/lowcmd`，发布 `rt/lowstate` 和 `rt/simstate`；对 `MotionSwitcherServer` 缺失提供兼容降级。 |
| `viewer_entry.py` | 可视化进程。订阅 `rt/simstate`，在单独的 Mujoco viewer 中显示机器人状态。 |
| `viewer_dds.py` | 给 viewer 定义自用的 DDS 消息 `SimState_`。 |
| `wireless_controller.py` | 用键盘模拟 Unitree 无线手柄，并把按键编码成 `wireless_remote` 字节流；支持字符键、方向键、以及 X11 小键盘 vk 映射。 |
| `mjcf/` | Mujoco 场景、机器人 XML 和网格资源。 |

当前默认使用的场景文件是：

- `mjcf/scene_mjx_flat_terrain.xml`

它会进一步 include：

- `mjcf/g1_mjx.xml`
- `mjcf/assets/` 里的 STL 和贴图资源

---

## 3. 它和真机接口如何对齐

`G1_deploy/state_machine` 侧最关键的两个话题是：

- `rt/lowcmd`
- `rt/lowstate`

这两个名字在 `robot_controller.hpp` 里就是硬编码常量，所以仿真端也严格复用了同名话题。

### 3.1 `rt/lowcmd`

- 来源：`G1_deploy` 的 `RobotController::LowCmdwriteHandler()`
- 去向：`sim_interface/robot_entry.py`

仿真端收到 `LowCmd_` 后，会对 29 个关节按下面的公式写控制量：

```text
tau + kp * (q_des - q) + kd * (dq_des - dq)
```

这意味着仿真里测试到的是：

- 话题类型是否对齐
- 目标关节角、KP、KD 是否正常下发
- 控制链路在 API 层能不能跑通

### 3.2 `rt/lowstate`

- 来源：`sim_interface/robot_entry.py`
- 去向：`G1_deploy` 的 `RobotController::LowStateMessageHandler()`

仿真端会从 Mujoco 当前状态构造：

- IMU 四元数、角速度、加速度、RPY
- 29 个主关节的 `q/dq/ddq/tau_est`
- `wireless_remote` 字节流
- CRC

这让 `G1_deploy` 的 `RobotInterface`、手柄解释和状态机切换逻辑都可以沿用真机代码。

### 3.3 `rt/simstate`

这是仿真自用话题，不参与真机 API 对齐。

- 来源：`robot_entry.py`
- 去向：`viewer_entry.py`

它只携带 viewer 需要的 `qpos/qvel/tick`，避免可视化和控制主链耦合在一起。

---

## 4. 运行前准备

## 4.1 系统前提

推荐在 Linux 图形桌面环境下运行，因为当前实现同时依赖：

- `mujoco.viewer` 的本地窗口
- `pynput` 的键盘监听

如果你是在纯 SSH 无桌面环境下调试：

- 可以只运行 `robot_entry.py`，不启动 viewer。
- 但键盘监听通常仍然需要可用的桌面会话或正确的输入设备权限。

## 4.2 Python 依赖

`sim_interface` 当前直接 import 了这些 Python 依赖：

- `numpy`
- `scipy`
- `mujoco`
- `pynput`
- `cyclonedds`
- `unitree_sdk2py`

其中最关键的是两类：

1. Mujoco 运行依赖
2. Unitree Python SDK 依赖

建议至少先安装：

```bash
python -m pip install numpy scipy mujoco==3.3.1 pynput cyclonedds
```

其中 `mujoco==3.3.1` 是当前训练环境依赖里已经出现的版本，和这个仓库更一致。

## 4.3 安装 `unitree_sdk2py`

仓库里没有内置 `unitree_sdk2py` 源码，所以这部分需要你自己准备。

推荐做法：

1. 克隆 Unitree 官方 Python SDK 仓库。
2. 在同一个 Python 环境里执行可编辑安装。

示例：

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
python -m pip install -e .
```

如果这一步没做，`robot_entry.py` 和 `viewer_entry.py` 会直接报：

```text
ModuleNotFoundError: No module named 'unitree_sdk2py'
```

另外，不同版本的 `unitree_sdk2py` 里，`motion_switcher_server` 子模块可能不存在。

当前实现已对这一点做了兼容：

- 会打印 `[WARN] MotionSwitcherServer not available...`
- 仿真主链路仍会继续运行，不会因为这个模块缺失而直接退出。

## 4.4 检查模型资源

启动前至少确认下面这几个路径存在：

```text
sim_interface/mjcf/scene_mjx_flat_terrain.xml
sim_interface/mjcf/g1_mjx.xml
sim_interface/mjcf/assets/
```

当前仓库里 `assets/` 已经存在，包含 G1 所需的 STL 和贴图资源。

---

## 5. 快速启动

最常用的是直接从 `sim_interface` 目录启动：

```bash
cd /home/lianyunrui/Projects/G1TrackingDeploy-Any2Track/sim_interface
python main.py
```

它会拉起两个子进程：

1. `robot_entry.py`
2. `viewer_entry.py`

正常情况下你会看到：

- 一个 Mujoco viewer 窗口
- 控制台输出 `Simulation thread started`
- 控制台输出 `Viewer thread started`
- 控制台输出 `Motion switcher server started`
- 控制台输出 `Keyboard listener started`
- 控制台输出 `Simulation hold mode active (default pose). Press key '9' to enable physics stepping`

如果 `robot_entry.py` 或 `viewer_entry.py` 在启动后异常退出，`main.py` 会直接报错并退出，例如：

```text
主进程遇到错误: 子进程 'robot_entry.py' 意外退出，退出码: <code>
```

这可以避免“viewer 还开着、但机器人 DDS 主链已经挂掉”的假成功状态。

如果你只想保留 DDS 仿真，不想开 viewer，可以单独运行：

```bash
cd /home/lianyunrui/Projects/G1TrackingDeploy-Any2Track/sim_interface
python robot_entry.py
```

---

## 6. 如何和 G1_deploy 联调

推荐按下面顺序操作。

### 6.1 第一步：先启动仿真端

终端 1：

```bash
cd /home/lianyunrui/Projects/G1TrackingDeploy-Any2Track/sim_interface
python main.py
```

### 6.2 第二步：编译 G1_deploy

终端 2：

```bash
cd /home/lianyunrui/Projects/G1TrackingDeploy-Any2Track/G1_deploy
./build.sh
```

### 6.3 第三步：直接从 build/bin 启动状态机

这里建议不要直接复用 `start_deploy.sh` 做仿真联调，原因有两个：

1. 当前 `main.cpp` 实际没有消费脚本传入的网卡名。
2. `start_deploy.sh` 里的 `PARAM_PATH` 还是旧的 `../../example/state_machine/params/`，而当前仓库实际参数目录是 `../../state_machine/params/`。

更稳妥的启动方式是：

```bash
cd /home/lianyunrui/Projects/G1TrackingDeploy-Any2Track/G1_deploy/build/bin
./state_machine_example --param ../../state_machine/params/
```

正常情况下，状态机会在终端循环输出：

```text
Press R2 to start!
```

等待阶段还会周期输出一条调试信息：

```text
WaitR2 debug: remote[2]=..., remote[3]=..., parsed_r2=...
```

这条日志可用于快速判断 `R2` 是否已经进入 `wireless_remote`。

### 6.4 第四步：用键盘模拟遥控器

注意：这里的“手柄按键”不是在 `G1_deploy` 终端里输入，而是通过 `sim_interface` 的键盘监听编码到 `wireless_remote` 后，再随 `rt/lowstate` 发给状态机。

所以你要做的是：

1. 让仿真会话处于活跃状态。
2. 在本机键盘上按映射按键。

常用映射如下：

| 键盘 | 模拟手柄 | 作用 |
| --- | --- | --- |
| `4` | `R2` | 启动 `G1_deploy` 控制线程 |
| `a` | `A` | 切到 `STAND` |
| `x` | `X` | 切到 `LOCO` |
| `b` | `B` | 切换 tracking 模式组 |
| `1` | `L1` | 方向键组合修饰键 |
| `2` | `L2` | 方向键组合修饰键 |
| `3` | `R1` | 方向键组合修饰键 |
| 方向键 | `Up/Down/Left/Right` | 选择 tracking 槽位 |
| `r` | `Reset` | 重置 Mujoco 中的机器人姿态 |
| `6` | `F1` | 退出程序 |
| `8` | `Start` | 模拟 Start 键 |
| `9` | `SimStart` | 启用仿真步进（`mj_step`） |

补充说明：

- 在 X11 环境下，支持小键盘 `KP_1` ~ `KP_8` 对应 `L1/L2/R1/R2/Select/F1/F3/Start`。
- 在 X11 环境下，支持小键盘 `KP_1` ~ `KP_9`，其中 `KP_9` 对应 `SimStart`。
- 按键监听默认只打印命中的按键（例如 `Button R2 pressed`）。
- 如需排查“按键没有命中映射”，可启用：

```bash
SIM_KEY_DEBUG=1 python main.py
```

这会输出未映射键的 token 信息，便于补映射。

摇杆映射如下：

| 键盘 | 模拟摇杆 |
| --- | --- |
| `j / l` | 左摇杆 X 负 / 正 |
| `k / i` | 左摇杆 Y 负 / 正 |
| `u / o` | 右摇杆 X 负 / 正 |

一个最小联调顺序通常是：

1. 在仿真端按 `4`，相当于 `R2`，启动控制线程。
2. 按 `a` 进入站立。
3. 按 `x` 进入 locomotion。
4. 按 `9` 启用仿真步进（此时机器人才开始真正随动力学演化）。
5. 再按方向键或 `L1/R1/L2/R2 + 方向键` 触发 tracking controller。

---

## 7. 运行行为说明

## 7.1 线程频率

`robot_entry.py` 当前有两个周期线程：

- 仿真步进线程：`sim_dt = 0.002`，也就是 500 Hz
- 状态转发线程：`viewer_dt = 0.02`，也就是 50 Hz

其中：

- `rt/lowstate` 由仿真线程高频发布
- `rt/simstate` 由 viewer 线程低频发布

补充：

- 程序启动后默认处于 hold 模式，不执行 `mj_step`。
- hold 模式下会持续把机器人钉在 `DEFAULT_QPOS`，并继续发布 `rt/lowstate`，用于先完成 `R2/A` 等流程。
- 按下 `SimStart`（键盘 `9`）后，开始真实物理步进。

## 7.2 默认姿态

仿真启动后，即使还没有外部控制器发命令，内部也会用 `DEFAULT_QPOS`、`KPs`、`KDs` 把机器人维持在一个默认站立姿态附近。

所以你看到机器人一开始就是立起来的，这不是外部策略已经启动，而是仿真端自己给了初始 PD 目标。

## 7.3 重置行为

按下 `r` 时，仿真会：

1. `mj_resetData`
2. 把 `qpos` 恢复成 `DEFAULT_QPOS`
3. 重新 `mj_forward`

这一步只影响 Mujoco 仿真态，不会自动重置 `G1_deploy` 里已经运行的 controller 内部状态。

---

## 8. 已知限制

当前 `sim_interface` 更适合做“API 联通测试”和“基础控制观测”，不等于一套完整真机替身。已知限制如下：

1. `LowState_` 里有些字段只是占位值，例如 `version`、`mode_pr`、`mode_machine`、`reserve`。
2. 只认真填充了前 29 个主电机关节状态，后 6 个电机状态仍然是默认空对象。
3. `tau_est` 使用的是 Mujoco 里的约束力相关量，和真机电机估计力矩不是严格等价。
4. 当前 viewer 只是被动显示 `qpos/qvel`，没有额外调试 UI。
5. 键盘映射目前没有给右摇杆 Y 轴分配按键，所以依赖 `Ry` 的逻辑在这里测不到。
6. 仿真端只模拟了状态机需要的主链路，不覆盖真实硬件所有 side effect。

---

## 9. 常见问题

## 9.1 `ModuleNotFoundError: No module named 'unitree_sdk2py'`

说明 Python SDK 没装，按前面的 `unitree_sdk2_python` 安装步骤补齐即可。

## 9.2 Mujoco 打不开 XML 或找不到 mesh

优先检查：

- `sim_interface/mjcf/scene_mjx_flat_terrain.xml`
- `sim_interface/mjcf/g1_mjx.xml`
- `sim_interface/mjcf/assets/`

通常是资源目录不完整，或者当前工作目录不对。

## 9.3 `G1_deploy` 一直停在 `Press R2 to start!`

优先确认 4 件事：

1. `sim_interface/robot_entry.py` 是否已经启动。
2. DDS 双方是否在同一台机器、同一个默认 domain 下。
3. 仿真端是否打印了 `Button R2 pressed/released`。
4. `G1_deploy` 侧 `WaitR2 debug` 是否出现 `remote[2]=16`（表示 `R2` 位已经置 1）。

常见判读：

- `Button R2 pressed` 有，但 `remote[2]` 始终为 0：说明按键事件没进入 lowstate 发布链，优先检查仿真侧是否真的在跑 `robot_entry.py`。
- `remote[2]=16` 且 `parsed_r2=1`：下一步应进入 `--------------- Start ---------------`。
- 若按的是数字小键盘，优先尝试主键盘 `4` 交叉验证。

## 9.4 viewer 环境报错

如果你当前机器没有可用图形界面，可以先只运行：

```bash
python robot_entry.py
```

先验证 DDS 和控制主链是否正常，再单独处理图形环境。

## 9.5 启动时出现 `MotionSwitcherServer` 警告

若看到类似：

```text
[WARN] MotionSwitcherServer not available, running without mode switcher service
```

这在当前实现中是可接受的兼容降级，不会阻断 `rt/lowcmd` / `rt/lowstate` 主链路。

---

## 10. 建议的最小验收流程

如果你只是想确认 `sim_interface` 已经配置好，可以按这个最小流程：

1. 启动 `python main.py`，确认 viewer 正常打开。
2. 启动 `G1_deploy/build/bin/state_machine_example --param ../../state_machine/params/`。
3. 在仿真端按 `4`，让状态机离开等待态。
4. 按 `a` 再按 `x`，确认状态机能进入 `STAND -> LOCO`。
5. 按一个方向键，确认 `rt/lowcmd` 已经驱动 Mujoco 机器人持续运动。

做到这一步，就说明这套仿真接口已经能承担“真机 API 冒烟测试”的作用。