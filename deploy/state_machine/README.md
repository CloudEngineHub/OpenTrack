# 建立新状态机说明


## state_machine.hpp
1. 在枚举类STATES中添加一个新的状态枚举值，例如：DANCE2 
2. 在类SimpleStateMachine中添加一个新的成员函数如toDance2()，用于处理从其他状态切换到DANCE2状态的逻辑。

## 新建状态机的头文件和源文件 
在这里写部署的核心代码
**新建状态机的头文件和源文件**,并在文件中实现对应状态机类的逻辑，要继承BasicUserController基类，并仿照其他状态机如 fsm_dance_controller 文件来 **重写基类虚函数**。


## robot_controller.hpp

1. 添加对应状态机头文件
2. 在状态机指针对象列表FSMStateList中添加一个新的状态指针成员，例如：FsmDanceController* ctrl_dance2;
3. 在RobotController构造函数中，初始化新添加的状态指针成员。
4. 更新FSMStateList的deletePtr函数，确保在析构时释放所有状态机对象所占用的内存。
5. 在RobotController类中添加一个用于切换到DANCE2状态的公共接口函数，例如：void toDance2Callback();在其中将**基类指针指向新的状态机对象**，**并编写刚切入某一状态机后需要的初始化**等  调用SimpleStateMachine类的toDance2方法，并设置当前状态为DANCE2。
6. 在void UpdateStateMachine()中添加手柄与状态机的映射关系

# 编译 
可能需要修改CMakelists.txt文件，添加新状态机的源码文件路径

为什么推理和发命令写在两个线程里面？难道不应该保证他们的执行顺序吗？