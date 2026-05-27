#pragma once

#include <algorithm>

namespace unitree::common
{
    enum class STATES
    {
        DAMPING = 0,
        STAND = 1,
        DANCE = 2,
        LOCO = 3 // 用遥控器控制走路策略
    };

    // 简单切换状态机类，每一个函数检测是否可以切换为函数名所述状态枚举
    class SimpleStateMachine
    {
    public:
        SimpleStateMachine(double pd_ratio_init = 0.1, double delta_pd = 0.005) : state(STATES::DAMPING) {}

        bool Stop()
        {
            state = STATES::DAMPING;
            return true;
        }

        bool toStand()
        {
            if (state == STATES::DAMPING || state == STATES::DANCE || state == STATES::LOCO)
            {
                state = STATES::STAND;
                return true;
            }
            else
            {
                return false;
            }
        }

        bool toDance()
        {
            if (state == STATES::LOCO)
            {
                state = STATES::DANCE;
                return true;
            }
            else
            {
                return false;
            }
        }

        bool toLoco()
        {
            if (state == STATES::STAND || state == STATES::DANCE)
            {
                state = STATES::LOCO;
                return true;
            }
            else
            {
                return false;
            }
        }

        STATES state;
    };
} // namespace unitree
