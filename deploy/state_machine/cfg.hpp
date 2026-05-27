#pragma once

#include "unitree/common/json/jsonize.hpp"
#include <vector>
#include <iostream>

namespace unitree::common
{
    class ExampleCfg : public Jsonize
    {
    public:
        ExampleCfg() : kp{}, kd{}, dt(0)
        {
            kp.fill(0);
            kd.fill(0);
        }

        void fromJson(JsonMap &json)
        {
            std::vector<float> temp;
            FromJson(json["kp"], temp);
            std::copy_n(temp.begin(), 29, kp.begin());
            std::vector<float> temp1;
            FromJson(json["kd"], temp1);
            std::copy_n(temp1.begin(), 29, kd.begin());
            
            FromJson(json["dt"], dt);
            FromAny<float>(json["init_pos"], init_pos);
        }

        void toJson(JsonMap &json) const
        {
            ToJson(std::vector<float>(kp.begin(), kp.end()), json["kp"]);
            ToJson(std::vector<float>(kd.begin(), kd.end()), json["kd"]);
            ToJson(dt, json["dt"]);
            ToAny<float>(init_pos, json["init_pos"]);
        }

        std::array<float, 29> kp;
        std::array<float, 29> kd;
        float dt;

        std::vector<float> init_pos;
    };
}