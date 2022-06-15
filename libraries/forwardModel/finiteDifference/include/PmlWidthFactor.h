#pragma once

namespace fwi
{
    namespace forwardModels
    {
        struct PMLWidthFactor
        {
            double x;
            double z;

            PMLWidthFactor()
                : x(0.0)
                , z(0.0)
            {
            }
            PMLWidthFactor(int x_, double z_)
                : x(x_)
                , z(z_)
            {
            }
        };
    }   // namespace forwardModels
}   // namespace fwi
