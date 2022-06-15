#pragma once

namespace fwi
{
    namespace inversionMethods
    {
        struct cgDeltaAmplification
        {
            double start;
            double slope;

            cgDeltaAmplification()
                : start(0.0)
                , slope(0.0)
            {
            }
            cgDeltaAmplification(double start_, double slope_)
                : start(start_)
                , slope(slope_)
            {
            }
        };
    }   // namespace inversionMethods
}   // namespace fwi
