#pragma once

namespace fwi
{
    namespace core
    {
        struct freqInfo
        {
            double min;
            double max;
            int nTotal;

            freqInfo()
                : min(0.0)
                , max(0.0)
                , nTotal(0)
            {
            }
            freqInfo(double min_, double max_, int nTotal_)
                : min(min_)
                , max(max_)
                , nTotal(nTotal_)
            {
            }
        };
    }   // namespace core
}   // namespace fwi
