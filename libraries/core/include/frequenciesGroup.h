#pragma once

#include <cassert>
#include <iostream>
#include <vector>
#define _USE_MATH_DEFINES
#include "freqInfo.h"
#include <cmath>

namespace fwi
{
    namespace core
    {
        class FrequenciesGroup
        {
            FrequenciesGroup operator=(const FrequenciesGroup &) = delete;

        public:
            const int count;
            const double c0;
            const double dFreq;
            std::vector<double> freq;
            std::vector<double> k;

            FrequenciesGroup(freqInfo freqStruct, double c_0_);

            void Print(int nFreqTotal);
        };
    }   // namespace core
}   // namespace fwi
