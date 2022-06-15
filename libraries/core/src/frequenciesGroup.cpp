#include "frequenciesGroup.h"
#include "log.h"

namespace fwi
{
    namespace core
    {
        static const double pi = std::atan(1.0) * 4.0;

        FrequenciesGroup::FrequenciesGroup(freqInfo freqStruct, double c_0_)
            : count(freqStruct.nTotal)
            , c0(c_0_)
            , dFreq((freqStruct.max - freqStruct.min) / (std::max(freqStruct.nTotal, 2) - 1))
            , freq()
            , k()
        {
            if(std::abs(pi - 3.14159) > 0.0001)
            {
                throw std::invalid_argument("pi differs to much from the real value of pi.");
            }
            for(int i = 0; i < freqStruct.nTotal; i++)
            {
                freq.push_back(freqStruct.min + i * dFreq);
                double omega = 2.0 * pi * freq[i];
                k.push_back(omega / c0);
            }
        }

        void FrequenciesGroup::Print(int nFreqTotal)
        {
            L_(io::linfo) << "Total number of frequencies is " << nFreqTotal << ". Values:";
            for(int i = 0; i < nFreqTotal; i++)
            {
                L_(io::linfo) << "f = " << freq[i] << ", k = " << k[i];
            }
        }
    }   // namespace core
}   // namespace fwi
