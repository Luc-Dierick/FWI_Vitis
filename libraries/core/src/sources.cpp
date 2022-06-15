#include "sources.h"
#include "log.h"

namespace fwi
{
    namespace core
    {
        Sources::Sources(const std::array<double, 2> xMin, const std::array<double, 2> xMax, int count_)
            : count(count_)
            , xSrc()
        {
            assert(count > 1);

            std::array<double, 2> dx = calculateDistance(xMin, xMax);

            for(int i = 0; i < count; i++)
            {
                std::array<double, 2> sourceArray;
                for(int j = 0; j < 2; j++)
                {
                    sourceArray[j] = xMin[j] + i * dx[j];
                }
                xSrc.push_back(sourceArray);
            }
        }

        std::array<double, 2> Sources::calculateDistance(const std::array<double, 2> xMin, const std::array<double, 2> xMax)
        {
            std::array<double, 2> dx;
            for(int j = 0; j < 2; j++)
            {
                dx[j] = (xMax[j] - xMin[j]) / (count - 1);
            }
            return dx;
        }

        void Sources::Print()
        {
            L_(io::linfo) << "Total number of Sources is " << count << ". Positions:";
            for(int i = 0; i < count; i++)
            {
                L_(io::linfo) << "x = (" << xSrc[i][0] << ", " << xSrc[i][1] << ")";
            }
        }
    }   // namespace core
}   // namespace fwi
