#include "receivers.h"
#include "log.h"

namespace fwi
{
    namespace core
    {
        Receivers::Receivers(const std::array<double, 2> xMin, const std::array<double, 2> xMax, int count_)
            : count(count_)
            , xRecv()
        {
            assert(count > 1);

            std::array<double, 2> dx = calculateDistance(xMin, xMax);

            for(int i = 0; i < count; i++)
            {
                std::array<double, 2> receiverArray;
                for(int j = 0; j < 2; j++)
                {
                    receiverArray[j] = xMin[j] + i * dx[j];
                }
                xRecv.push_back(receiverArray);
            }
        }

        std::array<double, 2> Receivers::calculateDistance(const std::array<double, 2> xMin, const std::array<double, 2> xMax)
        {
            std::array<double, 2> dx;
            for(int j = 0; j < 2; j++)
            {
                dx[j] = (xMax[j] - xMin[j]) / (count - 1);
            }
            return dx;
        }

        void Receivers::Print()
        {
            L_(io::linfo) << "Total number is Receivers is " << count << ". Positions:";
            for(int i = 0; i < count; i++)
            {
                L_(io::linfo) << "x = (" << xRecv[i][0] << ", " << xRecv[i][1] << ")";
            }
        }
    }   // namespace core
}   // namespace fwi
