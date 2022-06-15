#pragma once

#include <array>

namespace fwi
{
    namespace forwardModels
    {
        class FiniteDifferenceGrid2D
        {
        public:
            FiniteDifferenceGrid2D(const std::array<double, 2> &x_min_, const std::array<double, 2> &x_max_, const std::array<int, 2> &nx_);

            const std::array<int, 2> &GetGridDimensions() const;
            const std::array<double, 2> &GetMeshSize() const;
            const std::array<double, 2> &GetGridStart() const;
            const std::array<double, 2> &GetGridEnd() const;

            int GetNumberOfGridPoints() const { return nGridPoints; }
            double GetDomainArea() const { return (x_max[0] - x_min[0]) * (x_max[1] - x_min[1]); }

        private:
            const std::array<double, 2> x_min;
            const std::array<double, 2> x_max;
            const std::array<int, 2> nx;
            std::array<double, 2> dx;
            const int nGridPoints;
        };
    }   // namespace forwardModels
}   // namespace fwi
