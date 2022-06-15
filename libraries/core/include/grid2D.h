#pragma once

#include <array>

namespace fwi
{
    namespace core
    {
        class grid2D
        {
        private:
            const std::array<double, 2> _xMin;
            const std::array<double, 2> _xMax;
            const std::array<int, 2> _nx;
            std::array<double, 2> _dx;

            int _nGridPoints;
            double _cellVolume;

            grid2D &operator=(const grid2D &) = delete;

        public:
            grid2D(const std::array<double, 2> &x_min_, const std::array<double, 2> &x_max_, const std::array<int, 2> &nx_);

            const std::array<int, 2> &getGridDimensions() const { return _nx; }
            const std::array<double, 2> &getCellDimensions() const { return _dx; }
            const std::array<double, 2> &getGridStart() const { return _xMin; }
            const std::array<double, 2> &getGridEnd() const { return _xMax; }

            int getNumberOfGridPoints() const { return _nGridPoints; }
            double getCellVolume() const { return _cellVolume; }
            double getDomainArea() const { return (_xMax[0] - _xMin[0]) * (_xMax[1] - _xMin[1]); }

            bool operator==(const grid2D &rhs) const;
        };
    }   // namespace core
}   // namespace fwi
