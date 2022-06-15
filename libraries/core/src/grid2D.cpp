#include "grid2D.h"

namespace fwi
{
    namespace core
    {
        grid2D::grid2D(const std::array<double, 2> &x_min_, const std::array<double, 2> &x_max_, const std::array<int, 2> &nx_)
            : _xMin(x_min_)
            , _xMax(x_max_)
            , _nx(nx_)
            , _dx()
            , _nGridPoints()
            , _cellVolume()
        {
            // Fix this with std::transform so that we can move it to the
            // initializer list and make all members const
            _dx[0] = (_xMax[0] - _xMin[0]) / static_cast<double>(_nx[0]);
            _dx[1] = (_xMax[1] - _xMin[1]) / static_cast<double>(_nx[1]);

            _cellVolume = _dx[0] * _dx[1];
            _nGridPoints = _nx[0] * _nx[1];   // Should be renamed to nCells.
        }

        bool grid2D::operator==(const grid2D &rhs) const
        {
            if(_xMin != rhs._xMin || _xMax != rhs._xMax || _nx != rhs._nx || _dx != rhs._dx)
            {
                return false;
            }
            return true;
        }
    }   // namespace core
}   // namespace fwi
