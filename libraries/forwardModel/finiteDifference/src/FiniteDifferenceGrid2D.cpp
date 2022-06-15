#include "FiniteDifferenceGrid2D.h"

namespace fwi
{
    namespace forwardModels
    {
        /* The finite difference mesh size is different from the cell dimension in, e.g.,
         * a finite volume context, hence the separate class */

        FiniteDifferenceGrid2D::FiniteDifferenceGrid2D(const std::array<double, 2> &x_min_, const std::array<double, 2> &x_max_, const std::array<int, 2> &nx_)
            : x_min(x_min_)
            , x_max(x_max_)
            , nx(nx_)
            , dx()
            , nGridPoints(nx[0] * nx[1])
        {
            // Fix this with std::transform so that we can move it to the
            // initializer list and make all members const
            dx[0] = (x_max[0] - x_min[0]) / static_cast<double>(nx[0] - 1);
            dx[1] = (x_max[1] - x_min[1]) / static_cast<double>(nx[1] - 1);
        }

        const std::array<double, 2> &FiniteDifferenceGrid2D::GetGridEnd() const { return x_max; }

        const std::array<double, 2> &FiniteDifferenceGrid2D::GetGridStart() const { return x_min; }

        const std::array<double, 2> &FiniteDifferenceGrid2D::GetMeshSize() const { return dx; }

        const std::array<int, 2> &FiniteDifferenceGrid2D::GetGridDimensions() const { return nx; }
    }   // namespace forwardModels
}   // namespace fwi
