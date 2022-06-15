#pragma once
#include "genericInput.h"
#include "inversionInterface.h"
#include "log.h"

namespace fwi
{
    namespace inversionMethods
    {
        struct RegularisationParameters
        {
            double deltaSquared;
            double errorFunctional;
            std::vector<core::dataGrid2D<double>> gradientChi;
            core::dataGrid2D<double> gradientChiNormSquared;
            core::dataGrid2D<double> b;
            core::dataGrid2D<double> bSquared;
            core::dataGrid2D<double> gradient;

            RegularisationParameters(const core::grid2D &grid_)
                : deltaSquared(0.0)
                , errorFunctional(1.0)
                , gradientChi(std::vector<core::dataGrid2D<double>>(2, core::dataGrid2D<double>(grid_)))
                , gradientChiNormSquared(grid_)
                , b(grid_)
                , bSquared(grid_)
                , gradient(grid_)
            {
            }
        };
    }   // namespace inversionMethods
}   // namespace fwi
