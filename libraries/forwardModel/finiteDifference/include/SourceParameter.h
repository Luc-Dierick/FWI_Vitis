#pragma once

namespace fwi
{
    namespace forwardModels
    {
        struct SourceParameter
        {
            int r;
            double beta;

            SourceParameter()
                : r(0)
                , beta(0.0)
            {
            }
            SourceParameter(int r_, double beta_)
                : r(r_)
                , beta(beta_)
            {
            }
        };
    }   // namespace forwardModels
}   // namespace fwi
