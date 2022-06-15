#pragma once

namespace fwi
{
    namespace inversionMethods
    {
        struct RandomInversionInput
        {
            double toleranceOuter;
            int nMaxOuter;
            int nMaxInner;
        };
    }   // namespace inversionMethods
}   // namespace fwi
