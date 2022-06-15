#pragma once

namespace fwi
{
    namespace inversionMethods
    {
        struct gradientDescentInversionInput
        {
            double gamma0;
            double x0;
            double h;
            int iter;
        };
    }   // namespace inversionMethods
}   // namespace fwi
