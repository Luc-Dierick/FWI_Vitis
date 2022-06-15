#pragma once

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fwi
{
    namespace utilities
    {
        inline double dist(double x, double z) { return std::pow(std::pow(z, 2.0) + std::pow(x, 2.0), 0.5); }

        inline std::vector<std::string> returnInputDirectory(int c, char **v = NULL)
        {
            if(c < 2)
            {
                std::vector<std::string> arguments = {"./"};
                return arguments;
            }
            std::vector<std::string> arguments(v + 1, c + v);
            return arguments;
        }
    }   // namespace utilities
}   // namespace fwi
