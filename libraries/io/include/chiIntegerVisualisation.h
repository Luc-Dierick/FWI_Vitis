#pragma once

#include <cmath>
#include <fstream>
#include <iostream>

namespace fwi
{
    namespace io
    {
        // this function visualises the chi field in an integer matrix form (where chi is normalised between values 2 and 7), where 2 represents background
        // medium and 7 represents the subsurface. 1 and 8 are acceptable out of bounds, while 0 and 9 imply a relatively significant out of bound estimation of
        // chi
        void chi_visualisation_in_integer_form(std::string filename, int nxt);
    }   // namespace io
}   // namespace fwi
