#pragma once
#include <iostream>
#include <vector>

namespace fwi
{
    namespace io
    {
        struct argumentReader
        {
            std::string dir;
            std::string forward;
            std::string inversion;
            std::string stepdir;
            std::string stepsize;
            bool help;
            bool version;

            argumentReader() = delete;  
            argumentReader(std::vector<std::string>& arguments);
        };
    }   // namespace io
}   // namespace fwi
