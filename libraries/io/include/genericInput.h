#pragma once

#include "freqInfo.h"
#include <array>
#include <string>

namespace fwi
{
    namespace io
    {
        struct genericInput
        {
            std::string caseFolder;
            std::string inputFolder;
            std::string outputLocation;
            std::string runName;
            std::string fileName;
            std::string forward;
            std::string inversion;
            double c0;
            core::freqInfo freq;
            std::array<double, 2> reservoirTopLeftCornerInM;
            std::array<double, 2> reservoirBottomRightCornerInM;
            std::array<double, 2> sourcesTopLeftCornerInM;
            std::array<double, 2> sourcesBottomRightCornerInM;
            std::array<double, 2> receiversTopLeftCornerInM;
            std::array<double, 2> receiversBottomRightCornerInM;
            std::array<int, 2> nGridOriginal;
            std::array<int, 2> nGrid;
            int nSources;
            int nReceivers;
            bool verbose;
        };
    }   // namespace io
}   // namespace fwi
