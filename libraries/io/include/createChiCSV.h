#pragma once

#include "genericInput.h"
#include <streambuf>
#include <string>

namespace fwi
{
    namespace io
    {
        // creates a csv file with the chi values from the file input_filename and save them onto a file called output_filename
        // these csv files could be later used to post-process the chi data as required
        void createCsvFilesForChi(std::string inputFilePath, genericInput input, std::string postfix);
    }   // namespace io
}   // namespace fwi
