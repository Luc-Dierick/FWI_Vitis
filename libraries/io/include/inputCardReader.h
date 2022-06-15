#pragma once

#include "json.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fwi
{
    namespace io
    {
        class inputCardReader
        {
        public:
            virtual ~inputCardReader() = default;

        protected:
            static nlohmann::json readFile(const std::string &fileName);
            static std::string removeLastSlash(const std::string &caseFolderWithSlash);
            static std::string getRunName(const std::string &caseFolder);
        };
    }   // namespace io
}   // namespace fwi
