#pragma once

#include "json.h"
#include <string>
#include <vector>

namespace fwi
{
    namespace io
    {
        class ReadJsonHelper
        {
        public:
            static void verifyJsonFileContainsParameter(const nlohmann::json &jsonFile, const std::string &filePath, const std::string &parameterName);
            template<typename T> static T tryGetParameterFromJson(const nlohmann::json &jsonFile, const std::string &filePath, const std::string &parameterName)
            {
                verifyJsonFileContainsParameter(jsonFile, filePath, parameterName);
                return jsonFile[parameterName];
            }
        };
    }   // namespace io
}   // namespace fwi
