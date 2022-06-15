#include "ReadJsonHelper.h"
#include "json.h"
#include <exception>
#include <iostream>
#include <string.h>

namespace fwi
{
    namespace io
    {
        void ReadJsonHelper::verifyJsonFileContainsParameter(const nlohmann::json &jsonFile, const std::string &filePath, const std::string &parameterName)
        {
            if(!jsonFile.contains(parameterName))
            {
                throw std::invalid_argument("Argument " + parameterName + " is missing in: " + filePath);
            }
        }
    }   // namespace io
}   // namespace fwi
