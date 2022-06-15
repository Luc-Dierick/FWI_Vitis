#pragma once

#include "FiniteDifferenceForwardModelInput.h"
#include "ReadJsonHelper.h"
#include "inputCardReader.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fwi
{
    namespace forwardModels
    {
        class finiteDifferenceForwardModelInputCardReader : public io::inputCardReader
        {
        public:
            finiteDifferenceForwardModelInputCardReader(const std::string &caseFolder, const std::string &filename = "FiniteDifferenceFMInput.json");
            const finiteDifferenceForwardModelInput getInput() const { return _input; }

        private:
            finiteDifferenceForwardModelInput _input;
            const std::string _fileName;
            void readJsonFile(const std::string &filePath);
            void readPMLWidthFactorParameters(const nlohmann::json &jsonFile);
            void readSourceParameters(const nlohmann::json &jsonFile);
            void readBoundaryConditionType(const nlohmann::json &jsonFile);
        };
    }   // namespace forwardModels
}   // namespace fwi
