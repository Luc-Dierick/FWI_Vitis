#pragma once

#include "ReadJsonHelper.h"
#include "conjugateGradientInversionInput.h"
#include "inputCardReader.h"
#include <fstream>
#include <sstream>
#include <string>

namespace fwi
{
    namespace inversionMethods
    {
        class ConjugateGradientInversionInputCardReader : public io::inputCardReader
        {
        private:
            ConjugateGradientInversionInput _input;

            const std::string _fileName;
            void readJsonFile(const std::string &filePath);

            void readIterParameter(const nlohmann::json &jsonFile);
            void readDeltaAmplificationParameter(const nlohmann::json &jsonFile);

        public:
            ConjugateGradientInversionInputCardReader(const std::string &caseFolder, const std::string &filename = "ConjugateGradientInversionInput.json");
            const ConjugateGradientInversionInput getInput() const { return _input; }
        };
    }   // namespace inversionMethods
}   // namespace fwi
