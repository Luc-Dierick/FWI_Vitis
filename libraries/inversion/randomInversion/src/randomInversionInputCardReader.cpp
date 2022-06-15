#include "randomInversionInputCardReader.h"
#include "ReadJsonHelper.h"
#include "json.h"
#include <iostream>

namespace fwi
{
    namespace inversionMethods
    {
        RandomInversionInputCardReader::RandomInversionInputCardReader(const std::string &caseFolder)
            : io::inputCardReader()
        {
            static const std::string stringInputFolder = "/input/";
            std::string filePath = caseFolder + stringInputFolder + _fileName;

            readJsonFile(filePath);
        }

        void RandomInversionInputCardReader::readJsonFile(const std::string &filePath)
        {
            nlohmann::json jsonFile = readFile(filePath);

            static const std::string parameterToleranceOuter = "toleranceOuter";
            double toleranceOuter = io::ReadJsonHelper::tryGetParameterFromJson<double>(jsonFile, _fileName, parameterToleranceOuter);
            if(toleranceOuter <= 0)
            {
                throw std::invalid_argument("Invalid outer tolerance (" + std::to_string(toleranceOuter) + " <= 0) in: " + _fileName);
            }
            _input.toleranceOuter = toleranceOuter;

            static const std::string parameterMaxIterationsInner = "nMaxInner";
            int maxIterationsInner = io::ReadJsonHelper::tryGetParameterFromJson<int>(jsonFile, _fileName, parameterMaxIterationsInner);
            if(maxIterationsInner <= 0)
            {
                throw std::invalid_argument("Invalid max numer of iterations inner (" + parameterMaxIterationsInner + ") (" +
                                            std::to_string(maxIterationsInner) + " <= 0) in: " + _fileName);
            }
            _input.nMaxInner = maxIterationsInner;

            static const std::string parameterMaxIterationsOuter = "nMaxOuter";
            int maxIterationsOuter = io::ReadJsonHelper::tryGetParameterFromJson<int>(jsonFile, _fileName, parameterMaxIterationsOuter);
            if(maxIterationsOuter <= 0)
            {
                throw std::invalid_argument("Invalid max numer of iterations outer (" + parameterMaxIterationsOuter + ") (" +
                                            std::to_string(maxIterationsOuter) + " <= 0) in: " + _fileName);
            }
            _input.nMaxOuter = maxIterationsOuter;
        }
    }   // namespace inversionMethods
}   // namespace fwi
