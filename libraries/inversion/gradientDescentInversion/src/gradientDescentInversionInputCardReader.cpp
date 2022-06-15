#include "gradientDescentInversionInputCardReader.h"
#include "json.h"
#include <iostream>

namespace fwi
{
    namespace inversionMethods
    {
        gradientDescentInversionInputCardReader::gradientDescentInversionInputCardReader(const std::string &caseFolder, const std::string &filename)
            : io::inputCardReader()
            , _fileName(filename)
        {
            const std::string stringInputFolder = "/input/";
            std::string filePath = caseFolder + stringInputFolder + _fileName;

            readJsonFile(filePath);
        }

        void gradientDescentInversionInputCardReader::readJsonFile(const std::string &filePath)
        {
            nlohmann::json jsonFile = readFile(filePath);

            const std::string parameterGamma = "gamma0";
            _input.gamma0 = io::ReadJsonHelper::tryGetParameterFromJson<double>(jsonFile, _fileName, parameterGamma);

            const std::string parameterInitialX = "x0";
            _input.x0 = io::ReadJsonHelper::tryGetParameterFromJson<double>(jsonFile, _fileName, parameterInitialX);

            const std::string parameterStepSize = "h";
            double stepSize = io::ReadJsonHelper::tryGetParameterFromJson<double>(jsonFile, _fileName, parameterStepSize);
            if(stepSize <= 0)
            {
                throw std::invalid_argument("Invalid step size h (" + std::to_string(stepSize) + " <= 0) in: " + _fileName);
            }
            _input.h = stepSize;

            const std::string parameterIteration = "iter";
            int nrOfIterations = io::ReadJsonHelper::tryGetParameterFromJson<int>(jsonFile, _fileName, parameterIteration);
            if(nrOfIterations <= 0)
            {
                throw std::invalid_argument("Invalid numer of iterations (" + std::to_string(nrOfIterations) + " <= 0) in: " + _fileName);
            }
            _input.iter = nrOfIterations;
        }
    }   // namespace inversionMethods
}   // namespace fwi
