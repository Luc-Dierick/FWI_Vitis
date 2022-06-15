#include "conjugateGradientInversionInputCardReader.h"
#include "json.h"
#include <iostream>

namespace fwi
{
    namespace inversionMethods
    {
        ConjugateGradientInversionInputCardReader::ConjugateGradientInversionInputCardReader(const std::string &caseFolder, const std::string &filename)
            : io::inputCardReader()
            , _fileName(filename)
        {
            const std::string stringInputFolder = "/input/";
            std::string filePath = caseFolder + stringInputFolder + _fileName;

            readJsonFile(filePath);
        }

        void ConjugateGradientInversionInputCardReader::readJsonFile(const std::string &filePath)
        {
            nlohmann::json jsonFile = readFile(filePath);

            readIterParameter(jsonFile);
            readDeltaAmplificationParameter(jsonFile);

            const std::string parameterRegularisationEnabled = "do_reg";
            _input.doRegularisation = io::ReadJsonHelper::tryGetParameterFromJson<bool>(jsonFile, _fileName, parameterRegularisationEnabled);

            const std::string parameterMaxNrOfIterations = "n_max";
            int maxNrOfIterations = io::ReadJsonHelper::tryGetParameterFromJson<int>(jsonFile, _fileName, parameterMaxNrOfIterations);
            if(maxNrOfIterations <= 0)
            {
                throw std::invalid_argument(
                    "Invalid number of iterations" + parameterMaxNrOfIterations + " (" + std::to_string(maxNrOfIterations) + " <= 0) in: " + _fileName);
            }
            _input.n_max = maxNrOfIterations;
        }

        void ConjugateGradientInversionInputCardReader::readIterParameter(const nlohmann::json &jsonFile)
        {
            const std::string parameterIter = "Iter1";
            const std::string parameterNumber = "n";
            const std::string parameterTolerance = "tolerance";

            nlohmann::json iterObject = io::ReadJsonHelper::tryGetParameterFromJson<nlohmann::json>(jsonFile, _fileName, parameterIter);

            int nrOfIterations = io::ReadJsonHelper::tryGetParameterFromJson<int>(iterObject, _fileName, parameterNumber);
            if(nrOfIterations <= 0)
            {
                throw std::invalid_argument("Invalid numer of iterations (" + std::to_string(nrOfIterations) + " <= 0) in: " + _fileName);
            }

            double tolerance = io::ReadJsonHelper::tryGetParameterFromJson<double>(iterObject, _fileName, parameterTolerance);
            if(tolerance <= 0)
            {
                throw std::invalid_argument("Invalid tolerance (" + std::to_string(tolerance) + " <= 0) in: " + _fileName);
            }
            _input.iteration1 = iter1(nrOfIterations, tolerance);
        }

        void ConjugateGradientInversionInputCardReader::readDeltaAmplificationParameter(const nlohmann::json &jsonFile)
        {
            const std::string parameterDeltaAmplification = "DeltaAmplification";
            const std::string parameterStart = "start";
            const std::string parameterSlope = "slope";

            nlohmann::json daObject = io::ReadJsonHelper::tryGetParameterFromJson<nlohmann::json>(jsonFile, _fileName, parameterDeltaAmplification);

            double start = io::ReadJsonHelper::tryGetParameterFromJson<double>(daObject, _fileName, parameterStart);
            if(start <= 0)
            {
                throw std::invalid_argument("Invalid start step (" + std::to_string(start) + " <= 0) in: " + _fileName);
            }

            double slope = io::ReadJsonHelper::tryGetParameterFromJson<double>(daObject, _fileName, parameterSlope);
            if(slope <= 0)
            {
                throw std::invalid_argument("Invalid slope (" + std::to_string(slope) + " <= 0) in: " + _fileName);
            }
            _input.dAmplification = cgDeltaAmplification(start, slope);
        }
    }   // namespace inversionMethods
}   // namespace fwi
