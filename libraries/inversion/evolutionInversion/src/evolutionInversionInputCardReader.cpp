#include "evolutionInversionInputCardReader.h"
#include "json.h"
#include <iostream>

namespace fwi
{
    namespace inversionMethods
    {
        EvolutionInversionInputCardReader::EvolutionInversionInputCardReader(const std::string &caseFolder)
            : io::inputCardReader()
        {
            static const std::string stringInputFolder = "/input/";
            std::string filePath = caseFolder + stringInputFolder + _fileName;

            readJsonFile(filePath);
        }

        void EvolutionInversionInputCardReader::readJsonFile(const std::string &filePath)
        {
            nlohmann::json jsonFile = readFile(filePath);

            static const std::string parameterGamma = "toleranceOuter";
            double toleranceOuter = io::ReadJsonHelper::tryGetParameterFromJson<double>(jsonFile, _fileName, parameterGamma);
            if(toleranceOuter <= 0)
            {
                throw std::invalid_argument("Invalid outer tolerance (" + std::to_string(toleranceOuter) + " <= 0) in: " + _fileName);
            }
            _input.toleranceOuter = toleranceOuter;

            static const std::string parameterInitialX = "nGenerations";
            int nrOfGenerations = io::ReadJsonHelper::tryGetParameterFromJson<int>(jsonFile, _fileName, parameterInitialX);
            if(nrOfGenerations <= 0)
            {
                throw std::invalid_argument("Invalid number of generations (" + std::to_string(nrOfGenerations) + " <= 0) in: " + _fileName);
            }
            _input.nGenerations = nrOfGenerations;

            static const std::string parameterStepSize = "nChildrenPerGeneration";
            int nrofChildrenPerGeneration = io::ReadJsonHelper::tryGetParameterFromJson<int>(jsonFile, _fileName, parameterStepSize);
            if(nrofChildrenPerGeneration <= 0)
            {
                throw std::invalid_argument(
                    "Invalid number of children per generation (" + std::to_string(nrofChildrenPerGeneration) + " <= 0) in: " + _fileName);
            }
            _input.nChildrenPerGeneration = nrofChildrenPerGeneration;
        }
    }   // namespace inversionMethods
}   // namespace fwi
