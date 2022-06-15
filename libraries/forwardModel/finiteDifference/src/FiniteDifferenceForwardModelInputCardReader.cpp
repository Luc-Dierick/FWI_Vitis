#include <iostream>
#include <map>

#include "FiniteDifferenceForwardModelInputCardReader.h"
#include "json.h"

namespace fwi
{
    namespace forwardModels
    {
        finiteDifferenceForwardModelInputCardReader::finiteDifferenceForwardModelInputCardReader(const std::string &caseFolder, const std::string &filename)
            : io::inputCardReader()
            , _fileName(filename)
        {
            const std::string stringInputFolder = "/input/";
            std::string filePath = caseFolder + stringInputFolder + _fileName;
            readJsonFile(filePath);
        }

        void finiteDifferenceForwardModelInputCardReader::readJsonFile(const std::string &filePath)
        {
            nlohmann::json jsonFile = readFile(filePath);
            readPMLWidthFactorParameters(jsonFile);
            readSourceParameters(jsonFile);
            readBoundaryConditionType(jsonFile);
        }

        void finiteDifferenceForwardModelInputCardReader::readPMLWidthFactorParameters(const nlohmann::json &jsonFile)
        {
            const std::string parameterPMLWidthFactor = "PMLWidthFactor";
            const std::string parameterX = "x";
            const std::string parameterZ = "z";
            nlohmann::json iterObject = io::ReadJsonHelper::tryGetParameterFromJson<nlohmann::json>(jsonFile, _fileName, parameterPMLWidthFactor);

            double x = io::ReadJsonHelper::tryGetParameterFromJson<double>(iterObject, _fileName, parameterX);
            double z = io::ReadJsonHelper::tryGetParameterFromJson<double>(iterObject, _fileName, parameterZ);

            _input.pmlWidthFactor = PMLWidthFactor(x, z);
        }

        void finiteDifferenceForwardModelInputCardReader::readSourceParameters(const nlohmann::json &jsonFile)
        {
            const std::string parameterSource = "SourceParameter";
            const std::string parameterHalfWidth = "r";
            const std::string parameterShape = "beta";

            nlohmann::json iterObject = io::ReadJsonHelper::tryGetParameterFromJson<nlohmann::json>(jsonFile, _fileName, parameterSource);

            double halfWidth = io::ReadJsonHelper::tryGetParameterFromJson<double>(iterObject, _fileName, parameterHalfWidth);
            double shape = io::ReadJsonHelper::tryGetParameterFromJson<double>(iterObject, _fileName, parameterShape);

            _input.sourceParameter = SourceParameter(halfWidth, shape);
        }

        void finiteDifferenceForwardModelInputCardReader::readBoundaryConditionType(const nlohmann::json &jsonFile)
        {
            const std::string boundaryConditionTypeArgument = "boundaryConditionType";
            std::string boundaryConditionType = io::ReadJsonHelper::tryGetParameterFromJson<std::string>(jsonFile, _fileName, boundaryConditionTypeArgument);

            std::map<std::string, BoundaryConditionType> boundaryConditionTypeMap{{"PML", BoundaryConditionType::PML},
                {"FirstOrderABC", BoundaryConditionType::FirstOrderABC}, {"SecondOrderABC", BoundaryConditionType::SecondOrderABC}};
            try
            {
                _input.boundaryConditionType = boundaryConditionTypeMap.at(boundaryConditionType);
            }
            catch(std::exception &e)
            {
                std::stringstream message;

                message << "Invalid input boundary condition in " + _fileName + ". The supported types are: ";
                auto delim = "";
                for(const auto &parameter : boundaryConditionTypeMap)
                {
                    message << delim << parameter.first;
                    delim = ", ";
                }
                message << ".";
                throw std::invalid_argument(message.str());
            }
        }
    }   // namespace forwardModels
}   // namespace fwi
