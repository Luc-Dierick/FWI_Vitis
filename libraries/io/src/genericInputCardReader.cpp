#include "genericInputCardReader.h"
#include "json.h"
#include <exception>
#include <iostream>
#include <string>

namespace fwi
{
    namespace io
    {
        genericInputCardReader::genericInputCardReader(const fwi::io::argumentReader inputArguments, const std::string &filename)
            : _fileName(filename)
        {
            std::string filePath = setFolders(inputArguments.dir);
            readJsonFile(filePath, inputArguments);
        }

        std::string genericInputCardReader::setFolders(const std::string &caseFolderWithSlash)
        {
            const std::string stringInputFolder = "/input/";
            const std::string stringOutputFolder = "/output/";

            std::string caseFolder = removeLastSlash(caseFolderWithSlash);
            std::string runName = getRunName(caseFolder);

            _input.caseFolder = caseFolder;
            _input.runName = runName;
            _input.inputFolder = caseFolder + stringInputFolder;
            _input.outputLocation = caseFolder + stringOutputFolder;

            std::string filePath = caseFolder + stringInputFolder + _fileName;
            return filePath;
        }

        void genericInputCardReader::readJsonFile(const std::string &filePath, const fwi::io::argumentReader inputArguments)
        {
            nlohmann::json jsonFile = readFile(filePath);

            readC0Parameter(jsonFile);
            readFreqParameter(jsonFile);
            readReservoirParameter(jsonFile);
            readSourcesParameter(jsonFile);
            readReceiversParameter(jsonFile);
            readGridOriginalParameter(jsonFile);
            readGridParameter(jsonFile);
            readNSourcesParameter(jsonFile);
            readNReceiversParameter(jsonFile);

            const std::string parameterFileName = "fileName";
            _input.fileName = ReadJsonHelper::tryGetParameterFromJson<std::string>(jsonFile, _fileName, parameterFileName);

            const std::string parameterForward = "forward";
            _input.forward = (!inputArguments.forward.empty()) ? inputArguments.forward : ReadJsonHelper::tryGetParameterFromJson<std::string>(jsonFile, _fileName, parameterForward);

            const std::string parameterInversion = "inversion";
            _input.inversion = (!inputArguments.inversion.empty()) ? inputArguments.inversion : ReadJsonHelper::tryGetParameterFromJson<std::string>(jsonFile, _fileName, parameterInversion);

            const std::string parameterVerbose = "verbosity";

            _input.verbose = ReadJsonHelper::tryGetParameterFromJson<bool>(jsonFile, _fileName, parameterVerbose);
        }

        void genericInputCardReader::readXZParametersFromJsonObject(const nlohmann::json &jsonFile, const std::string &fileName, std::array<double, 2> &array)
        {
            const std::string parameterX = "x";
            double xValue = ReadJsonHelper::tryGetParameterFromJson<double>(jsonFile, fileName, parameterX);

            const std::string parameterZ = "z";
            double zValue = ReadJsonHelper::tryGetParameterFromJson<double>(jsonFile, fileName, parameterZ);

            array = {xValue, zValue};
        }

        void genericInputCardReader::readXZParametersFromJsonObject(const nlohmann::json &jsonFile, const std::string &fileName, std::array<int, 2> &array)
        {
            const std::string parameterX = "x";
            int xValue = ReadJsonHelper::tryGetParameterFromJson<int>(jsonFile, fileName, parameterX);

            const std::string parameterZ = "z";
            int zValue = ReadJsonHelper::tryGetParameterFromJson<int>(jsonFile, fileName, parameterZ);

            array = {xValue, zValue};
        }

        void genericInputCardReader::readGrid(const nlohmann::json &jsonFile, const std::string &parameterNGrid, std::array<int, 2> &nGridArray)
        {
            nlohmann::json nGridOriginalJsonObject = ReadJsonHelper::tryGetParameterFromJson<nlohmann::json>(jsonFile, _fileName, parameterNGrid);
            readXZParametersFromJsonObject(nGridOriginalJsonObject, _fileName, nGridArray);

            if(nGridArray[0] <= 3 || nGridArray[1] <= 3)
            {
                throw std::invalid_argument("Grid size of " + parameterNGrid + " (" + std::to_string(nGridArray[0]) + "," + std::to_string(nGridArray[1]) +
                                            ") is invalid because (x > 3, z > 3) is required in  " + _fileName);
            }
        }

        void genericInputCardReader::readDevice(const nlohmann::json &jsonFile, const std::string &parameterTopLeft, const std::string &parameterBottomRight,
            std::array<double, 2> &topLeftArray, std::array<double, 2> &bottemRightArray)
        {
            nlohmann::json topLeftJsonObject = ReadJsonHelper::tryGetParameterFromJson<nlohmann::json>(jsonFile, _fileName, parameterTopLeft);
            readXZParametersFromJsonObject(topLeftJsonObject, _fileName, topLeftArray);

            nlohmann::json bottomRightJsonObject = ReadJsonHelper::tryGetParameterFromJson<nlohmann::json>(jsonFile, _fileName, parameterBottomRight);
            readXZParametersFromJsonObject(bottomRightJsonObject, _fileName, bottemRightArray);

            if(topLeftArray[0] > bottemRightArray[0] || topLeftArray[1] > bottemRightArray[1])
            {
                throw std::invalid_argument("Invalid ranges for " + parameterTopLeft + " and " + parameterBottomRight + "(" + std::to_string(topLeftArray[0]) +
                                            " > " + std::to_string(bottemRightArray[0]) + "(" + std::to_string(topLeftArray[1]) + " > " +
                                            std::to_string(bottemRightArray[1]) + " in: " + _fileName);
            }
        }

        void genericInputCardReader::readC0Parameter(const nlohmann::json &jsonFile)
        {
            const std::string parameterC0 = "c_0";
            double c0 = ReadJsonHelper::tryGetParameterFromJson<double>(jsonFile, _fileName, parameterC0);
            if(c0 <= 0)
            {
                throw std::invalid_argument("Invalid value for c0 (" + std::to_string(c0) + " <= 0) in: " + _fileName);
            }
            _input.c0 = c0;
        }

        void genericInputCardReader::readFreqParameter(const nlohmann::json &jsonFile)
        {
            const std::string parameterFreq = "Freq";
            const std::string parameterFreqMin = "min";
            const std::string parameterFreqMax = "max";
            const std::string parameterFreqNTotal = "nTotal";

            nlohmann::json freqJsonObject = ReadJsonHelper::tryGetParameterFromJson<nlohmann::json>(jsonFile, _fileName, parameterFreq);

            double min = ReadJsonHelper::tryGetParameterFromJson<double>(freqJsonObject, _fileName, parameterFreqMin);
            double max = ReadJsonHelper::tryGetParameterFromJson<double>(freqJsonObject, _fileName, parameterFreqMax);
            if(min >= max || min <= 0)
            {
                throw std::invalid_argument("Invalid ranges for frequenties (" + std::to_string(min) + " >= " + std::to_string(max) + ") (" +
                                            std::to_string(min) + " <= 0) in: " + _fileName);
            }

            double nTotal = ReadJsonHelper::tryGetParameterFromJson<double>(freqJsonObject, _fileName, parameterFreqNTotal);
            if(nTotal <= 1)
            {
                throw std::invalid_argument("Invalid number of frequenties (" + parameterFreqNTotal + " <= " + std::to_string(nTotal) + ") in: " + _fileName);
            }
            _input.freq = core::freqInfo(min, max, nTotal);
        }

        void genericInputCardReader::readReservoirParameter(const nlohmann::json &jsonFile)
        {
            const std::string parameterReservoirTopLeft = "reservoirTopLeft";
            const std::string parameterReservoirBottomRight = "reservoirBottomRight";

            std::array<double, 2> topLeftArray;
            std::array<double, 2> bottemRightArray;

            readDevice(jsonFile, parameterReservoirTopLeft, parameterReservoirBottomRight, topLeftArray, bottemRightArray);

            _input.reservoirTopLeftCornerInM = topLeftArray;
            _input.reservoirBottomRightCornerInM = bottemRightArray;
        }

        void genericInputCardReader::readSourcesParameter(const nlohmann::json &jsonFile)
        {
            const std::string parameterSourcesTopLeft = "sourcesTopLeft";
            const std::string parameterSourcesBottomRight = "sourcesBottomRight";

            std::array<double, 2> topLeftArray;
            std::array<double, 2> bottemRightArray;

            readDevice(jsonFile, parameterSourcesTopLeft, parameterSourcesBottomRight, topLeftArray, bottemRightArray);

            _input.sourcesTopLeftCornerInM = topLeftArray;
            _input.sourcesBottomRightCornerInM = bottemRightArray;
        }

        void genericInputCardReader::readReceiversParameter(const nlohmann::json &jsonFile)
        {
            const std::string parameterReceiversTopLeft = "receiversTopLeft";
            const std::string parameterReceiversBottomRight = "receiversBottomRight";

            std::array<double, 2> topLeftArray;
            std::array<double, 2> bottemRightArray;

            readDevice(jsonFile, parameterReceiversTopLeft, parameterReceiversBottomRight, topLeftArray, bottemRightArray);

            _input.receiversTopLeftCornerInM = topLeftArray;
            _input.receiversBottomRightCornerInM = bottemRightArray;
        }

        void genericInputCardReader::readGridOriginalParameter(const nlohmann::json &jsonFile)
        {
            const std::string parameterNGridOriginal = "ngrid_original";

            std::array<int, 2> nGridOriginalArray;
            readGrid(jsonFile, parameterNGridOriginal, nGridOriginalArray);

            _input.nGridOriginal = nGridOriginalArray;
        }

        void genericInputCardReader::readGridParameter(const nlohmann::json &jsonFile)
        {
            const std::string parameterNGrid = "ngrid";

            std::array<int, 2> nGridArray;
            readGrid(jsonFile, parameterNGrid, nGridArray);

            _input.nGrid = nGridArray;
        }

        void genericInputCardReader::readNSourcesParameter(const nlohmann::json &jsonFile)
        {
            const std::string parameterNSources = "nSources";
            int nSources = ReadJsonHelper::tryGetParameterFromJson<int>(jsonFile, _fileName, parameterNSources);
            if(nSources <= 1)
            {
                throw std::invalid_argument("Invalid number of core::Sources (" + std::to_string(nSources) + "<= 1) in: " + _fileName);
            }
            _input.nSources = nSources;
        }

        void genericInputCardReader::readNReceiversParameter(const nlohmann::json &jsonFile)
        {
            const std::string parameterNReceivers = "nReceivers";
            int nReceivers = ReadJsonHelper::tryGetParameterFromJson<int>(jsonFile, _fileName, parameterNReceivers);
            if(nReceivers <= 1)
            {
                throw std::invalid_argument("Invalid number of core::Receivers (" + std::to_string(nReceivers) + "<= 1) in: " + _fileName);
            }
            _input.nReceivers = nReceivers;
        }
    }   // namespace io
}   // namespace fwi
