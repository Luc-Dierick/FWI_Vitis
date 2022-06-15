#pragma once

#include "genericInput.h"
#include "inputCardReader.h"
#include "argumentReader.h"
#include <ReadJsonHelper.h>
#include <string>
#include <vector>

namespace fwi
{
    namespace io
    {
        class genericInputCardReader : public inputCardReader
        {
        public:
            genericInputCardReader(const fwi::io::argumentReader inputArguments, const std::string &filename = "GenericInput.json");
            genericInput getInput() const { return _input; }

        private:
            genericInput _input;

            const std::string _fileName;
            std::string setFolders(const std::string &caseFolderWithSlash);
            void readJsonFile(const std::string &filePath, const fwi::io::argumentReader inputArguments);

            void readXZParametersFromJsonObject(const nlohmann::json &jsonFile, const std::string &fileName, std::array<double, 2> &array);
            void readXZParametersFromJsonObject(const nlohmann::json &jsonFile, const std::string &fileName, std::array<int, 2> &array);

            void readGrid(const nlohmann::json &jsonFile, const std::string &parameterNGrid, std::array<int, 2> &nGridArray);
            void readDevice(const nlohmann::json &jsonFile, const std::string &parameterTopLeft, const std::string &parameterBottomRight,
                std::array<double, 2> &topLeftArray, std::array<double, 2> &bottemRightArray);

            void readC0Parameter(const nlohmann::json &jsonFile);
            void readFreqParameter(const nlohmann::json &jsonFile);
            void readReservoirParameter(const nlohmann::json &jsonFile);
            void readSourcesParameter(const nlohmann::json &jsonFile);
            void readReceiversParameter(const nlohmann::json &jsonFile);
            void readGridOriginalParameter(const nlohmann::json &jsonFile);
            void readGridParameter(const nlohmann::json &jsonFile);
            void readNSourcesParameter(const nlohmann::json &jsonFile);
            void readNReceiversParameter(const nlohmann::json &jsonFile);
        };
    }   // namespace io
}   // namespace fwi
