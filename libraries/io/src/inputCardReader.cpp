#include "inputCardReader.h"
#include "log.h"

namespace fwi
{
    namespace io
    {
        nlohmann::json inputCardReader::readFile(const std::string &fileName)
        {
            std::ifstream in(fileName);
            if(!in.is_open())
            {
                throw std::invalid_argument("Could not open file at " + fileName);
            }

            nlohmann::json jsonFile;
            in >> jsonFile;
            return jsonFile;
        }

        std::string inputCardReader::removeLastSlash(const std::string &caseFolderWithSlash)
        {
            std::string caseFolder = caseFolderWithSlash;
            if(caseFolder[caseFolder.size() - 1] == '/')
            {
                caseFolder = caseFolder.substr(0, caseFolder.size() - 1);
            }
            return caseFolder;
        }

        std::string inputCardReader::getRunName(const std::string &caseFolder)
        {
            const std::string defaultRunName = "default";

            std::string runName = caseFolder;

            // Remove directory part
            const std::size_t idx = runName.find_last_of('/');
            if(idx != std::string::npos)
            {
                runName = runName.substr(idx + 1);
            }

            if(runName == ".")
            {
                runName = defaultRunName;
            }

            return runName;
        }
    }   // namespace io
}   // namespace fwi
