#pragma once

#include "evolutionInversionInput.h"
#include "inputCardReader.h"
#include <ReadJsonHelper.h>
#include <fstream>
#include <sstream>
#include <string>

namespace fwi
{
    namespace inversionMethods
    {
        class EvolutionInversionInputCardReader : public io::inputCardReader
        {
        public:
            EvolutionInversionInputCardReader(const std::string &caseFolder);
            const EvolutionInversionInput getInput() const { return _input; }

        private:
            EvolutionInversionInput _input;

            const std::string _fileName = "EvolutionInversionInput.json";
            void readJsonFile(const std::string &filePath);
        };
    }   // namespace inversionMethods
}   // namespace fwi
