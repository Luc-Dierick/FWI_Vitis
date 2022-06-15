#pragma once

#include "ReadJsonHelper.h"
#include "gradientDescentInversionInput.h"
#include "inputCardReader.h"
#include <fstream>
#include <sstream>
#include <string>

namespace fwi
{
    namespace inversionMethods
    {
        class gradientDescentInversionInputCardReader : public io::inputCardReader
        {
        private:
            gradientDescentInversionInput _input;

            const std::string _fileName;
            void readJsonFile(const std::string &filePath);

        public:
            gradientDescentInversionInputCardReader(const std::string &caseFolder, const std::string &filename = "GradientDescentInversionInput.json");
            const gradientDescentInversionInput getInput() const { return _input; }
        };
    }   // namespace inversionMethods
}   // namespace fwi
