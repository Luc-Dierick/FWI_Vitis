#pragma once

#include <sstream>
#include <string>
#include <vector>

namespace fwi
{
    namespace io
    {
        class CSVReader
        {
        public:
            std::string const &operator[](std::size_t index) const;
            std::size_t size() const;
            void readNextRow(std::istream &str);

        private:
            std::vector<std::string> m_data;
        };

        inline std::istream &operator>>(std::istream &str, CSVReader &data)
        {
            data.readNextRow(str);
            return str;
        }
    }   // namespace io
}   // namespace fwi
