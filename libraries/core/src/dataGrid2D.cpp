#include "dataGrid2D.h"
#include "log.h"

namespace fwi
{
    namespace core
    {
        // Complex numbers
        template<>
        void dataGrid2D<std::complex<double>>::toFile(const std::string &filePath) const
        {
            std::ofstream file;
            file.open(filePath, std::ios::out | std::ios::trunc);
            if(file.is_open())
            {
                for(int i = 0; i < getNumberOfGridPoints(); i++)
                {
                    file << std::showpos << std::setprecision(17) << "(" << real(_data[i]) << imag(_data[i]) << "j)";
                }
                file.close();
            }
            else
            {
                throw std::runtime_error("Unable to write DataGrid2D to file: " + filePath);
            }
        }

        template<>
        void dataGrid2D<std::complex<double>>::fromFile(const std::string &filePath)
        {
            std::ifstream file(filePath, std::ios::in);
            if(file.is_open())
            {
                for(int i = 0; i < getNumberOfGridPoints(); i++)
                {
                    file >> _data[i];
                }
                file.close();
            }
            else
            {
                throw std::runtime_error("Unable to load DataGrid2D from file: " + filePath);
            }
        }

        template<>
        void dataGrid2D<std::complex<double>>::random()
        {
            for(int i = 0; i < getNumberOfGridPoints(); i++)
            {
                _data[i] = std::complex<double>(double(std::rand()) / double(RAND_MAX), double(std::rand()) / double(RAND_MAX));
            }
        }

        // Double
        template<>
        void dataGrid2D<double>::toFile(const std::string &filePath) const
        {
            std::ofstream file;
            file.open(filePath, std::ios::out | std::ios::trunc);
            if(file.is_open())
            {
                for(int i = 0; i < getNumberOfGridPoints(); i++)
                {
                    file << std::setprecision(17) << _data[i] << std::endl;
                }
                file.close();
            }
            else
            {
                throw std::runtime_error("Unable to write dataGrid2D to file: " + filePath);
            }
        }

        template<>
        void dataGrid2D<double>::fromFile(const std::string &filePath)
        {
            std::ifstream file(filePath, std::ios::in);
            if(file.is_open())
            {
                for(int i = 0; i < getNumberOfGridPoints(); i++)
                {
                    file >> _data[i];
                }
                file.close();
            }
            else
            {
                throw std::runtime_error("Unable to load dataGrid2D from file: " + filePath);
            }
        }

        template<>
        void dataGrid2D<double>::random()
        {
            for(int i = 0; i < getGrid().getNumberOfGridPoints(); i++)
            {
                _data[i] = double(std::rand()) / double(RAND_MAX);
            }
        }

        template<>
        void dataGrid2D<double>::randomSaurabh()
        {
            for(int i = 0; i < getGrid().getNumberOfGridPoints(); i++)
            {
                int temp = rand() % 1800;
                _data[i] = (double)temp / 10000;
            }
        }

        template<>
        void dataGrid2D<double>::conjugate()
        {
            // Conjugate of a double is the identity function.
        }

        template<>
        void dataGrid2D<std::complex<double>>::conjugate()

        {
            for(int i = 0; i < getNumberOfGridPoints(); i++)
            {
                _data[i] = std::conj(_data[i]);
            }
        }
    }   // namespace core
}   // namespace fwi
