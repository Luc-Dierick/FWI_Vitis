#pragma once

#include "ignoreStupidWarnings.h"
#include "grid2D.h"
#include <array>
#include <cassert>
#include <complex>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>

namespace fwi
{
    namespace core
    {
        template<class _V = double>
        class dataGrid2D
        {
        private:
            const grid2D &_grid;
            std::vector<_V> _data;

        public:
            explicit dataGrid2D(const grid2D &grid)
                : _grid(grid),
                  _data(std::vector<_V>(getNumberOfGridPoints()))
            {

            }

            dataGrid2D(const dataGrid2D<_V> &rhs)
                : dataGrid2D(rhs.getGrid())
            {
                assert(getGrid() == rhs.getGrid());
                const std::vector<_V> &rhsData = rhs.getData();
                *this = rhsData;
            }

            /**
             * This constructor enables static casting of dataGrids of different types,
             * if _W is castable to _V.
             */
            template <class _W = _V>
            dataGrid2D(const dataGrid2D<_W> &rhs) : dataGrid2D(rhs.getGrid())
            {
                const std::vector<_W> &rhsData = rhs.getData();
                *this = rhsData;
            }
            ~dataGrid2D() {}

            const std::vector<_V> &getData() const { return _data; }
            void setData(int i, _V input) {_data[i] = input;}

            void setValueAtIndex(const _V value, const int index) { _data[index] = value; }
            void addValueAtIndex(const _V value, const int index) { _data[index] += value; }

            const grid2D &getGrid() const { return _grid; }
            int getNumberOfGridPoints() const { return _grid.getNumberOfGridPoints(); }
            double getCellVolume() const { return _grid.getCellVolume(); }

            void zero()
            {
                for(int i = 0; i < getNumberOfGridPoints(); ++i)
                {
                    _data[i] = 0;
                }
            }
            void square()
            {
                for(int i = 0; i < getNumberOfGridPoints(); i++)
                {
                    _data[i] *= _data[i];
                }
            }

            void sqrt()
            {
                for(int i = 0; i < getNumberOfGridPoints(); i++)
                {
                    _data[i] = std::sqrt(_data[i]);
                }
            }

            void reciprocal()
            {
                for(int i = 0; i < getNumberOfGridPoints(); i++)
                {
                    if(_data[i] == 0.0)
                    {
                        throw std::overflow_error("reciprocal devides by zero");
                    }
                    _data[i] = 1.0 / _data[i];
                }
            }

            void conjugate();

            void random();
            void randomSaurabh();

            double norm() const
            {
                return std::sqrt(innerProduct(*this));
            }

            double relNorm() const
            {
                return std::sqrt(innerProduct(*this) / getNumberOfGridPoints());
            }

            _V summation() const
            {
                _V result = 0.0;
                for(int i = 0; i < getNumberOfGridPoints(); i++)
                {
                    result += _data[i];
                }
                return result;
            }

            double innerProduct(const dataGrid2D<_V> &rhs) const
            {
                assert(getGrid() == rhs.getGrid());

                double prod = 0.0;
                const std::vector<_V> &rhsData = rhs.getData();
                for(int i = 0; i < getNumberOfGridPoints(); i++)
                {
                    prod += std::real(_data[i] * std::conj(rhsData[i]));
                }
                return prod;
            }

            template <class _W = _V>
            _V dotProduct(const dataGrid2D<_W> &rhs) const
            {
                assert(getGrid() == rhs.getGrid());

                _V sum = 0;
                const std::vector<_W> &rhsData = rhs.getData();
                for(int i = 0; i < getNumberOfGridPoints(); i++)
                {
                    sum += _data[i] * rhsData[i];
                }
                return sum;
            }

            void gradient(std::vector<dataGrid2D<_V>> &gradientField) const
            {
                const std::array<int, 2> &nx = getGrid().getGridDimensions();
                const std::array<double, 2> &dx = getGrid().getCellDimensions();

                for(int i = 0; i < nx[1]; i++)
                {
                    for(int j = 0; j < nx[0]; j++)
                    {
                        int index = i * nx[0] + j;

                        // direction 1 dx
                        _V gradientDx;
                        if(j == 0)
                        {
                            gradientDx = (_data[i * nx[0] + j + 2] - 4 * _data[i * nx[0] + j + 1] + 3 * _data[i * nx[0] + j]) / (-2.0 * dx[0]);
                        }
                        else if(j == nx[0] - 1)
                        {
                            gradientDx = (_data[i * nx[0] + j - 2] - 4 * _data[i * nx[0] + j - 1] + 3 * _data[i * nx[0] + j]) / (2.0 * dx[0]);
                        }
                        else
                        {
                            gradientDx = (_data[i * nx[0] + j + 1] - _data[i * nx[0] + j - 1]) / (2.0 * dx[0]);
                        }
                        gradientField[0].setValueAtIndex(gradientDx, index);

                        // direction 2 dz
                        _V gradientDz;
                        if(i == 0)
                        {
                            gradientDz = (_data[(i + 2) * nx[0] + j] - 4 * _data[(i + 1) * nx[0] + j] + 3 * _data[i * nx[0] + j]) / (-2.0 * dx[1]);
                        }
                        else if(i == nx[1] - 1)
                        {
                            gradientDz = (_data[(i - 2) * nx[0] + j] - 4 * _data[(i - 1) * nx[0] + j] + 3 * _data[i * nx[0] + j]) / (2.0 * dx[1]);
                        }
                        else
                        {
                            gradientDz = (_data[(i + 1) * nx[0] + j] - _data[(i - 1) * nx[0] + j]) / (2.0 * dx[1]);
                        }
                        gradientField[1].setValueAtIndex(gradientDz, index);
                    }
                }
            }

            dataGrid2D<double> getRealPart() const
            {
                dataGrid2D<double> result(getGrid());

                std::vector<double> realData(getNumberOfGridPoints(), 0.0);
                for(int i = 0; i < getNumberOfGridPoints(); i++)
                {
                    realData[i] = std::real(_data[i]);
                }
                result = realData;
                return result;
            }

            // TODO: Implement these functions in a generic way, for now
            // kept the manual implementations to maintain backward compatibility.
            void toFile(const std::string &fileName) const;
            void fromFile(const std::string &fileName);


            // Operators
            dataGrid2D<_V>& operator=(const dataGrid2D<_V> &rhs)
            {
                if(this == &rhs)
                {
                    throw std::logic_error("Assign operator with itself");
                }

                assert(getGrid() == rhs.getGrid());
                const std::vector<_V> &rhsData = rhs.getData();
                *this = rhsData;
                return *this;
            }

            template<class _W = _V>
            dataGrid2D<_V>& operator=(const std::vector<_W> &rhs)
            {
                assert(getNumberOfGridPoints() == (int)rhs.size());
                for(int i = 0; i < getNumberOfGridPoints(); i++)
                {
                    _data[i] = rhs[i];
                }
                return *this;
            }

            template<class _W = _V>
            dataGrid2D<_V>& operator=(const _W rhs)
            {
                for(int i = 0; i < getNumberOfGridPoints(); i++)
                {
                    _data[i] = rhs;
                }
                return *this;
            }

            template<class _W = _V>
            dataGrid2D<_V> &operator+=(const dataGrid2D<_W> &rhs)
            {
                assert(getGrid() == rhs.getGrid());
                auto &rhsData = rhs.getData();
                *this += rhsData;
                return *this;
            }

            template<class _W = _V>
            dataGrid2D<_V> &operator+=(const std::vector<_W> &rhs)
            {
                assert(getNumberOfGridPoints() == (int)rhs.size());
                for(int i = 0; i < getNumberOfGridPoints(); i++)
                {
                    _data[i] += rhs[i];
                }

                return *this;
            }

            template<class _W = _V>
            dataGrid2D<_V> &operator+=(const _W rhs)
            {
                for(int i = 0; i < getNumberOfGridPoints(); i++)
                {
                    _data[i] += rhs;
                }

                return *this;
            }

            template<class _W = _V>
            dataGrid2D<_V> &operator-=(const dataGrid2D<_W> &rhs)
            {
                assert(getGrid() == rhs.getGrid());
                const auto &rhsData = rhs.getData();
                *this -= rhsData;
                return *this;
            }

            template<class _W = _V>
            dataGrid2D<_V> &operator-=(const std::vector<_W> &rhs)
            {
                assert(getNumberOfGridPoints() == (int)rhs.size());
                for(int i = 0; i < getNumberOfGridPoints(); i++)
                {
                    _data[i] -= rhs[i];
                }

                return *this;
            }

            dataGrid2D<_V> &operator-=(const _V rhs)
            {
                for(int i = 0; i < getNumberOfGridPoints(); i++)
                {
                    _data[i] -= rhs;
                }

                return *this;
            }

            template<class _W = _V>
            dataGrid2D<_V> &operator*=(const dataGrid2D<_W> &rhs)
            {
                assert(getGrid() == rhs.getGrid());
                const std::vector<_W> &rhsData = rhs.getData();
                for(int i = 0; i < getNumberOfGridPoints(); i++)
                {
                    _data[i] *= rhsData[i];
                }
                return *this;
            }

            template<class _W = _V>
            dataGrid2D<_V> &operator*=(const std::vector<_W> &rhs)
            {
                assert(getNumberOfGridPoints() == (int)rhs.size());
                for(int i = 0; i < getNumberOfGridPoints(); i++)
                {
                    _data[i] *= rhs[i];
                }

                return *this;
            }

            template<class _W = _V>
            dataGrid2D<_V> &operator*=(const _W rhs)
            {
                for(int i = 0; i < getNumberOfGridPoints(); i++)
                {
                    _data[i] *= rhs;
                }

                return *this;
            }

            template<class _W = _V>
            dataGrid2D<_V> &operator/=(const dataGrid2D<_W> &rhs)
            {
                const auto &rhsData = rhs.getData();
                *this /= rhsData;
                return *this;
            }

            template<class _W = _V>
            dataGrid2D<_V> &operator/=(const std::vector<_W> &rhs)
            {
                assert(getNumberOfGridPoints() == (int)rhs.size());
                for(int i = 0; i < getNumberOfGridPoints(); i++)
                {
                    if(rhs[i] == 0.0)
                    {
                        throw std::overflow_error("Operator devides by zero");
                    }
                    _data[i] /= rhs[i];
                }

                return *this;
            }

            template<class _W = _V>
            dataGrid2D<_V> &operator/=(const _W rhs)
            {
                if(rhs == 0.0)
                {
                    throw std::overflow_error("Operator devides by zero");
                }

                for(int i = 0; i < getNumberOfGridPoints(); i++)
                {
                    _data[i] /= rhs;
                }

                return *this;
            }

        }; // class declaration/definition

        template <class _V>
        inline double innerProduct(const dataGrid2D<_V> &t1, const dataGrid2D<_V> &t2)
        {
            return t1.innerProduct(t2);
        }

        template <class _V, class _W>
        inline _V dotProduct(const dataGrid2D<_V> &t1, const dataGrid2D<_W> &t2)
        {
            return t1.dotProduct(t2);
        }

        template <class _V, class _W>
        inline dataGrid2D<_V> operator+(const dataGrid2D<_V> &t1, const _W &t2)
        {
            dataGrid2D t3(t1);
            t3 += t2;
            return t3;
        }

        template <class _V>
        inline dataGrid2D<_V> operator-(const dataGrid2D<_V> &t1, const dataGrid2D<_V> &t2)
        {
            dataGrid2D t3(t1);
            t3 -= t2;
            return t3;
        }

        template <class _V, class _W>
        inline dataGrid2D<_V> operator*(const dataGrid2D<_V> &t1, const dataGrid2D<_W> &t2)
        {
            dataGrid2D<_V> t3(t1);
            t3 *= t2;
            return t3;
        }

        template <class _V>
        inline dataGrid2D<_V> operator*(const dataGrid2D<_V> &t1, const _V t2)
        {
            dataGrid2D<_V> t3(t1);
            t3 *= t2;
            return t3;
        }
        template <class _V>
        inline dataGrid2D<_V> operator*(const _V t1, const dataGrid2D<_V> &t2) { return t2 * t1; }

        template <class _V>
        inline dataGrid2D<_V> operator/(const dataGrid2D<_V> &t1, const dataGrid2D<_V> t2)
        {
            dataGrid2D t3(t1);
            t3 /= t2;
            return t3;
        }

        template <class _V, class _W>
        inline dataGrid2D<_V> operator/(const dataGrid2D<_V> &t1, const _W t2)
        {
            dataGrid2D t3(t1);
            t3 /= t2;
            return t3;
        }
    }   // namespace core
}   // namespace fwi
