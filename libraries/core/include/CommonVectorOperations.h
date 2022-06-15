#pragma once

#include "ignoreStupidWarnings.h"
#include <algorithm>
#include <cassert>
#include <complex>
#include <numeric>
#include <vector>

namespace fwi
{
    namespace core
    {
        /**
         * @brief Computes the l2 norm squared of a std::vector<std::complex<double>>
         * @param data is a reference to the vector to use to calculate its l2 norm squared
         * @return calculated l2 norm squared
         */
        inline double l2NormSquared(const std::vector<std::complex<double>> &data)
        {
            double initialValue = 0.0;
            auto elementOperation = [](double accumulatedValue, std::complex<double> vectorElement) { return (accumulatedValue + std::norm(vectorElement)); };
            return std::accumulate(data.begin(), data.end(), initialValue, elementOperation);
        }

        /**
         * @brief Defines substraction operator between std::vector<std::complex<double>> of equal size, it performs element wise substraction
         * @param operandA constant reference to the first operand
         * @param operandB constant reference to the second operand
         * @return a vector where its components are calculated as vector[i] = operandA[i] - operandB[i]
         */
        inline std::vector<std::complex<double>> operator-(const std::vector<std::complex<double>> &operandA, const std::vector<std::complex<double>> &operandB)
        {
            assert(operandA.size() == operandB.size() && "Substraction is not defined for vectors of different size");
            std::vector<std::complex<double>> result(operandA.size());
            std::transform(
                operandA.begin(), operandA.end(), operandB.begin(), result.begin(), [](std::complex<double> x, std::complex<double> y) { return x - y; });
            return result;
        }

        /**
         * @brief Defines addition operator between std::vector<std::complex<double>> of equal size, it performs element wise addition
         * @param operandA constant reference to the first operand
         * @param operandB constant reference to the second operand
         * @return a vector where its components are calculated as vector[i] = operandA[i] + operandB[i]
         */
        inline std::vector<std::complex<double>> operator+(const std::vector<std::complex<double>> &operandA, const std::vector<std::complex<double>> &operandB)
        {
            assert(operandA.size() == operandB.size() && "Addition is not defined for vectors of different size");
            std::vector<std::complex<double>> result(operandA.size());
            std::transform(
                operandA.begin(), operandA.end(), operandB.begin(), result.begin(), [](std::complex<double> x, std::complex<double> y) { return x + y; });
            return result;
        }

    }   // namespace core
}   // namespace fwi
