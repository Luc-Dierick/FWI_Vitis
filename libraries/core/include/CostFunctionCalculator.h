#pragma once

#include "CommonVectorOperations.h"

namespace fwi
{
    namespace core
    {
        /**
         * @brief The CostFunctionCalculator class contains the target function to be optimized
         */
        class CostFunctionCalculator
        {
        public:
            /**
             * @brief The CostFunctionEnum enum Defines the possible cost functions that can be selected
             */
            enum class CostFunctionEnum
            {
                leastSquares
            };
            /**
             * @brief CostFunctionCalculator Class constructor
             * @param costFunctionSelected sets the function used to calculate the cost, if not defined it will default to CostFunctionEnum::leastSquares
             */
            explicit CostFunctionCalculator(CostFunctionEnum costFunctionSelected = CostFunctionEnum::leastSquares)
                : _costFunction(costFunctionSelected)
            {
            }

            /**
             * @brief calculateCost Calculates cost from current estimation using the defined functions
             * @param pData reference pressure field data
             * @param pDataEst current estimation of pressure field
             * @param eta
             * @return cost
             */
            double calculateCost(const std::vector<std::complex<double>> &pData, const std::vector<std::complex<double>> &pDataEst, double eta) const
            {   // TODO: Add switch case to change the calculation of the cost function according to the selected enum value (add this whenever we have more
                // than one option)
                return eta * l2NormSquared(pData - pDataEst);
            }

            /**
             * @brief getCostFunction Getter function to know the selected cost function
             * @return selected cost function
             */
            CostFunctionEnum getCostFunction() const { return _costFunction; }

        private:
            const CostFunctionEnum _costFunction;
        };
    }   // namespace core
}   // namespace fwi
