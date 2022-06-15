#pragma once

#include "CostFunctionCalculator.h"
#include "ForwardModelInterface.h"
#include "genericInput.h"
#include "gradientDescentInversionInput.h"
#include "inversionInterface.h"

namespace fwi
{
    namespace inversionMethods
    {
        class gradientDescentInversion : public inversionInterface
        {
        private:
            forwardModels::ForwardModelInterface *_forwardModel;
            const core::CostFunctionCalculator &_costCalculator;
            gradientDescentInversionInput _gdInput;

            const core::grid2D &_grid;
            const core::Sources &_source;
            const core::Receivers &_receiver;
            const core::FrequenciesGroup _freq;

            core::dataGrid2D<double> gradientDescent(core::dataGrid2D<double> chiEstimate, const std::vector<double> &dfdx, const double gamma);
            std::ofstream openResidualLogFile(io::genericInput &gInput);

        public:
            gradientDescentInversion(const core::CostFunctionCalculator &costCalculator, forwardModels::ForwardModelInterface *forwardModel,
                const gradientDescentInversionInput &gdInput);

            gradientDescentInversion(const gradientDescentInversion &) = delete;
            gradientDescentInversion &operator=(const gradientDescentInversion &) = delete;

            void logResidualResults(int iteration, double residual, bool isConverged);

            core::dataGrid2D<double> reconstruct(const std::vector<std::complex<double>> &pData, io::genericInput gInput);
            std::vector<double> differential(core::dataGrid2D<double> xi, const std::vector<std::complex<double>> &pData, double eta, double dxi);
            double determineGamma(const core::dataGrid2D<double> chiEstimatePrevious, const core::dataGrid2D<double> chiEstimateCurrent, std::vector<double> dFdxPrevious,
                std::vector<double> dFdx);
        };
    }   // namespace inversionMethods
}   // namespace fwi
