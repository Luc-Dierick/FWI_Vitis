#pragma once

#include "dataGrid2D.h"
#include "frequenciesGroup.h"
#include "grid2D.h"
#include "log.h"
#include "receivers.h"
#include "sources.h"
#include <complex>

namespace fwi
{
    namespace forwardModels
    {
        /**
         * @brief Interface for ForwardModel classes
         */
        class ForwardModelInterface
        {
        public:
            ForwardModelInterface() = default;

            virtual ~ForwardModelInterface() = default;
            ///@{
            /** Getter functions for ForwardModel initialization parameters */
            virtual const core::grid2D &getGrid() = 0;
            virtual const core::Sources &getSource() = 0;
            virtual const core::Receivers &getReceiver() = 0;
            virtual const core::FrequenciesGroup &getFreq() = 0;
            virtual const std::vector<core::dataGrid2D<std::complex<double>>> &getKernel() = 0;

            ///@}
            /**
             * @brief calculatePData Calculates pressure field from media properties
             * @param chiEst reference to media properties
             * @return pressure field
             */
            virtual std::vector<std::complex<double>> calculatePressureField(const core::dataGrid2D<double> &chiEst) = 0;

            std::vector<std::complex<double>> kappa1D;
            // TODO: Evaluate the following functions for removal
            virtual void calculatePTot(const core::dataGrid2D<double> &chiEst) = 0;
            virtual void calculateKappa() = 0;
        };
    }   // namespace forwardModels
}   // namespace fwi
