#ifdef SYCL
#include "CostFunctionCalculator.h"
#include "ForwardModelInterface.h"
#include "conjugateGradientInversion.h"
#include "conjugateGradientInversionInput.h"
#include "conjugateGradientInversionInputCardReader.h"
#include "genericInput.h"
#include "inversionInterface.h"
#include "log.h"
#include "regularization.h"
#include <fstream>
#include <sstream>
#include <CL/sycl.hpp>

// #include <CL/sycl/INTEL/fpga_extensions.hpp>

namespace fwi
{
    namespace inversionMethods
    {
        class conjugateGradientDataParallelCppInversion : public ConjugateGradientInversion
        {
        public:
            conjugateGradientDataParallelCppInversion(const core::CostFunctionCalculator &costCalculator, forwardModels::ForwardModelInterface *forwardModel,
                const ConjugateGradientInversionInput &invInput);

            ~conjugateGradientDataParallelCppInversion();

            // Overloading conjugateGradient parameters
            void getUpdateDirectionInformation(
                const std::vector<std::complex<double>> &residualVector, core::dataGrid2D<std::complex<double>> &kappaTimesResidual);

            std::vector<std::complex<double>> calculateKappaTimesZeta(const core::dataGrid2D<double> &zeta, std::vector<std::complex<double>> &kernel);

        protected:
        // SYCL attributes:
            sycl::queue Q{sycl::gpu_selector{}};
        };
    }   // namespace inversionMethods
}   // namespace fwi

#endif