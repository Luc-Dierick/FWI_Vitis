#ifdef SYCL
#include "FiniteDifferenceForwardModel.h"
#include "FiniteDifferenceForwardModelInput.h"
#include "ForwardModelInterface.h"
#include "greensFunctions.h"
#include "greensSerial.h"

namespace fwi
{
    namespace forwardModels
    {
        class FiniteDifferenceForwardModelDataParallelCPP : public FiniteDifferenceForwardModel
        {
        public:
            FiniteDifferenceForwardModelDataParallelCPP(const core::grid2D &grid, const core::Sources &source, const core::Receivers &receiver,
                const core::FrequenciesGroup &freq, const finiteDifferenceForwardModelInput &fmInput);

            ~FiniteDifferenceForwardModelDataParallelCPP();

            // Overriding ForwardModel
            std::vector<std::complex<double>> calculatePressureField(const core::dataGrid2D<double> &chiEst);
            void calculateKappa();
        };
        
    }   // namespace forwardModels

}   // namespace fwi
#endif