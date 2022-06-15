#pragma once

#include "ForwardModelInterface.h"
#include "freqInfo.h"
#include "frequenciesGroup.h"
#include "greensFunctions.h"
#include "greensSerial.h"

#include <gmock/gmock.h>

namespace fwi
{
    namespace forwardModels
    {
        class ForwardModelMock : public ForwardModelInterface
        {
        public:
            ForwardModelMock() {}

            MOCK_METHOD(const core::grid2D &, getGrid, (), (override));

            MOCK_METHOD(const core::Sources &, getSource, (), (override));
            MOCK_METHOD(const core::Receivers &, getReceiver, (), (override));
            MOCK_METHOD(const core::FrequenciesGroup &, getFreq, (), (override));
            MOCK_METHOD(const std::vector<core::dataGrid2D<std::complex<double>>> &, getKernel, (), (override));
            MOCK_METHOD(std::vector<std::complex<double>>, calculatePressureField, (const core::dataGrid2D<double> &chiEst), (override));

            MOCK_METHOD(void, calculatePTot, (const core::dataGrid2D<double> &chiEst), (override));

            MOCK_METHOD(void, calculateKappa, (), (override));
        };
    }   // namespace forwardModels
}   // namespace fwi
