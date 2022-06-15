#pragma once

#include "PmlWidthFactor.h"
#include "SourceParameter.h"

#include <string>

namespace fwi
{
    namespace forwardModels
    {
        enum BoundaryConditionType
        {
            PML,             // Perfectly Matched Layers
            FirstOrderABC,   // First Order Absorbing Boundary Layers
            SecondOrderABC   // Second Order Absorbing Boundary Layers
        };

        struct finiteDifferenceForwardModelInput
        {
            PMLWidthFactor pmlWidthFactor;
            SourceParameter sourceParameter;
            BoundaryConditionType boundaryConditionType;
        };
    }   // namespace forwardModels
}   // namespace fwi
