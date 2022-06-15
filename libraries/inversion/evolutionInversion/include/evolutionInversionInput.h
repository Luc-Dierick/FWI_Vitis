#pragma once

namespace fwi
{
    namespace inversionMethods
    {
        struct EvolutionInversionInput
        {
            double toleranceOuter;
            int nGenerations;
            int nChildrenPerGeneration;
        };
    }   // namespace inversionMethods
}   // namespace fwi
