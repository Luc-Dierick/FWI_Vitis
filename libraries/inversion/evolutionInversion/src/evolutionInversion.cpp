#include "evolutionInversion.h"
#include "CommonVectorOperations.h"
#include "log.h"
#include "progressBar.h"

using fwi::core::operator-;

namespace fwi
{
    namespace inversionMethods
    {
        EvolutionInversion::EvolutionInversion(
            const core::CostFunctionCalculator &costCalculator, forwardModels::ForwardModelInterface *forwardModel, const EvolutionInversionInput &eiInput)
            : _forwardModel(forwardModel)
            , _costCalculator(costCalculator)
            , _eiInput(eiInput)
            , _grid(forwardModel->getGrid())
            , _source(forwardModel->getSource())
            , _receiver(forwardModel->getReceiver())
            , _freq(forwardModel->getFreq())
        {
        }

        core::dataGrid2D<double> EvolutionInversion::reconstruct(const std::vector<std::complex<double>> &pData, io::genericInput gInput)
        {
            io::progressBar bar(_eiInput.nGenerations * _eiInput.nChildrenPerGeneration);

            double favouriteChildResSq, childResSq, preParentResSq;
            double mutationRate = 0.05;

            std::default_random_engine generator;
            std::normal_distribution<double> distribution(0.0, mutationRate);

            // Create initial guess, generation 0, Adam
            core::dataGrid2D<double> parent(_grid);
            parent.randomSaurabh();

            auto pDataEst = _forwardModel->calculatePressureField(parent);
            double parentResSq = core::l2NormSquared(pData - pDataEst);
            preParentResSq = parentResSq;
            L_(io::linfo) << "Parent Res | Gen | Mutation Rate | Convergence State" << std::endl;
            std::cerr << "\n";

            std::ofstream residualLogFile = openResidualLogFile(gInput);

            bool isConverged = false;

            // main loop// Looping through the generations
            int counter = 1;
            for(int it = 0; it < _eiInput.nGenerations; it++)
            {
                core::dataGrid2D<double> favouriteChild = parent;   // This is the best child so far
                favouriteChildResSq = parentResSq;
                // start the inner loop// Generating children (currently not parallel, only 1 child at a time is stored)
                for(int it1 = 0; it1 < _eiInput.nChildrenPerGeneration; it1++)
                {
                    auto child = createRandomChild(parent, generator, distribution);
                    pDataEst = _forwardModel->calculatePressureField(child);
                    childResSq = core::l2NormSquared(pData - pDataEst);

                    if(childResSq < favouriteChildResSq)
                    {
                        favouriteChild = child;
                        favouriteChildResSq = childResSq;
                    }

                    residualLogFile << std::setprecision(17) << favouriteChildResSq << "," << counter << std::endl;
                    counter++;   // store the residual value in the residual log
                    ++bar;
                }

                isConverged = (favouriteChildResSq < _eiInput.toleranceOuter);

                std::string converganceMessage = isConverged ? "Converged" : "Not Converged";

                L_(io::linfo) << std::setprecision(7) << parentResSq << "   |  " << it << " | " << mutationRate << " | " << converganceMessage << std::endl;

                if(favouriteChildResSq == parentResSq && preParentResSq == favouriteChildResSq)
                {
                    mutationRate /= 1.1;
                    distribution = std::normal_distribution<double>(0.0, mutationRate);
                }

                parent = favouriteChild;
                preParentResSq = parentResSq;
                parentResSq = favouriteChildResSq;
            }

            residualLogFile.close();   // close the residual.log file

            return parent;
        }

        core::dataGrid2D<double> EvolutionInversion::createRandomChild(
            const core::dataGrid2D<double> &parent, std::default_random_engine &generator, std::normal_distribution<double> &distribution) const
        {
            core::dataGrid2D<double> child(parent.getGrid());
            auto parentData = parent.getData();
            for(int i = 0; i < child.getGrid().getNumberOfGridPoints(); i++)
            {
                auto newVal = parentData[i] + distribution(generator);
                if(newVal > 0.18)
                {
                    newVal -= 0.18;   // If larger than 0.18, loops back from 0
                }
                if(newVal < 0.00)
                {
                    newVal += 0.18;   // If negative, loops back from 0.18
                }

                child.setValueAtIndex(newVal, i);
            }

            return child;
        }

        std::ofstream EvolutionInversion::openResidualLogFile(io::genericInput &gInput)
        {
            std::string filePath = gInput.outputLocation + gInput.runName + "Residual" + ".log";

            std::ofstream residualLogFile;
            residualLogFile.open(filePath, std::ios::out | std::ios::trunc);
            if(!residualLogFile)
            {
                throw std::invalid_argument("Unable to store residuals from file : " + filePath);
            }

            return residualLogFile;
        }
    }   // namespace inversionMethods
}   // namespace fwi
