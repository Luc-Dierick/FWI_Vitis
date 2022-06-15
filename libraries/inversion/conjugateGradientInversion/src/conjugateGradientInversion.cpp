#include "conjugateGradientInversion.h"
#include "CommonVectorOperations.h"
#include "dataGrid2D.h"
#include "log.h"
#include "progressBar.h"
#include <iostream>
#include <memory>
#include <xcl2.hpp>

#define TAG_COMMAND 0
#define COMMAND_GETUPDATEDIRECTION 1
#define COMMAND_EXIT -99
#define TAG_SIZE 1
#define TAG_RESARRAY 2
#define TAG_RESULT 3

using fwi::core::operator-;

namespace fwi
{
    namespace inversionMethods
    {
        ConjugateGradientInversion::ConjugateGradientInversion(const core::CostFunctionCalculator &costCalculator,
            forwardModels::FiniteDifferenceForwardModel *forwardModel, const ConjugateGradientInversionInput &invInput)
            : _forwardModel(forwardModel)
            , _costCalculator(costCalculator)
            , _cgInput(invInput)
            , _grid(forwardModel->getGrid())
            , _sources(forwardModel->getSource())
            , _receivers(forwardModel->getReceiver())
            , _frequencies(forwardModel->getFreq())
            , _chiEstimate(_grid)
        {
        }

        core::dataGrid2D<double> ConjugateGradientInversion::reconstruct(const std::vector<std::complex<double>> &pData, io::genericInput gInput)
        {

            // Initialization of variables
            io::progressBar progressBar(_cgInput.n_max);
            const double eta = 1.0 / _costCalculator.calculateCost(pData, std::vector<std::complex<double>>(pData.size(), 0.0), 1.0);
            _chiEstimate.zero();
            std::ofstream residualLogFile = openResidualLogFile(gInput);
            bool isConverged = false;
            int counter = 1;

            // Initialize conjugate gradient parameters
            core::dataGrid2D gradientCurrent(_grid), gradientPrevious(_grid), zeta(_grid);
            double residualCurrent = 0.0, residualPrevious = 0.0, alpha = 0.0;
            RegularisationParameters regularisationCurrent(_grid);
            RegularisationParameters regularisationPrevious(_grid);
            regularisationPrevious.bSquared.zero();

            // Update contrast-function first time
            auto pDataEst = _forwardModel->calculatePressureField(_chiEstimate);
            std::vector<std::complex<double>> residualVector = pData - pDataEst;
            residualCurrent = _costCalculator.calculateCost(pData, pDataEst, eta);

            zeta = calculateUpdateDirection(residualVector, gradientCurrent, eta);
            alpha = calculateStepSize(zeta, residualVector);   // eq: optimalStepSizeCG

            _chiEstimate += alpha * zeta;   // eq: contrastUpdate

            gradientPrevious = gradientCurrent;
            residualPrevious = residualCurrent;

            std::cout << _cgInput.n_max <<std::endl;

    
            // main loop
            for(int it = 0; it < 1000; it++)
            {
//            	clock_t t_s = clock();
                // Calculate the pressure data from _chiEstimate
                auto pDataEst = _forwardModel->calculatePressureField(_chiEstimate); 
                residualVector = pData - pDataEst;

                // Check residual
                residualCurrent = _costCalculator.calculateCost(pData, pDataEst, eta);   // eq: errorFunc
                isConverged = (residualCurrent < _cgInput.iteration1.tolerance);
                logResidualResults(0, it, residualCurrent, counter, residualLogFile, isConverged);

                if((it > 0) && ((isConverged) || (std::abs(residualPrevious - residualCurrent) < _cgInput.iteration1.tolerance)))
                {
                    progressBar.setCounter(_cgInput.iteration1.n + progressBar.getCounter() - (progressBar.getCounter() % _cgInput.iteration1.n));
                    break;
                }

                // Initialize Regularisation parameters
                // Note: deltaAmplification decreases the step size for increasing iteration step
                double deltaAmplification = _cgInput.dAmplification.start / (_cgInput.dAmplification.slope * it + 1.0);

                // Start iteration with Regularisation
                calculateRegularisationParameters(regularisationPrevious, regularisationCurrent, deltaAmplification);
                zeta = calculateUpdateDirectionRegularisation(residualVector, gradientCurrent, gradientPrevious, eta, regularisationCurrent,
                    regularisationPrevious, zeta, residualPrevious);   // eq: updateDirectionsCG
                alpha = calculateStepSizeRegularisation(regularisationPrevious, regularisationCurrent, residualVector, eta, residualPrevious, zeta);

                _chiEstimate += alpha * zeta;   // eq: contrastUpdate

                // Save regularisation variables for next iteration
                gradientPrevious = gradientCurrent;
                residualPrevious = residualCurrent;
                _chiEstimate.gradient(regularisationCurrent.gradientChi);
                calculateRegularisationErrorFunctional(regularisationPrevious, regularisationCurrent);

                regularisationPrevious.deltaSquared = regularisationCurrent.deltaSquared;
                regularisationPrevious.bSquared = regularisationCurrent.bSquared;

                // Update counter
                counter++;
                progressBar++;
            }
            residualLogFile.close();

            return _chiEstimate;
        }

        std::ofstream ConjugateGradientInversion::openResidualLogFile(io::genericInput &gInput)
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

        core::dataGrid2D<double> ConjugateGradientInversion::calculateUpdateDirection(
            std::vector<std::complex<double>> &residualVector, core::dataGrid2D<double> &gradientCurrent, double eta)
        { // This function is never called... :) 
            core::dataGrid2D<std::complex<double>> kappaTimesResidual(
                _grid);   // eq: integrandForDiscreteK of README, KappaTimesResidual is the argument of Re()
            getUpdateDirectionInformation(residualVector, kappaTimesResidual);
            gradientCurrent = eta * kappaTimesResidual.getRealPart();

            return gradientCurrent;
        }

        double ConjugateGradientInversion::calculateStepSize(const core::dataGrid2D<double> &zeta, std::vector<std::complex<double>> &residualVector)
        {
            double alphaNumerator = 0.0;
            double alphaDenominator = 0.0;

            std::vector<std::complex<double>> kappaTimesZeta = _forwardModel->calculatePressureField(zeta);
            
            for(size_t i = 0; i < kappaTimesZeta.size(); i++)
            {
                alphaNumerator += std::real(conj(residualVector[i]) * kappaTimesZeta[i]);
                alphaDenominator += std::real(conj(kappaTimesZeta[i]) * kappaTimesZeta[i]);
            }

            if(alphaDenominator == 0.0)
            {
                throw std::overflow_error("ConjugateGradient: the computation of alpha devides by zero.");
            }
            double alpha = alphaNumerator / alphaDenominator;

            return alpha;
        }

        void ConjugateGradientInversion::logResidualResults(int it1, int it, double residual, int counter, std::ofstream &residualLogFile, bool isConverged)
        {
            std::string convergenceMessage = isConverged ? "Converged" : "Not Converged";
            L_(io::linfo) << it1 + 1 << "/" << _cgInput.iteration1.n << "\t (" << it + 1 << "/" << _cgInput.n_max << ")\t res: " << std::setprecision(17)
                          << residual << "\t" << convergenceMessage;
            residualLogFile << std::setprecision(17) << residual << "," << counter << std::endl;
        }

        void ConjugateGradientInversion::calculateRegularisationParameters(
            RegularisationParameters &regularisationPrevious, RegularisationParameters &regularisationCurrent, double deltaAmplification)
        {
            _chiEstimate.gradient(regularisationPrevious.gradientChi);

            regularisationPrevious.gradientChiNormSquared = (regularisationPrevious.gradientChi[0] * regularisationPrevious.gradientChi[0]) +
                                                            (regularisationPrevious.gradientChi[1] * regularisationPrevious.gradientChi[1]);

            regularisationCurrent.bSquared = calculateWeightingFactor(regularisationPrevious);   // # eq. 2.22
            regularisationCurrent.b = regularisationCurrent.bSquared;
            regularisationCurrent.b.sqrt();

            regularisationCurrent.deltaSquared = calculateSteeringFactor(regularisationPrevious, regularisationCurrent, deltaAmplification);   // eq. 2.23
            regularisationCurrent.gradient = calculateRegularisationGradient(regularisationPrevious);
        }

        core::dataGrid2D<double> ConjugateGradientInversion::calculateWeightingFactor(const RegularisationParameters &regularisationPrevious)
        {
            core::dataGrid2D bSquared(_grid);
            bSquared = regularisationPrevious.gradientChiNormSquared + regularisationPrevious.deltaSquared;
            bSquared.reciprocal();
            bSquared *= (1.0 / (_grid.getDomainArea()));
            return bSquared;
        }

        double ConjugateGradientInversion::calculateSteeringFactor(
            const RegularisationParameters &regularisationPrevious, const RegularisationParameters &regularisationCurrent, double deltaAmplification)
        {
            core::dataGrid2D bTimesGradientChiXSquared = regularisationCurrent.b * regularisationPrevious.gradientChi[0];
            bTimesGradientChiXSquared.square();
            core::dataGrid2D bTimesGradientChiZSquared = regularisationCurrent.b * regularisationPrevious.gradientChi[1];
            bTimesGradientChiZSquared.square();

            double bTimesGradientChiNormSquared = (bTimesGradientChiXSquared + bTimesGradientChiZSquared).summation();
            double bSquaredSummed = regularisationCurrent.bSquared.summation();

            return deltaAmplification * 0.5 * bTimesGradientChiNormSquared / bSquaredSummed;
        }

        core::dataGrid2D<double> ConjugateGradientInversion::calculateRegularisationGradient(const RegularisationParameters &regularisationPrevious)
        {
            std::vector<core::dataGrid2D<double>> temporaryGradient(2, core::dataGrid2D(_grid));

            core::dataGrid2D bSquaredTimesGradientChiX = regularisationPrevious.bSquared * regularisationPrevious.gradientChi[0];
            bSquaredTimesGradientChiX.gradient(temporaryGradient);
            core::dataGrid2D gradientBSquaredTimesGradientChiX = temporaryGradient[0];

            core::dataGrid2D bSquaredTimesGradientChiZ = regularisationPrevious.bSquared * regularisationPrevious.gradientChi[1];
            bSquaredTimesGradientChiZ.gradient(temporaryGradient);
            core::dataGrid2D gradientBSquaredTimesGradientChiZ = temporaryGradient[1];

            return gradientBSquaredTimesGradientChiX + gradientBSquaredTimesGradientChiZ;
        }

        core::dataGrid2D<double> ConjugateGradientInversion::calculateUpdateDirectionRegularisation(std::vector<std::complex<double>> &residualVector,
            core::dataGrid2D<double> &gradientCurrent, const core::dataGrid2D<double> &gradientPrevious, const double eta,
            const RegularisationParameters &regularisationCurrent, const RegularisationParameters &regularisationPrevious, const core::dataGrid2D<double> &zeta,
            double residualPrevious)
        {
            core::dataGrid2D<std::complex<double>> kappaTimesResidual(_grid);
            kappaTimesResidual.zero();
            getUpdateDirectionInformation(residualVector, kappaTimesResidual);
            gradientCurrent = eta * regularisationPrevious.errorFunctional * kappaTimesResidual.getRealPart() +
                              residualPrevious * regularisationCurrent.gradient;   // eq: 2.25 of thesis
            double gamma = gradientCurrent.innerProduct(gradientCurrent - gradientPrevious) /
                           gradientPrevious.innerProduct(gradientPrevious);   // eq: PolakRibiereDirection README

            return gradientCurrent + gamma * zeta;
        }

        std::vector<std::complex<double>> ConjugateGradientInversion::calculateKappaTimesZeta(
            const core::dataGrid2D<double> &zeta, std::vector<std::complex<double>> &kernel)
        {   

//        	return _forwardModel->calculatePressureField(zeta);
            auto kappa = _forwardModel->getKernel();
            for(int i = 0; i < _frequencies.count * _sources.count * _receivers.count; i++)
            {
                kernel[i] = dotProduct(kappa[i], zeta);
            }
//            clock_t t_e = clock();
            // std::cout << "time to calculateKappaTimeZeta = " << (t_e - t_s) / (double) CLOCKS_PER_SEC <<  " s" << std::endl;
            return kernel;
        }

        double ConjugateGradientInversion::calculateStepSizeRegularisation(const RegularisationParameters &regularisationPrevious,
            RegularisationParameters &regularisationCurrent, const std::vector<std::complex<double>> &residualVector, const double eta,
            const double fDataPrevious, const core::dataGrid2D<double> &zeta)
        {
            std::vector<std::complex<double>> kappaTimesZeta(_frequencies.count * _sources.count * _receivers.count);
            calculateKappaTimesZeta(zeta, kappaTimesZeta);

            double a0 = fDataPrevious;

            double a1 = 0.0;
            double a2 = 0.0;

            for(size_t i = 0; i < kappaTimesZeta.size(); i++)
            {
                a1 += -2.0 * eta * std::real(conj(residualVector[i]) * kappaTimesZeta[i]);
                a2 += eta * std::real(conj(kappaTimesZeta[i]) * kappaTimesZeta[i]);
            }

            core::dataGrid2D bGradientChiSquaredXDirection =
                (regularisationCurrent.b * regularisationPrevious.gradientChi[0]) * (regularisationCurrent.b * regularisationPrevious.gradientChi[0]);
            core::dataGrid2D bGradientChiSquaredZDirection =
                (regularisationCurrent.b * regularisationPrevious.gradientChi[1]) * (regularisationCurrent.b * regularisationPrevious.gradientChi[1]);
            double b0 = ((bGradientChiSquaredXDirection.summation() + bGradientChiSquaredZDirection.summation()) +
                            regularisationPrevious.deltaSquared * regularisationCurrent.bSquared.summation()) *
                        _grid.getCellVolume();

            std::vector<core::dataGrid2D<double>> gradientZeta(2, core::dataGrid2D(_grid));
            zeta.gradient(gradientZeta);

            core::dataGrid2D bGradientZetabGradientChiX =
                (regularisationCurrent.b * gradientZeta[0]) * (regularisationCurrent.b * regularisationPrevious.gradientChi[0]);
            core::dataGrid2D bGradientZetabGradientChiZ =
                (regularisationCurrent.b * gradientZeta[1]) * (regularisationCurrent.b * regularisationPrevious.gradientChi[1]);
            double b1 = 2.0 * (bGradientZetabGradientChiX.summation() + bGradientZetabGradientChiZ.summation()) * _grid.getCellVolume();

            core::dataGrid2D bTimesGradientZetaXdirection = regularisationCurrent.b * gradientZeta[0];
            core::dataGrid2D bTimesGradientZetaZdirection = regularisationCurrent.b * gradientZeta[1];
            bTimesGradientZetaXdirection.square();
            bTimesGradientZetaZdirection.square();
            double b2 = (bTimesGradientZetaXdirection.summation() + bTimesGradientZetaZdirection.summation()) * _grid.getCellVolume();

            double derA = 4.0 * a2 * b2;
            double derB = 3.0 * (a2 * b1 + a1 * b2);
            double derC = 2.0 * (a2 * b0 + a1 * b1 + a0 * b2);
            double derD = a1 * b0 + a0 * b1;

            return findRealRootFromCubic(derA, derB, derC, derD);
        }

        double ConjugateGradientInversion::findRealRootFromCubic(double a, double b, double c, double d)
        {
            double f = ((3.0 * c / a) - (std::pow(b, 2) / std::pow(a, 2))) / 3.0;
            double g = ((2.0 * std::pow(b, 3) / std::pow(a, 3)) - (9.0 * b * c / std::pow(a, 2)) + (27.0 * d / a)) / 27.0;
            double h = (std::pow(g, 2) / 4.0) + (std::pow(f, 3) / 27.0);
            double r = -(g / 2.0) + std::sqrt(h);
            double s = std::cbrt(r);
            double t = -(g / 2.0) - std::sqrt(h);
            double u = std::cbrt(t);

            double realRoot = s + u - (b / (3.0 * a));

            return realRoot;
        }

        void ConjugateGradientInversion::calculateRegularisationErrorFunctional(
            RegularisationParameters &regularisationPrevious, RegularisationParameters &regularisationCurrent)
        {
            core::dataGrid2D<double> gradientChiNormsquaredCurrent(_grid);
            gradientChiNormsquaredCurrent = (regularisationCurrent.gradientChi[0] * regularisationCurrent.gradientChi[0]) +
                                            (regularisationCurrent.gradientChi[1] * regularisationCurrent.gradientChi[1]);

            core::dataGrid2D<double> integral = (gradientChiNormsquaredCurrent + regularisationPrevious.deltaSquared) /
                                                (regularisationPrevious.gradientChiNormSquared + regularisationPrevious.deltaSquared);

            regularisationPrevious.errorFunctional = (1.0 / (_grid.getDomainArea())) * integral.summation() * _grid.getCellVolume();
        }

        void ConjugateGradientInversion::getUpdateDirectionInformation(
            const std::vector<std::complex<double>> &residualVector, core::dataGrid2D<std::complex<double>> &kappaTimesResidual)
        {

        	_forwardModel->getUpdateDirectionInformation(residualVector,kappaTimesResidual);
        	return;
			cl_int err;
        	std::vector<std::complex<float>,Eigen::aligned_allocator<std::complex<float>>> result(500);
        	std::vector<std::complex<float>,Eigen::aligned_allocator<std::complex<float>>> residualV(residualVector.begin(),residualVector.end());
        	int reserrors =0;
        	int kappaerrors=0;
        	  for(int i = 0; i< 500; i++){
        	            	if((std::complex<float>)residualVector[i] != residualV[i]){
        	            		reserrors++;
        	            	}
        	            }

        	  int t=0;
        	  for(int i = 0; i< 500; i++){
        		  for(int j = 0; j < 500; j++){
        	            	if(_forwardModel->kappa1D[t] != (std::complex<float>)_forwardModel->getKernel()[i].getData()[j]){
        	            		kappaerrors++;
        	            	}
        	            	t++;
        	            }
        	  }
              std::cout << "kappaerrors: " << kappaerrors << std::endl;
              std::cout << "reserrors: " << reserrors << std::endl;

        	 	 //Allocate Buffer in Global Memory
        	    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and Device-to-host communication
        		OCL_CHECK(err, cl::Buffer buffer_in1(_forwardModel->context, CL_MEM_USE_HOST_PTR,sizeof(std::complex<float>)*500,
        	                                          residualV.data(), &err));

        	    OCL_CHECK(err, cl::Buffer buffer_in2(_forwardModel->context, CL_MEM_USE_HOST_PTR, sizeof(std::complex<float>)*500*500,
        	                                         _forwardModel->kappa1D.data(), &err));

        	    OCL_CHECK(err, cl::Buffer buffer_output(_forwardModel->context, CL_MEM_USE_HOST_PTR , sizeof(std::complex<float>)*500,
        	                                            result.data(), &err));

        	    OCL_CHECK(err, err = _forwardModel->krnl_vector_update.setArg(0, buffer_in1));
        	    OCL_CHECK(err, err = _forwardModel->krnl_vector_update.setArg(1, buffer_in2));
        	    OCL_CHECK(err, err = _forwardModel->krnl_vector_update.setArg(2, buffer_output));

        	    // Copy input data to device global memory
        	    OCL_CHECK(err, err = _forwardModel->q.enqueueMigrateMemObjects({buffer_in1, buffer_in2}, 0 /* 0 means from host*/));

        	    // Launch the Kernel
        	    OCL_CHECK(err, err = _forwardModel->q.enqueueTask(_forwardModel->krnl_vector_update));

        	    // Copy Result from Device Global Memory to Host Local Memory
        	    OCL_CHECK(err, err = _forwardModel->q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST));
        	    _forwardModel->q.finish();
        	    // OPENCL HOST CODE AREA END

        	    std::cout <<"kappa1d " << _forwardModel->kappa1D[3] << "vkappa " << _forwardModel->getKernel()[0].getData()[3] << std::endl;
        	    std::cout <<"resv " << residualV[3] << "resVV " << residualVector[3] << std::endl;
        	    std::cout << "res 0" << result[17] << std::endl;

        	    for(size_t i = 0; i < result.size(); i++){
        	    	kappaTimesResidual.setData(i,(std::complex<double>)result[i]);
        	    }

////
//            int l_i, l_j;
//            kappaTimesResidual.zero();
//            core::dataGrid2D<std::complex<double>> kDummy(_grid);
//            auto kappa = _forwardModel->getKernel();
//            for(int i = 0; i < _frequencies.count; i++)
//            {
//                l_i = i * _receivers.count * _sources.count;
//                for(int j = 0; j < _receivers.count; j++)
//                {
//                    l_j = j * _receivers.count;
//                    for(int k = 0; k < _receivers.count; k++)
//                    {
//                        kDummy = kappa[l_i + l_j + k];
//                        kDummy.conjugate();
//                        kappaTimesResidual += kDummy * residualVector[l_i + l_j + k];
//                    }
//                }
//            }
//            std::cout << "correct 0 "<< kappaTimesResidual.getData()[17] <<std::endl;
//            int errors = 0;
//            for(int i = 0; i< 100; i++){
//            	if((std::complex<float>)kappaTimesResidual.getData()[i] != result[i]){
//            		errors++;
//            	}
//            }
//            std::cout << "errors: " << errors << std::endl;
//            clock_t t_e = clock();
            // std::cout << "time to getUpdateDirectionInformation = " << (t_e - t_s) / (double) CLOCKS_PER_SEC <<  " s" << std::endl;
        }
    }   // namespace inversionMethods
}   // namespace fwi
