#ifdef SYCL
#include "conjugateGradientDataParallelCppInversion.h"

namespace fwi
{
    namespace inversionMethods
    {
        conjugateGradientDataParallelCppInversion::conjugateGradientDataParallelCppInversion(const core::CostFunctionCalculator &costCalculator,
            forwardModels::FiniteDifferenceForwardModel *forwardModel, const ConjugateGradientInversionInput &invInput)
            : ConjugateGradientInversion(costCalculator, forwardModel, invInput)
        {
            L_(io::linfo) << "Running SYCL code in parallel on " << Q.get_device().get_info<sycl::info::device::name>() << std::endl;
        }

        conjugateGradientDataParallelCppInversion::~conjugateGradientDataParallelCppInversion() {}

        void conjugateGradientDataParallelCppInversion::getUpdateDirectionInformation(
            const std::vector<std::complex<double>> &residualVector, core::dataGrid2D<std::complex<double>> &kappaTimesResidual)
        {
            // clock_t t_s = clock();

            int rows, cols;
            rows = _forwardModel->getKernel().size();
            cols = _forwardModel->getKernel()[0].getNumberOfGridPoints();
            auto NDRange = sycl::range(cols);
            // GPU allocation
            typedef sycl::buffer<std::complex<double>> complexBuf;
            complexBuf bufKapRes(NDRange);
            complexBuf bufResVec(residualVector);
            complexBuf bufKappa(_forwardModel->kappa1D);

            // clock_t t_s1 = clock();

            Q.submit(
                [&](sycl::handler &h)
                {
                    sycl::accessor accKapRes(bufKapRes, h, sycl::write_only);
                    sycl::accessor accResVec(bufResVec, h, sycl::read_only);
                    sycl::accessor accKappa(bufKappa, h, sycl::read_only);

                    h.parallel_for(NDRange,
                        [=](sycl::id<1> idx)
                        {
                            int col = idx[0];
                            int rowcol;
                            std::complex<double> conj;
                            for(int row = 0; row < rows; row++)
                            {
                                rowcol = row * cols + col;
                                // sycl with cuda-backend does not understand complex*complex
                                conj.real(accKappa[rowcol].real() * accResVec[row].real() + accKappa[rowcol].imag() * accResVec[row].imag());
                                conj.imag(-accKappa[rowcol].real() * accResVec[row].imag() - accKappa[rowcol].imag() * accResVec[row].real());
                                accKapRes[col] += conj;
                            }
                        });
                });
            // clock_t t_e1 = clock();

            sycl::host_accessor accKapRes{bufKapRes};
            // write data back to datastructure that FWI is used to
            for(int n = 0; n < cols; n++)
            {
                kappaTimesResidual.setData(n, accKapRes[n]);
            }
            // clock_t t_e = clock();
            // std::cout << "getUpdateDirectionInformation = " << (t_e - t_s) / (double)CLOCKS_PER_SEC << " s" << std::endl;
        }

        std::vector<std::complex<double>> conjugateGradientDataParallelCppInversion::calculateKappaTimesZeta(
            const core::dataGrid2D<double> &zeta, std::vector<std::complex<double>> &kernel)
        {
            // clock_t t_s = clock();

            int rows = _forwardModel->getKernel().size();
            int cols = _forwardModel->getKernel()[0].getNumberOfGridPoints();

            // GPU allocation
            sycl::buffer<std::complex<double>> bufKernel{kernel};
            sycl::buffer<double> bufZeta{zeta.getData()};
            sycl::buffer<std::complex<double>> bufKappa(_forwardModel->kappa1D);

            Q.submit(
                [&](sycl::handler &h)
                {
                    auto NDRange = sycl::range(rows);

                    sycl::accessor accKernel(bufKernel, h, sycl::read_write);
                    sycl::accessor accZeta(bufZeta, h, sycl::read_only);
                    sycl::accessor accKappa(bufKappa, h, sycl::read_only);

                    // parallel calculation
                    h.parallel_for(NDRange,
                        [=](sycl::id<1> idx)
                        {
                            int i = idx[0];
                            for(int j = 0; j < cols; j++)
                            {
                                // lookup time or copying of data takes a long time.
                                accKernel[i] += accKappa[i * cols + j] * accZeta[j];
                            }
                        });
                });
            // clock_t t_e = clock();
            // std::cout << "calculateKappaTimeZeta = " << (t_e - t_s) / (double)CLOCKS_PER_SEC << " s" << std::endl;
            return kernel;
        }
    }   // namespace inversionMethods
}   // namespace fwi
#endif
