#ifdef SYCL
#include "FiniteDifferenceForwardModelDataParallelCPP.h"
#include "Helmholtz2D.h"
#include <complex>
#include <vector>

#include <CL/sycl.hpp>

namespace fwi
{
    namespace forwardModels
    {
        FiniteDifferenceForwardModelDataParallelCPP::FiniteDifferenceForwardModelDataParallelCPP(const core::grid2D &grid, const core::Sources &source,
            const core::Receivers &receiver, const core::FrequenciesGroup &freq, const finiteDifferenceForwardModelInput &fmInput)
            : FiniteDifferenceForwardModel(grid, source, receiver, freq, fmInput)
        {
        }

        FiniteDifferenceForwardModelDataParallelCPP::~FiniteDifferenceForwardModelDataParallelCPP() {}

        std::vector<std::complex<double>> FiniteDifferenceForwardModelDataParallelCPP::calculatePressureField(const core::dataGrid2D<double> &chiEst)
        {
            // clock_t t_s = clock();
            int rows = _freq.count * _source.count * _receiver.count;
            int cols = _vkappa[0].getData().size();
            auto NDRange = sycl::range(rows);

            std::vector<std::complex<double>> kOperator(_freq.count * _source.count * _receiver.count);
            sycl::queue Q;

            // GPU allocation
            sycl::buffer<std::complex<double>> kOperBuf{kOperator};
            sycl::buffer<double> chiBuf{chiEst.getData()};
            sycl::buffer<std::complex<double>> bufKappa{kappa1D};

            Q.submit(
                [&](sycl::handler &h)
                {
                    // accesmodes for data on GPU
                    sycl::accessor accKoper(kOperBuf, h, sycl::read_write);
                    sycl::accessor accChi(chiBuf, h, sycl::read_only);
                    sycl::accessor accKappa(bufKappa, h, sycl::read_only);
                    // parallel calculation
                    h.parallel_for(NDRange,
                        [=](sycl::id<1> idx)
                        {
                            int i = idx[0];
                            for(int j = 0; j < cols; j++)
                            {
                                accKoper[i] += accKappa[i * cols + j] * accChi[j];
                            }
                        });
                });
            // clock_t t_e = clock();
            // std::cout << "calculatePressurefield = " << (t_e - t_s) / (double)CLOCKS_PER_SEC << "s" << std::endl;
            return kOperator;
        }

        void FiniteDifferenceForwardModelDataParallelCPP::calculateKappa()
        {
            int li, lj;
            for(int i = 0; i < _freq.count; i++)
            {
                li = i * _receiver.count * _source.count;

                for(int j = 0; j < _receiver.count; j++)
                {
                    lj = j * _source.count;

                    for(int k = 0; k < _source.count; k++)
                    {
                        _vkappa[li + lj + k] = *(_Greens[i].getReceiverCont(j)) * (_vpTot[i * _source.count + k]);
                    }
                }
            }
            int num_rows = _vkappa.size();
            int num_cols = _vkappa[0].getNumberOfGridPoints();
            // get data out of _vkappa and into kappa1D
            for(int i = 0; i < num_rows; i++)
            {
                for(int j = 0; j < num_cols; j++)
                {
                    kappa1D.push_back(_vkappa[i].getData()[j]);
                }
            }
        }

    }   // namespace forwardModels

}   // namespace fwi

#endif