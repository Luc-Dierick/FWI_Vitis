#include "integralForwardModel.h"
#include "integralForwardModelInputCardReader.h"
#include "log.h"

namespace fwi
{
    namespace forwardModels
    {
        IntegralForwardModel::IntegralForwardModel(const core::grid2D &grid, const core::Sources &source, const core::Receivers &receiver,
            const core::FrequenciesGroup &freq, const integralForwardModelInput &fmInput)
            : _grid(grid)
            , _source(source)
            , _receiver(receiver)
            , _freq(freq)
            , _Greens()
            , _p0()
            , _vpTot()
            , _vKappa()
            , _fmInput(fmInput)
        {
            L_(io::linfo) << "Creating Greens function field...";
            createGreens();
            L_(io::linfo) << "Creating p0...";
            createP0();
            createPTot(freq, source);
            createKappa(freq, source, receiver);
            calculateKappa();
        }

        IntegralForwardModel::~IntegralForwardModel() {}

        void IntegralForwardModel::createP0()
        {
            for(int i = 0; i < _freq.count; i++)
            {
                int li = i * _source.count;

                for(int j = 0; j < _source.count; j++)
                {
                    _p0[li + j] = core::dataGrid2D<std::complex<double>>(_grid);
                    _p0[li + j] = *(_Greens[i].getReceiverCont(j)) / (_freq.k[i] * _freq.k[i] * _grid.getCellVolume());
                }
            }
        }

        void IntegralForwardModel::createGreens()
        {
            for(int i = 0; i < _freq.count; i++)
            {
                _Greens.push_back(core::greensRect2DCpu(_grid, core::greensFunctions::Helmholtz2D, _source, _receiver, _freq.k[i]));
            }
        }

        void IntegralForwardModel::createPTot(const core::FrequenciesGroup &freq, const core::Sources &source)
        {
            int li;

            for(int i = 0; i < freq.count; i++)
            {
                li = i * source.count;

                for(int j = 0; j < source.count; j++)
                {
                    _vpTot[li + j] = core::dataGrid2D<std::complex<double>>(_p0[li + j]);
                }
            }
        }

        void IntegralForwardModel::createKappa(const core::FrequenciesGroup &freq, const core::Sources &source, const core::Receivers &receiver)
        {
            for(int i = 0; i < freq.count * source.count * receiver.count; i++)
            {
                _vKappa.push_back(core::dataGrid2D<std::complex<double>>(_grid));
            }
        }

        core::dataGrid2D<std::complex<double>> IntegralForwardModel::calcTotalField(
            const core::greensRect2DCpu &G, const core::dataGrid2D<double> &chi, const core::dataGrid2D<std::complex<double>> &p_init)
        {
            assert(G.getGrid() == p_init.getGrid());

            core::dataGrid2D<std::complex<double>> chi_p(_grid), chi_p_old(_grid);
            core::dataGrid2D<std::complex<double>> dW(_grid), p_tot(_grid), f_rhs(_grid), matA_j(_grid);

            int n_cell = _grid.getNumberOfGridPoints();

            std::vector<core::dataGrid2D<std::complex<double>>> phi;
            Matrix<std::complex<double>, Dynamic, Dynamic, ColMajor> matA;
            Matrix<std::complex<double>, Dynamic, 1, ColMajor> b_f_rhs;
            Matrix<std::complex<double>, Dynamic, 1, ColMajor> alpha;

            std::vector<std::complex<double>> matA_j_data, rhs_data;
            matA.conservativeResize(n_cell, 1);
            b_f_rhs.conservativeResize(n_cell, 1);

            double res = 0.0;

            chi_p_old.zero();

            p_tot = p_init;

            for(int it = 0; it < _fmInput.nrOfIterations; it++)
            {
                chi_p = p_tot * chi;

                dW = chi_p - chi_p_old;

                double dWNorm = dW.norm();
                double chi_pNorm = chi_p.norm();

                res = dWNorm / chi_pNorm;

                if(res < _fmInput.tolerance && it != 0)
                {
                    if(true)
                    {
                        std::string itstring = std::to_string(it);
                        std::string line_to_print = "Convergence after " + itstring + "iterations";
                        L_(io::linfo) << line_to_print;
                    }

                    break;
                }

                if(_fmInput.calcAlpha)
                {
                    phi.push_back(G.contractWithField(dW));
                    f_rhs = G.contractWithField(p_init * chi);

                    rhs_data = f_rhs.getData();

                    matA_j = phi[it] - G.contractWithField(phi[it] * chi);
                    matA_j_data = matA_j.getData();

                    matA.conservativeResize(NoChange, it + 1);
                    alpha.conservativeResize(it + 1);

                    for(int i = 0; i < n_cell; i++)
                    {
                        matA(i, it) = matA_j_data[i];
                        b_f_rhs(i) = rhs_data[i];
                    }

                    alpha = matA.jacobiSvd(ComputeThinU | ComputeThinV).solve(b_f_rhs);

                    p_tot = p_init;

                    for(int j = 0; j < it + 1; j++)
                    {
                        p_tot += alpha[j] * phi[j];
                    }
                }
                else
                {
                    p_tot += G.dot1(dW);
                }

                chi_p_old = chi_p;
                chi_p.zero();
            }

            if(res >= _fmInput.tolerance)
            {
                L_(io::linfo) << "No convergence after " << _fmInput.nrOfIterations << " iterations."
                              << "Res = " << res;
            }

            return p_tot;
        }

        void IntegralForwardModel::calculatePTot(const core::dataGrid2D<double> &chiEst)
        {
            int li;

            for(int i = 0; i < _freq.count; i++)
            {
                li = i * _source.count;

                L_(io::linfo) << "Creating this->p_tot for " << i + 1 << "/ " << _freq.count << "freq";

                for(int j = 0; j < _source.count; j++)
                {
                    _vpTot[li + j] = calcTotalField(_Greens[i], chiEst, _p0[li + j]);
                }
            }
        }

        std::vector<std::complex<double>> IntegralForwardModel::calculatePressureField(const core::dataGrid2D<double> &chiEst)
        {
            std::vector<std::complex<double>> kOperator(_freq.count * _source.count * _receiver.count);
            applyKappa(chiEst, kOperator);
            return kOperator;
        }

        void IntegralForwardModel::calculateKappa()
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
                        _vKappa[li + lj + k] = *(_Greens[i].getReceiverCont(j)) * (_vpTot[i * _source.count + k]);
                    }
                }
            }
        }

        void IntegralForwardModel::applyKappa(const core::dataGrid2D<double> &CurrentPressureFieldSerial, std::vector<std::complex<double>> &kOperator)
        {
            for(int i = 0; i < _freq.count * _source.count * _receiver.count; i++)
            {
                kOperator[i] = dotProduct(_vKappa[i], CurrentPressureFieldSerial);
            }
        }

    }   // namespace forwardModels
}   // namespace fwi
