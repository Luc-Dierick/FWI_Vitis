#pragma once
#include "contraction.h"
#include "grid2D.h"
#include "receivers.h"
#include "sources.h"
#include "utilityFunctions.h"
#ifdef SYCL
    #include <CL/sycl.hpp>
#endif
#include <Eigen/Dense>
#include <cassert>
#include <dataGrid2D.h>

using namespace Eigen;

namespace fwi
{
    namespace core
    {
        class greensRect2DCpu
        {
        public:
            greensRect2DCpu(const grid2D &grid_, const std::function<std::complex<double>(double, double)> gFunc, const Sources &source_,
                const Receivers &receiver_, double k_);

            greensRect2DCpu(const greensRect2DCpu &rhs)
                : greensRect2DCpu(rhs.getGrid(), rhs.G_func, rhs.source, rhs.receiver, rhs.k)
            {
            }
            ~greensRect2DCpu();

            const std::complex<double> *getGreensVolume() const { return gVol; }

            const dataGrid2D<std::complex<double>>* getReceiverCont(int iRecv) const { return gRecv[iRecv]; }

            dataGrid2D<std::complex<double>> contractWithField(const dataGrid2D<std::complex<double>> &x) const;

            const grid2D &getGrid() const { return grid; }

            // Babak 2018 10 25: This method generates the dot product of two matrices Greens function and contrast Sources dW
            // Equation ID: "rel:buildField"

            dataGrid2D<std::complex<double>> dot1(const dataGrid2D<std::complex<double>> &dW) const;

        private:
            void createGreensVolume();

            void createGreensVolumeAnkit();

            void createGreensRecv();

            void deleteGreensRecv();

            std::function<std::complex<double>(double, double)> G_func;

            const grid2D grid;
            const Sources source;
            const Receivers receiver;
            const double k;

            std::complex<double> *gVol;
            Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor> G_vol2;
            void setGreensFunction(dataGrid2D<std::complex<double>> &greensFunctionField, const std::function<std::complex<double>(double, double)> func);

            greensRect2DCpu &operator=(const greensRect2DCpu &) = delete;

        protected:
        std::vector<dataGrid2D<std::complex<double>> *> gRecv;


        };
    }   // namespace core
}   // namespace fwi
