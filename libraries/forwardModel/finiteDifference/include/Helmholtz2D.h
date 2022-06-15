#pragma once

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4127)
#endif   // _MSC_VER
#ifdef SYCL
#include <CL/sycl.hpp>
#endif
#include "Eigen/Sparse"

#ifdef _MSC_VER
#pragma warning(pop)
#endif   // _MSC_VER

#include "FiniteDifferenceForwardModelInput.h"
#include "FiniteDifferenceGrid2D.h"
#include "dataGrid2D.h"
#include "frequenciesGroup.h"
#include "grid2D.h"
#include "sources.h"

#include <complex>
#include <string>
#include <vector>

namespace fwi
{
    namespace forwardModels
    {
        class Helmholtz2D
        {
        public:
            explicit Helmholtz2D(const core::grid2D &grid, const double freq, const core::Sources &source, const double c0, const core::dataGrid2D<double> &chi,
                const finiteDifferenceForwardModelInput &fmInput);
            ~Helmholtz2D();
            core::dataGrid2D<std::complex<double>> solve(const std::array<double, 2> &source, core::dataGrid2D<std::complex<double>> &pInit);
            Eigen::SparseMatrix<std::complex<double>> getSparseMatrixA() { return _A; }

        private:
            Eigen::SparseMatrix<std::complex<double>> _A;
            Eigen::VectorXcd _b;
            const core::grid2D _oldGrid;
            FiniteDifferenceGrid2D *_newGrid;
            std::array<int, 2> _PMLwidth;
            std::array<int, 2> _idxUpperLeftDomain, _idxLowerRightDomain;
            double _coordPMLLeft, _coordPMLRight, _coordPMLUp, _coordPMLDown;
            const double _freq;
            const double _c0;
            std::vector<double> _waveVelocity;
            Eigen::SparseLU<Eigen::SparseMatrix<std::complex<double>>, Eigen::NaturalOrdering<int>> _solver;
            const SourceParameter _sourceInput;

            void buildMatrix(BoundaryConditionType boundaryCondition);
            void buildVector(const std::array<double, 2> &source);
            void updateChi(const core::dataGrid2D<double> &chi);
            void CreatePMLMatrix(const double &omega, const std::array<double, 2> &dx, std::vector<Eigen::Triplet<std::complex<double>>> &triplets,
                const std::array<int, 2> &nx, const std::array<double, 2> &xMin) const;
            void CreateABCMatrix(const double &omega, const std::array<double, 2> &dx, std::vector<Eigen::Triplet<std::complex<double>>> &triplets,
                const std::array<int, 2> &nx) const;
            void CreateABCSecondOrderMatrix(const double &omega, const std::array<double, 2> &dx, std::vector<Eigen::Triplet<std::complex<double>>> &triplets,
                const std::array<int, 2> &nx) const;
        };
    }   // namespace forwardModels
}   // namespace fwi
