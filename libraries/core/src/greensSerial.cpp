#include "greensSerial.h"

using namespace Eigen;

namespace fwi
{
    namespace core
    {
        greensRect2DCpu::greensRect2DCpu(
            const grid2D &grid_, const std::function<std::complex<double>(double, double)> gFunc, const Sources &source_, const Receivers &receiver_, double k_)
            : G_func(gFunc)
            , grid(grid_)
            , source(source_)
            , receiver(receiver_)
            , k(k_)
            , gVol()
            , gRecv()
        {
            const std::array<int, 2> &nx = grid.getGridDimensions();
            gVol = new std::complex<double>[(2 * nx[1] - 1) * (2 * nx[0] - 1)];
            createGreensVolume();
            createGreensRecv();
            createGreensVolumeAnkit();
        }

        greensRect2DCpu::~greensRect2DCpu()
        {
            deleteGreensRecv();
            delete[] gVol;
        }

        dataGrid2D<std::complex<double>> greensRect2DCpu::contractWithField(const dataGrid2D<std::complex<double>> &x) const
        {
            // Assure we are working on the same grid
            assert(grid == x.getGrid());
            dataGrid2D<std::complex<double>> outputField(grid);
            const std::array<int, 2> &nx = grid.getGridDimensions();
            contractGreensRect2D(gVol, x, outputField, nx, 2 * nx[0] - 1);
            return outputField;
        }

        // Babak 2018 10 25: This method generates the dot product of two matrices Greens function and contrast sources dW
        // Equation ID: "rel:buildField"
        dataGrid2D<std::complex<double>> greensRect2DCpu::dot1(const dataGrid2D<std::complex<double>> &dW) const
        {
            const std::array<int, 2> &nx1 = grid.getGridDimensions();
            const int &nx = nx1[0];
            const int &nz = nx1[1];

            int l1, l2, l3, l4;

            const std::vector<std::complex<double>> &dWData = dW.getData();

            assert(grid == dW.getGrid());

            Matrix<std::complex<double>, Dynamic, 1, ColMajor> dW_vec, eigprod;
            Matrix<std::complex<double>, Dynamic, 1, ColMajor> dummy;

            dW_vec.resize(nx * nz, NoChange);
            eigprod.resize(nx * nz, NoChange);
            dummy.resize(nx, NoChange);

            for(int i = 0; i < nx * nz; i++)
            {
                dW_vec(i) = dWData[i];
            }

            for(int i = 0; i < nx * nz; i++)
            {
                eigprod(i) = double(0.0);
            }

            for(int i = 0; i < nz; i++)
            {
                l1 = i * nx;
                for(int j = 0; j < nz - i; j++)
                {
                    l2 = j * nx;
                    l3 = l1 + l2;

                    dummy = (G_vol2.block(0, l1, nx, nx)) * dW_vec.block(l1 + l2, 0, nx, 1);
                    eigprod.block(l2, 0, nx, 1) += dummy;

                    if((2 * i + j < nz) && (i > 0))
                    {
                        l4 = 2 * l1 + l2;
                        eigprod.block(l4, 0, nx, 1) += dummy;
                    }
                }
            }

            for(int i = 1; i < nz; i++)
            {
                if(i <= nz - i)
                {
                    l1 = i * nx;
                    for(int j = 0; j < i; j++)
                    {
                        l2 = j * nx;
                        l3 = l1 + l2;
                        eigprod.block(l3, 0, nx, 1) += (G_vol2.block(0, l1, nx, nx)) * dW_vec.block(l2, 0, nx, 1);
                    }
                }
                else
                {
                    l1 = i * nx;
                    int lnz_i = nz - i;
                    for(int j = 0; j < lnz_i; j++)
                    {
                        l2 = j * nx;
                        l3 = l1 + l2;
                        eigprod.block(l3, 0, nx, 1) += (G_vol2.block(0, l1, nx, nx)) * dW_vec.block(l2, 0, nx, 1);
                    }
                }
            }

            dataGrid2D<std::complex<double>> product(grid);
            std::vector<std::complex<double>> productData(product.getNumberOfGridPoints(), 0.0);
            for(int i = 0; i < nx * nz; i++)
            {
                productData[i] = eigprod(i);
            }
            product = productData;

            return product;
        }

        void greensRect2DCpu::createGreensVolume()
        {
            double vol = grid.getCellVolume();
            const std::array<int, 2> &nx = grid.getGridDimensions();
            const std::array<double, 2> &dx = grid.getCellDimensions();

            for(int i = -nx[1] + 1; i <= nx[1] - 1; i++)
            {
                double z = i * dx[1];
                for(int j = -nx[0] + 1; j <= nx[0] - 1; j++)
                {
                    double x = j * dx[0];
                    double r = utilities::dist(z, x);
                    std::complex<double> val = G_func(k, r);
                    gVol[(nx[1] + i - 1) * (2 * nx[0] - 1) + (nx[0] + j - 1)] = val * vol;
                }
            }
        }

        void greensRect2DCpu::createGreensVolumeAnkit()
        {
            double vol = grid.getCellVolume();
            const std::array<int, 2> &nx1 = grid.getGridDimensions();
            const std::array<double, 2> &dx = grid.getCellDimensions();
            const std::array<double, 2> &x_min = grid.getGridStart();
            const int &nx = nx1[0];
            const int &nz = nx1[1];
            G_vol2.resize(nx, nx * nz);

            double p2_z = x_min[1] + dx[1] / double(2.0);
            for(int i = 0; i < nx; i++)
            {
                double p2_x = x_min[0] + (i + double(0.5)) * dx[0];
                dataGrid2D<std::complex<double>> G_x(grid);
                setGreensFunction(
                    G_x, [this, vol, p2_x, p2_z](const double &x, const double &y) { return vol * G_func(k, utilities::dist(x - p2_x, y - p2_z)); });

                for(int j = 0; j < nx * nz; j++)
                {
                    G_vol2(i, j) = G_x.getData()[j];
                }
            }
        }

        void greensRect2DCpu::createGreensRecv()
        {
            double vol = grid.getCellVolume();
            dataGrid2D<std::complex<double>> G_bound_cpu(grid);
            for(int i = 0; i < receiver.count; i++)
            {
                double x_receiver = receiver.xRecv[i][0];
                double z_receiver = receiver.xRecv[i][1];
                dataGrid2D<std::complex<double>> *G_bound = new dataGrid2D<std::complex<double>>(grid);

                setGreensFunction(*G_bound, [this, vol, x_receiver, z_receiver](
                                                const double x, const double z) { return vol * G_func(k, utilities::dist(x - x_receiver, z - z_receiver)); });

                gRecv.push_back(G_bound);
            }
        }

        void greensRect2DCpu::deleteGreensRecv()
        {
            for(int i = 0; i < receiver.count; i++)
            {
                delete gRecv[i];
            }
        }

        void greensRect2DCpu::setGreensFunction(dataGrid2D<std::complex<double>> &greensFunctionField, const std::function<std::complex<double>(double, double)> func)
        {
            const std::array<int, 2> &nx = getGrid().getGridDimensions();
            const std::array<double, 2> &dx = getGrid().getCellDimensions();   // Babak 2018 10 29: get rid of template for grid_rect_2D
            const std::array<double, 2> &x_min = getGrid().getGridStart();     // Babak 2018 10 29: get rid of template for grid_rect_2D

            std::vector<std::complex<double>> greensFunctionData(greensFunctionField.getNumberOfGridPoints(), 0.0);

            for(int j = 0; j < nx[1]; j++)
            {
                double z = x_min[1] + (double(0.5) + j) * dx[1];
                int nx_j = nx[0] * j;

                for(int i = 0; i < nx[0]; i++)
                {
                    double x = x_min[0] + (double(0.5) + i) * dx[0];
                    greensFunctionData[nx_j + i] = func(x, z);
                }
            }

            greensFunctionField = greensFunctionData;
        }
    }   // namespace core
}   // namespace fwi
