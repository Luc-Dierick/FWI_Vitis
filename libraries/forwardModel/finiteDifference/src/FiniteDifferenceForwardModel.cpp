#include "FiniteDifferenceForwardModel.h"
#include "FiniteDifferenceForwardModelInputCardReader.h"
#include "Helmholtz2D.h"
#include "log.h"


typedef std::complex<float> complex_float;
#define	ROW 294
#define COL 100

namespace fwi
{
    namespace forwardModels
    {
        FiniteDifferenceForwardModel::FiniteDifferenceForwardModel(const core::grid2D &grid, const core::Sources &source, const core::Receivers &receiver,
            const core::FrequenciesGroup &freq, const finiteDifferenceForwardModelInput &fMInput, std::string xclbin)
            : _grid(grid)
            , _source(source)
            , _receiver(receiver)
            , _freq(freq)
            , _Greens()
            , _vpTot()
            , _vkappa()
            , _fMInput(fMInput)
        	, binaryFile(xclbin)
        {
            L_(io::linfo) << "Creating Greens function field...";
            createGreens();
            L_(io::linfo) << "Creating initial Ptot";
            createPTot(freq, source);
            createKappa(freq, source, receiver);
            calculateKappa();
            setup_openCL();
        }

        FiniteDifferenceForwardModel::~FiniteDifferenceForwardModel() {}

        void FiniteDifferenceForwardModel::setup_openCL(){
        	// OPENCL HOST CODE AREA START
        	    // get_xil_devices() is a utility API which will find the xilinx
        	    // platforms and will return list of devices connected to Xilinx platform
        	    auto devices = xcl::get_xil_devices();

        	    // read_binary_file() is a utility API which will load the binaryFile
        	    // and will return the pointer to file buffer.
        	    auto fileBuf = xcl::read_binary_file(binaryFile);
        	    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
        	    bool valid_device = false;
        	    for (unsigned int i = 0; i < devices.size(); i++) {
        	        auto device = devices[i];
        	        // Creating Context and Command Queue for selected Device
        	        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        	        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

        	        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        	        cl::Program program(context, {device}, bins, nullptr, &err);
        	        if (err != CL_SUCCESS) {
        	            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        	        } else {
        	            std::cout << "Device[" << i << "]: program successful!\n";
        	            OCL_CHECK(err, krnl_vector_update = cl::Kernel(program, "update", &err));
        	            std::cout <<"update kernel exists" <<std::endl;
        	            OCL_CHECK(err, krnl_vector_dotprod = cl::Kernel(program, "dotprod", &err));
        	            std::cout <<"dotprod kernel exists" <<std::endl;

        	            valid_device = true;
        	            break; // we break because we found a valid device
        	        }
        	    }
        	    if (!valid_device) {
        	        std::cout << "Failed to program any device found, exit!\n";
        	        exit(EXIT_FAILURE);
        	    }

//        	    // Allocate Memory in Host Memory UPDATE
//        	        std::vector<complex_float,Eigen::aligned_allocator<complex_float> > resVect(ROW);
//        	        std::vector<complex_float,Eigen::aligned_allocator<complex_float> > kappa(ROW*COL);
//        	        complex_float kappaSW[ROW][COL];
//        	        std::vector<complex_float,Eigen::aligned_allocator<complex_float> > kappaTimesResSW(COL,0);
//        	        std::vector<complex_float,Eigen::aligned_allocator<complex_float> > kappaTimesResHW(COL,0);
//
//        	        int i,j;
//        	        int t= 0;
//        	               for(i = 0; i<ROW; i++){
//        	                   for(j = 0; j<COL; j++){
//        	                	   kappaSW[i][j] = {i*1.0f,j*i*0.33f};
//        	                       kappa[t] = {i*1.0f,j*i*0.33f};
//        	                       t++;
//        	                   }
//        	               }
//
//
//        	               for(i = 0; i<ROW; i++){
//        	               	   resVect[i] = {i*1.0f,i*0.33f};
//        	               	  }
//        	               /** End of Initiation */
//
//        	            // Create test data and Software Result
//        	              complex_float conj;
//        	            for(int row = 0; row < ROW; ++row){
//        	                    for(int col = 0; col < COL; ++col){
//        	                        conj.real(kappaSW[row][col].real() * resVect[row].real() + kappaSW[row][col].imag() * resVect[row].imag());
//        	                        conj.imag(-kappaSW[row][col].real() * resVect[row].imag() - kappaSW[row][col].imag() * resVect[row].real());
//        	                        kappaTimesResSW[col] += conj;
//        	                    }
//
//        	                }

        	                // Allocate Buffer in Global Memory
        	                    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and Device-to-host communication
//        	                    OCL_CHECK(err, cl::Buffer buffer_in1u(context, CL_MEM_USE_HOST_PTR,sizeof(complex_float)*ROW,
//        	                                                          resVect.data(), &err));
//        	                    OCL_CHECK(err, cl::Buffer buffer_in2u(context, CL_MEM_USE_HOST_PTR, sizeof(complex_float)*ROW*COL,
//        	                                                         kappa.data(), &err));
//        	                    OCL_CHECK(err, cl::Buffer buffer_outputu(context, CL_MEM_USE_HOST_PTR , sizeof(complex_float)*COL,
//        	                                                            kappaTimesResHW.data(), &err));
//
//        	                    OCL_CHECK(err, err = krnl_vector_update.setArg(0, buffer_in1u));
//        	                    OCL_CHECK(err, err = krnl_vector_update.setArg(1, buffer_in2u));
//        	                    OCL_CHECK(err, err = krnl_vector_update.setArg(2, buffer_outputu));
//
//        	                    // Copy input data to device global memory
//        	                    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1u, buffer_in2u}, 0 /* 0 means from host*/));
//
//        	                    // Launch the Kernel
//        	                    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_update));
//
//        	                    // Copy Result from Device Global Memory to Host Local Memory
//        	                    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_outputu}, CL_MIGRATE_MEM_OBJECT_HOST));
//        	                    q.finish();
//        	                    // OPENCL HOST CODE AREA END

//        	                    // Compare the results of the Device to the simulation
//        	                    bool matchu2 = true;
//        	                    for (int i = 0; i < COL; i++) {
//        	                        if (kappaTimesResSW[i] != kappaTimesResHW[i]) {
//        	                            std::cout << "Error: Result mismatch" << std::endl;
//        	                            std::cout << "i = " << i << " CPU result = " << kappaTimesResSW[i]
//        	                                      << " Device result = " << kappaTimesResHW[i] << std::endl;
//        	                            matchu2 = false;
//        	                            break;
//        	                        }
//        	                    }
//        	                    std::cout << " UPDATE TEST 2" << (matchu2 ? "PASSED" : "FAILED") << std::endl;
                int num_rows = _vkappa.size();
                int num_cols = _vkappa[0].getNumberOfGridPoints();
                std::cout << "row: " << num_rows << "col: " << num_cols << std::endl;
                // store data of _vkappa contiguously into kappa1D
                for(int i = 0; i < num_rows; i++){
                    for(int j = 0; j < num_cols; j++){
                        this->kappa1D.push_back((std::complex<float>)_vkappa[i].getData()[j]);
                    }
                }
        }

        void FiniteDifferenceForwardModel::createGreens()
        {
            for(int i = 0; i < _freq.count; i++)
            {
                _Greens.push_back(core::greensRect2DCpu(_grid, core::greensFunctions::Helmholtz2D, _source, _receiver, _freq.k[i]));
            }
        }

        void FiniteDifferenceForwardModel::createPTot(const core::FrequenciesGroup &freq, const core::Sources &source)
        {
            for(int i = 0; i < freq.count; i++)
            {
                for(int j = 0; j < source.count; j++)
                {
                    _vpTot.push_back(
                        core::dataGrid2D<std::complex<double>>(*_Greens[i].getReceiverCont(j) / (_freq.k[i] * _freq.k[i] * _grid.getCellVolume())));
                }
            }
        }

        void FiniteDifferenceForwardModel::createKappa(const core::FrequenciesGroup &freq, const core::Sources &source, const core::Receivers &receiver)
        {
            for(int i = 0; i < freq.count * source.count * receiver.count; i++)
            {
                _vkappa.push_back(core::dataGrid2D<std::complex<double>>(_grid));
            }
        }

        void FiniteDifferenceForwardModel::calculatePTot(const core::dataGrid2D<double> &chiEst)
        {
            int li;

            for(int i = 0; i < _freq.count; i++)
            {
                li = i * _source.count;

                Helmholtz2D helmholtzFreq(_grid, _freq.freq[i], _source, _freq.c0, chiEst, _fMInput);

                L_(io::linfo) << "Creating this->p_tot for " << i + 1 << "/ " << _freq.count << "freq";

                for(int j = 0; j < _source.count; j++)
                {
                    L_(io::linfo) << "Solving p_tot for source: (" << _source.xSrc[j][0] << "," << _source.xSrc[j][1] << ")";
                    _vpTot[li + j] = helmholtzFreq.solve(_source.xSrc[j], _vpTot[li + j]);
                }
            }
        }

        void FiniteDifferenceForwardModel::getUpdateDirectionInformation(
                    const std::vector<std::complex<double>> &residualVector, core::dataGrid2D<std::complex<double>> &kappaTimesResidual)
                {
                	std::vector<std::complex<float>,Eigen::aligned_allocator<std::complex<float>>> result(500,0);
                	std::vector<std::complex<float>,Eigen::aligned_allocator<std::complex<float>>> residualV(residualVector.begin(),residualVector.end());

                	 	 //Allocate Buffer in Global Memory
                	    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and Device-to-host communication
                		OCL_CHECK(err, cl::Buffer buffer_in1u(context, CL_MEM_USE_HOST_PTR,sizeof(std::complex<float>)*500,
                	                                          residualV.data(), &err));

                	    OCL_CHECK(err, cl::Buffer buffer_in2u(context, CL_MEM_USE_HOST_PTR, sizeof(std::complex<float>)*500*500,
                	                                         this->kappa1D.data(), &err));

                	    OCL_CHECK(err, cl::Buffer buffer_outputu(context, CL_MEM_USE_HOST_PTR , sizeof(std::complex<float>)*500,
                	                                            result.data(), &err));

                	    OCL_CHECK(err, err = krnl_vector_update.setArg(0, buffer_in1u));
                	    OCL_CHECK(err, err = krnl_vector_update.setArg(1, buffer_in2u));
                	    OCL_CHECK(err, err = krnl_vector_update.setArg(2, buffer_outputu));

                	    // Copy input data to device global memory
                	    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1u, buffer_in2u}, 0 /* 0 means from host*/));

                	    // Launch the Kernel
                	    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_update));

                	    // Copy Result from Device Global Memory to Host Local Memory
                	    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_outputu}, CL_MIGRATE_MEM_OBJECT_HOST));
                	    q.finish();
                	    // OPENCL HOST CODE AREA END

                	  for(size_t i = 0; i < result.size(); i++){
                	    	kappaTimesResidual.setData(i,(std::complex<double>)result[i]);
                	    }

        //
//                    int l_i, l_j;
//                    kappaTimesResidual.zero();
//                    core::dataGrid2D<std::complex<double>> kDummy(_grid);
//                    for(int i = 0; i < _freq.count; i++)
//                    {
//                        l_i = i * _receiver.count * _source.count;
//                        for(int j = 0; j < _receiver.count; j++)
//                        {
//                            l_j = j * _receiver.count;
//                            for(int k = 0; k < _receiver.count; k++)
//                            {
//                                kDummy = _vkappa[l_i + l_j + k];
//                                kDummy.conjugate();
//                                kappaTimesResidual += kDummy * residualVector[l_i + l_j + k];
//                            }
//                        }
//                    }
//                    std::cout << "correct 0 "<< kappaTimesResidual.getData()[17] <<std::endl;
//                    int errors = 0;
//                    for(int i = 0; i< 100; i++){
//                    	if((std::complex<float>)kappaTimesResidual.getData()[i] != result[i]){
//                    		errors++;
//                    	}
//                    }
//                    std::cout << "errors: " << errors << std::endl;
//        //            clock_t t_e = clock();
//                    // std::cout << "time to getUpdateDirectionInformation = " << (t_e - t_s) / (double) CLOCKS_PER_SEC <<  " s" << std::endl;
                }

        std::vector<std::complex<double>> FiniteDifferenceForwardModel::calculatePressureField(const core::dataGrid2D<double> &chiEst)
        {

            std::vector<std::complex<double>> kOperator(_freq.count * _source.count * _receiver.count);
            applyKappa(chiEst, kOperator);
            return kOperator;


//            std::vector<std::complex<float>,Eigen::aligned_allocator<std::complex<float>>> kOperator(_freq.count * _source.count * _receiver.count);
//
//            std::vector<float,Eigen::aligned_allocator<float>> chi(chiEst.getData().begin(), chiEst.getData().end());


//           //  Allocate Buffer in Global Memory
//			// Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and Device-to-host communication
//			OCL_CHECK(err, cl::Buffer buffer_in2d(context, CL_MEM_USE_HOST_PTR,sizeof(float)*100,
//												  chi.data(), &err));
//			OCL_CHECK(err, cl::Buffer buffer_in1d(context, CL_MEM_USE_HOST_PTR, sizeof(std::complex<float>)*294*100,
//												 this->kappa1D.data(), &err));
//			OCL_CHECK(err, cl::Buffer buffer_outputd(context, CL_MEM_USE_HOST_PTR , sizeof(std::complex<float>)*294,
//													kOperator.data(), &err));
//
//			OCL_CHECK(err, err = krnl_vector_dotprod.setArg(0, buffer_in1d));
//			OCL_CHECK(err, err = krnl_vector_dotprod.setArg(1, buffer_in2d));
//			OCL_CHECK(err, err = krnl_vector_dotprod.setArg(2, buffer_outputd));
//
//			// Copy input data to device global memory
//			OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1d, buffer_in2d}, 0 /* 0 means from host*/));
//
//			// Launch the Kernel
//			OCL_CHECK(err, err = q.enqueueTask(krnl_vector_dotprod));
//
//			// Copy Result from Device Global Memory to Host Local Memory
//			OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_outputd}, CL_MIGRATE_MEM_OBJECT_HOST));
//			q.finish();
//			// OPENCL HOST CODE AREA END
//
//
////            clock_t t_e = clock();
//            // std::cout << "CALCULATEPRESSUREFIELD TAKES: " << (t_e - t_s) / (double) CLOCKS_PER_SEC << std::endl;
//            std::vector<std::complex<double>> r(kOperator.begin(), kOperator.end());
//			return r;
        }

        void FiniteDifferenceForwardModel::calculateKappa()
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
        }

        void FiniteDifferenceForwardModel::applyKappa(const core::dataGrid2D<double> &CurrentPressureFieldSerial, std::vector<std::complex<double>> &kOperator)
        {
            for(int i = 0; i < _freq.count * _source.count * _receiver.count; i++)
            {
                kOperator[i] = dotProduct(_vkappa[i], CurrentPressureFieldSerial);
            }
        }

        void FiniteDifferenceForwardModel::getResidualGradient(std::vector<std::complex<double>> &res, core::dataGrid2D<std::complex<double>> &kRes)
        {
            int l_i, l_j;

            kRes.zero();

            core::dataGrid2D<std::complex<double>> kDummy(_grid);

            for(int i = 0; i < _freq.count; i++)
            {
                l_i = i * _receiver.count * _source.count;

                for(int j = 0; j < _receiver.count; j++)
                {
                    l_j = j * _source.count;

                    for(int k = 0; k < _source.count; k++)
                    {
                        kDummy = _vkappa[l_i + l_j + k];
                        kRes += kDummy * res[l_i + l_j + k];
                    }
                }
            }
        }
    }   // namespace forwardModels
}   // namespace fwi
