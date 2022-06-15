#include <iostream>
#include <vector>
#include "HelpTextProcessing.h"
#include "genericInputCardReader.h"
#include "argumentReader.h"
#include "cpuClock.h"
#include "chiIntegerVisualisation.h"
#include "cpuClock.h"
#include "createChiCSV.h"
#include "csvReader.h"
#include "FiniteDifferenceForwardModel.h"
#include "FiniteDifferenceForwardModelInputCardReader.h"
#include "conjugateGradientInversion.h"
#include "conjugateGradientInversionInputCardReader.h"
#include "CostFunctionCalculator.h"
#include "genericInputCardReader.h"
#include "log.h"
#include <iostream>
#include <vector>


void printHelpOrVersion(fwi::io::argumentReader &fwiOpts);
void executeFullFWI(const fwi::io::argumentReader &fwiOpts);
void doProcess( const fwi::io::genericInput &gInput, std::string xclbin);
void writePlotInput(const fwi::io::genericInput &gInput, std::string msg);

int main(int argc, char *argv[])
{

    try
    {
        std::vector<std::string> arguments = {argv + 1, argv + argc};


        std::string xclbin = argv[1];
        fwi::io::argumentReader fwiOpts(arguments);

        for(const auto& v: arguments){
            std::cout <<"args: " << v << std::endl;
        }
        printHelpOrVersion(fwiOpts);

        fwi::io::genericInputCardReader genericReader(fwiOpts);
        const fwi::io::genericInput gInput = genericReader.getInput();
        doProcess(gInput,xclbin);
    }
    catch(const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        std::cout << "Use -h for help on parameter options and values." << std::endl;
        L_(fwi::io::lerror) << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void printHelpOrVersion(fwi::io::argumentReader &fwiOpts)
{
    if(fwiOpts.help)
    {
        std::cout << HELP_TEXT_PROCESSING << std::endl;
        std::exit(EXIT_SUCCESS);
    }

    if(fwiOpts.version)
    {
        std::cout << VERSION_PROCESSING << std::endl;
        std::exit(EXIT_SUCCESS);
    }
}

void doProcess(const fwi::io::genericInput& gInput, std::string xclbin)
{    std::cout << "Inversion Processing Started" << std::endl;

    // initialize the clock, grid sources receivers, grouped frequencies
    fwi::performance::CpuClock clock;
    fwi::core::grid2D grid(gInput.reservoirTopLeftCornerInM, gInput.reservoirBottomRightCornerInM, gInput.nGrid);
    fwi::core::Sources source(gInput.sourcesTopLeftCornerInM, gInput.sourcesBottomRightCornerInM, gInput.nSources);
    fwi::core::Receivers receiver(gInput.receiversTopLeftCornerInM, gInput.receiversBottomRightCornerInM, gInput.nReceivers);
    fwi::core::FrequenciesGroup freq(gInput.freq, gInput.c0);

    // initialize logging
    std::string logFileName = gInput.outputLocation + gInput.runName + "Process.log";

    if(!gInput.verbose)
    {
        std::cout << "Printing the program output onto a file named: " << logFileName << std::endl;
        fwi::io::initLogger(logFileName.c_str(), fwi::io::ldebug);
    }

    // Logging the setup
    L_(fwi::io::linfo) << "Starting Inversion Process";    L_(fwi::io::linfo) << "Starting Inversion Process";
    source.Print();
    receiver.Print();
    freq.Print(gInput.freq.nTotal);

    // Logging Chi from input
    L_(fwi::io::linfo) << "Visualisation of input chi (to be reconstructed)";
    fwi::io::chi_visualisation_in_integer_form(gInput.inputFolder + gInput.fileName + ".txt", gInput.nGridOriginal[0]);
    fwi::io::createCsvFilesForChi(gInput.inputFolder + gInput.fileName + ".txt", gInput, "chi_reference_");

    // Start inversion
    clock.Start();

    int magnitude = freq.count * source.count * receiver.count;
    std::vector<std::complex<double>> referencePressureData(magnitude);


    // Create model
    L_(fwi::io::linfo) << "Create ForwardModel";
    fwi::forwardModels::FiniteDifferenceForwardModel *model;
    fwi::forwardModels::finiteDifferenceForwardModelInputCardReader finitedifferencereader(gInput.caseFolder);
    model = new fwi::forwardModels::FiniteDifferenceForwardModel(grid, source, receiver, freq, finitedifferencereader.getInput(), xclbin);

    // Read chi from file and write to output file
    fwi::core::dataGrid2D chi(grid);
    std::string inputPath = gInput.inputFolder + gInput.fileName + ".txt";
    chi.fromFile(inputPath);

    std::cout<< chi.getData()[0] << std::endl;
    referencePressureData = model->calculatePressureField(chi);

    L_(fwi::io::linfo) << "Create inversionModel";
    const fwi::core::CostFunctionCalculator costCalculator(fwi::core::CostFunctionCalculator::CostFunctionEnum::leastSquares);
    fwi::inversionMethods::ConjugateGradientInversionInputCardReader conjugateGradientReader(gInput.caseFolder);
    fwi::inversionMethods::ConjugateGradientInversion *inverse;
    inverse = new fwi::inversionMethods::ConjugateGradientInversion(costCalculator, model, conjugateGradientReader.getInput());

    std::cout << "Calculating..." << std::endl;
    L_(fwi::io::linfo) << "Estimating Chi...";

    fwi::core::dataGrid2D chiEstimate = inverse->reconstruct(referencePressureData, gInput);

    L_(fwi::io::linfo) << "Writing to file";
    chiEstimate.toFile(gInput.outputLocation + "chi_est_" + gInput.runName + ".txt");

    clock.End();

    L_(fwi::io::linfo) << "Visualisation of the estimated chi using FWI";
    fwi::io::chi_visualisation_in_integer_form(gInput.outputLocation + "chi_est_" + gInput.runName + ".txt", gInput.nGrid[0]);
    fwi::io::createCsvFilesForChi(gInput.outputLocation + "chi_est_" + gInput.runName + ".txt", gInput, "chi_est_");

    std::string msg = clock.OutputString();
    writePlotInput(gInput, msg);
    fwi::io::endLogger();

    std::cout << "InversionProcess completed" << std::endl;
}

void writePlotInput(const fwi::io::genericInput &gInput, std::string msg)
{
    // This part is needed for plotting the chi values in postProcessing.py
    std::ofstream outputfwi;
    std::string runName = gInput.runName;
    outputfwi.open(gInput.outputLocation + runName + ".pythonIn");
    outputfwi << "This run was parametrized as follows:" << std::endl;
    outputfwi << "nxt   = " << gInput.nGrid[0] << std::endl;
    outputfwi << "nzt   = " << gInput.nGrid[1] << std::endl;
    outputfwi << "nxt_original   = " << gInput.nGridOriginal[0] << std::endl;
    outputfwi << "nzt_original   = " << gInput.nGridOriginal[1] << std::endl << msg;
    outputfwi.close();

    // This part is needed for plotting the chi values in postProcessing.py
    std::ofstream lastrun;
    lastrun.open(gInput.outputLocation + "/lastRunName.txt");
    lastrun << runName;
    lastrun.close();
}
