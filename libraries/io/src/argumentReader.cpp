#include "argumentReader.h"

namespace fwi
{
    namespace io
    {
        argumentReader::argumentReader(std::vector<std::string>& arguments) :
            dir(""),
            forward(""),
            inversion(""),
            stepdir(""),
            stepsize(""),
            help(false),
            version(false)
        {
            for(auto it = arguments.begin(); it != arguments.end(); ++it)
            {
                if(*it == "-h" || *it == "--help")
                    help = true;
                else if(*it == "-v" || *it == "--version")
                    version = true;
                else if((*it == "-d" || *it == "--dir") && (it + 1) != arguments.end())
                    dir = *++it; //"/home/ldierick/bulk/workspace/fwi/src/default/";
                else if((*it == "-f" || *it == "--forward") && (it + 1) != arguments.end())
                    forward = *++it;
                else if((*it == "-i" || *it == "--inversion") && (it + 1) != arguments.end())
                    inversion = *++it;
                else if((*it == "--stepdir") && (it + 1) != arguments.end())
                    stepdir = *++it;
                else if((*it == "--stepsize") && (it + 1) != arguments.end())
                    stepsize = *++it;
            }
        }
    }   // namespace io
}   // namespace fwi
