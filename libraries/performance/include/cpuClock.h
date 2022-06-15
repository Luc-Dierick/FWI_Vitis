#pragma once

#include <chrono>
#include <ctime>
#include <string>

namespace fwi
{
    namespace performance
    {
        class CpuClock
        {
        private:
            clock_t t_start;
            clock_t t_end;
            std::chrono::system_clock::time_point start;
            std::chrono::system_clock::time_point finish;

        public:
            CpuClock();
            ~CpuClock();
            void Start();
            void End();
            void MemoryUse(long &virtual_mem, long &physical_mem);
            std::string OutputString();
        };
    }   // namespace performance
}   // namespace fwi
