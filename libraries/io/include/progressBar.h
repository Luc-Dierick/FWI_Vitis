#pragma once

#include <iomanip>
#include <iostream>

#ifdef __unix__
#include <sys/ioctl.h>   //ioctl() and TIOCGWINSZ
#include <unistd.h>      //STDOUT_FILENO
#endif

namespace fwi
{
    namespace io
    {
        class progressBar
        {
        private:
            int _total;
            int _counter;
            int _terminalWidth;
            int findTerminalSize();
            void hideCursor();
            void showCursor();

        public:
            ~progressBar();
            progressBar(const int total = 100, const int counter = 0);
            void setCounter(const int counter);
            void setTerminalWidth(const int width);
            void setTotal(const int total);
            int getCounter() const;
            int getTotal() const;
            void print() const;
            void initBar() const;
            progressBar &operator++();
            progressBar &operator++(int);
            progressBar &operator--();
            progressBar &operator--(int);
        };
    }   // namespace io
}   // namespace fwi
