#include "progressBar.h"

namespace fwi
{
    namespace io
    {
        progressBar::~progressBar() { std::cerr << std::endl; }

        progressBar::progressBar(const int total, const int counter)
            : _total(total)
            , _counter(counter)
        {
            _terminalWidth = 80;   // findTerminalSize();
            initBar();
        }

        void progressBar::initBar() const
        {
            std::cerr << "      out of " << std::setw(5) << _total;   // << "[" << std::setw(5) << _counter;
        }

        void progressBar::setCounter(const int counter) { _counter = counter; }

        void progressBar::setTerminalWidth(const int width) { _terminalWidth = width; }

        void progressBar::setTotal(const int total)
        {
            _total = total;
            initBar();
        }

        int progressBar::getCounter() const { return _counter; }

        int progressBar::getTotal() const { return _total; }

        progressBar &progressBar::operator++()
        {
            _counter++;
            print();
            return *this;
        }

        progressBar &progressBar::operator++(int) { return ++*this; }

        // Define prefix decrement operator.
        progressBar &progressBar::operator--()
        {
            _counter--;
            print();
            return *this;
        }

        // Define postfix decrement operator.
        progressBar &progressBar::operator--(int) { return --*this; }

        void progressBar::print() const
        {
            std::cerr << "\r";
            std::cerr << std::setw(5) << _counter;   // << "/" << std::setw(5) << _total << "[";
        }
    }   // namespace io
}   // namespace fwi
