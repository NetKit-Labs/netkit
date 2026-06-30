#include "cli.hpp"
#include <iostream>

int main(int argc, char** argv)
{
    std::cout << std::unitbuf;
    return Cli::Run(argc, argv);
}
