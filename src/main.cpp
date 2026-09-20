#include "includes/Matrix.hpp"
#include <iostream>

int main() {
    Matrix mat1(6, 6);
    mat1.random(1, 100);
    Matrix mat2(6, 6);
    mat2.random(1, 100);

    Matrix addition = mat1 + mat2;
    Matrix product = mat1 * mat2;
    
    std::cout << "Matrix 1 : " << "\n";
    mat1.print();
    std::cout << "Matrix 2 : " << "\n";
    mat2.print();

    std::cout << "Addition Matrix : " << "\n";
    addition.print();

    std::cout << "Product Matrix : " << "\n";
    product.print();
    return 0;
}