#include <iostream>
#include <cmath>
#include <vector>

int main() {
    const size_t N = 10000000;
#ifdef USE_DOUBLE
    typedef double real;
#else 
    typedef float real;
#endif 

    std::vector<real>  arr(N);
    for(size_t i = 0; i < N; ++i){
        arr[i] = std::sin(2 * M_PI * static_cast<real>(i)/N);
    }

    real sum = 0.0;
    for(const auto&val:arr){
        sum += val;
    }

    std::cout<< "Sum: "<< sum << std::endl;
    return 0;
}