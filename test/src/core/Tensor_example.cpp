// #include <iostream>
// #include "core/LinearContainer.hpp"
// #include "core/Tensor.hpp"

// using namespace gema;

// int main(){

//     // kontejner pro definici prvotních rozměrů tenzoru
//     const LinearContainer<uint64_t> dimensionSizes{2, 2};

//     // konstrukce klasického tenzoru o prvcích typu double
//     Tensor<double> tensor = Tensor<double>(dimensionSizes);

//     // nastavování hodnot jednotlivým prvkům
//     tensor.setItem(5.5,  {0, 0});
//     tensor.setItem(-2.,  {0, 1});
//     tensor.setItem(10.,  {1, 0});
//     tensor.setItem(-0.1, {1, 1});

//     // přidání dimenze velikosti 3 mezi první a druhou již existující dimenzí
//     tensor.addDimension(3, 1); 

//     // transpozice, výchozí hodnoty jsou 0 a 1, čili první dvě dimenze
//     tensor.transposition();

//     // konstrukce dalšího tenzoru, tentokrát z inicializačního listu
//     Tensor<double> tensor2({2, 2, 3});

//     // nastavování prvků nového tenzoru, ty nenastavené budou mít výchozí hodnotu
//     tensor2.setItem(-5., {0, 0});
//     tensor2.setItem(1.,  {0, 1});

//     // součet prvků tenzorů
//     Tensor<double> result = tensor + tensor2;

//     // převedení na řetězec znaků a výpis
//     std::cout << result << "\n";

//     return 0;
// }
