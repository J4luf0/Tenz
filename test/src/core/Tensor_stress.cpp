#include <bitset>
#include <iostream>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "core/PoolUSM.hpp"
#include "core/Tensor.hpp"
#include "TestUtils.hpp"

using gema::PoolUSM;
using gema::Tensor;
using gema::LinearContainer;

constexpr uint64_t globalMultiplier = 1;



constexpr uint64_t constructionMultiplier = 16 * 16 * globalMultiplier;

TEST(tensor_stress_test, construction_001){

    const LinearContainer<uint64_t> dimensionSizes{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

    const uint64_t loopCount = (constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, construction_001_control){

    const LinearContainer<uint64_t> dimensionSizes{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

    const uint64_t loopCount = (constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        //Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, construction_002){

    const LinearContainer<uint64_t> dimensionSizes{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

    const uint64_t loopCount = (constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, construction_002_control){

    const LinearContainer<uint64_t> dimensionSizes{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

    const uint64_t loopCount = (constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        //Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, construction_003){

    const LinearContainer<uint64_t> dimensionSizes{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

    const uint64_t loopCount = (constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, construction_003_control){

    const LinearContainer<uint64_t> dimensionSizes{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

    const uint64_t loopCount = (constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        //Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, construction_004){

    const LinearContainer<uint64_t> dimensionSizes{256 * 256};

    const uint64_t loopCount = (constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, construction_004_control){

    const LinearContainer<uint64_t> dimensionSizes{256 * 256};

    const uint64_t loopCount = (constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        //Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, construction_005){

    const LinearContainer<uint64_t> dimensionSizes{256 * 256 * 2};

    const uint64_t loopCount = (constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, construction_005_control){

    const LinearContainer<uint64_t> dimensionSizes{256 * 256 * 2};

    const uint64_t loopCount = (constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        //Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, construction_006){

    const LinearContainer<uint64_t> dimensionSizes{256 * 256 * 4};

    const uint64_t loopCount = (constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, construction_006_control){

    const LinearContainer<uint64_t> dimensionSizes{256 * 256 * 4};

    const uint64_t loopCount = (constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        //Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}




constexpr uint64_t getItemMultiplier = 256 * 256 * 16 * globalMultiplier;

TEST(tensor_stress_test, getItem_001){

    const LinearContainer<uint64_t> dimensionSizes{256, 256, 256};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = (getItemMultiplier);

    Tensor<int> tensor = Tensor<int>(dimensionSizes);

    LinearContainer<uint64_t> currentCoords = {0, 0, 0};
    uint64_t i = 0;
    for(; i < loopCount; i++){
        
        tensor.getItem(currentCoords);

        if(i == loopCount - 1) break;

        Tensor<int>::incrementCoords(currentCoords, dimensionSizes);
    }
}

TEST(tensor_stress_test, getItem_001_control){

    const LinearContainer<uint64_t> dimensionSizes{256, 256, 256};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = (getItemMultiplier);

    Tensor<int> tensor = Tensor<int>(dimensionSizes);

    LinearContainer<uint64_t> currentCoords = {0, 0, 0};
    for(uint64_t i = 0; i < loopCount; i++){
        
        //tensor.getItem(currentCoords);

        if(i == loopCount - 1) break;

        Tensor<int>::incrementCoords(currentCoords, dimensionSizes);
    }
}

TEST(tensor_stress_test, getItem_002){

    const LinearContainer<uint64_t> dimensionSizes{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = (getItemMultiplier);

    Tensor<int> tensor = Tensor<int>(dimensionSizes);

    LinearContainer<uint64_t> currentCoords = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t i = 0;
    for(; i < loopCount; i++){
        
        tensor.getItem(currentCoords);

        if(i == loopCount - 1) break;

        Tensor<int>::incrementCoords(currentCoords, dimensionSizes);
    }
}

TEST(tensor_stress_test, getItem_002_control){

    const LinearContainer<uint64_t> dimensionSizes{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = (getItemMultiplier);

    Tensor<int> tensor = Tensor<int>(dimensionSizes);

    LinearContainer<uint64_t> currentCoords = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for(uint64_t i = 0; i < loopCount; i++){
        
        //tensor.getItem(currentCoords);

        if(i == loopCount - 1) break;

        Tensor<int>::incrementCoords(currentCoords, dimensionSizes);
    }
}




constexpr uint64_t transpositionMultiplier = 256 * globalMultiplier;

TEST(tensor_stress_test, transposition_001){

    const LinearContainer<uint64_t> dimensionSizes{16, 16, 16};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = transpositionMultiplier;

    Tensor<int> tensor = Tensor<int>(dimensionSizes);

    uint64_t i = 0;
    for(; i < loopCount; i++){
        tensor.transposition();
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, transposition_001_control){

    const LinearContainer<uint64_t> dimensionSizes{16, 16, 16};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = transpositionMultiplier;

    Tensor<int> tensor = Tensor<int>(dimensionSizes);

    uint64_t i = 0;
    for(; i < loopCount; i++){
        //tensor.transposition();
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, transposition_002){

    const LinearContainer<uint64_t> dimensionSizes{4096, 4096, 16};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = transpositionMultiplier / 256;

    Tensor<int> tensor = Tensor<int>(dimensionSizes);

    uint64_t i = 0;
    for(; i < loopCount; i++){
        tensor.transposition();
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, transposition_002_control){

    const LinearContainer<uint64_t> dimensionSizes{4096, 4096, 16};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = transpositionMultiplier / 256;

    Tensor<int> tensor = Tensor<int>(dimensionSizes);

    uint64_t i = 0;
    for(; i < loopCount; i++){
        //tensor.transposition();
        doNotOptimizeAway(&i);
    }
}


constexpr uint64_t resizeMultiplier = 256 * globalMultiplier;

TEST(tensor_stress_test, resize_001){

    const LinearContainer<uint64_t> dimensionSizes{16, 16, 16};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = resizeMultiplier;

    Tensor<int> tensor = Tensor<int>(dimensionSizes);

    uint64_t i = 0;
    for(; i < loopCount; i++){
        uint64_t a = std::max(i % 16, 1ul);
        uint64_t b = std::max((i + 6), 1ul);
        uint64_t c = std::max((i + 12) % 16, 1ul);
        tensor.resize({a, b, c});
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, resize_001_control){

    const LinearContainer<uint64_t> dimensionSizes{16, 16, 16};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = resizeMultiplier;

    Tensor<int> tensor = Tensor<int>(dimensionSizes);

    uint64_t i = 0;
    for(; i < loopCount; i++){
        uint64_t a = std::max(i % 16, 1ul);
        uint64_t b = std::max((i + 6), 1ul);
        uint64_t c = std::max((i + 12) % 16, 1ul);
        //tensor.resize({a, b, c});
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, resize_002){

    const LinearContainer<uint64_t> dimensionSizes{4096, 4096, 16};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = resizeMultiplier / 256;

    Tensor<int> tensor = Tensor<int>(dimensionSizes);

    uint64_t i = 0;
    for(; i < loopCount; i++){
        uint64_t a = std::max(i % 16, 1ul);
        uint64_t b = std::max((i + 6), 1ul);
        uint64_t c = std::max((i + 12) % 16, 1ul);
        tensor.resize({a, b, c});
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, resize_002_control){

    const LinearContainer<uint64_t> dimensionSizes{4096, 4096, 16};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = resizeMultiplier / 256;

    Tensor<int> tensor = Tensor<int>(dimensionSizes);

    uint64_t i = 0;
    for(; i < loopCount; i++){
        uint64_t a = std::max(i % 16, 1ul);
        uint64_t b = std::max((i + 6), 1ul);
        uint64_t c = std::max((i + 12) % 16, 1ul);
        //tensor.resize({a, b, c});
        doNotOptimizeAway(&i);
    }
}



constexpr uint64_t additionMultiplier = 256 * 16 * globalMultiplier;

TEST(tensor_stress_test, addition_001){

    const LinearContainer<uint64_t> dimensionSizes{16, 16, 16};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = additionMultiplier;

    Tensor<int> tensor = Tensor<int>(dimensionSizes);
    Tensor<int> tensor2 = Tensor<int>(dimensionSizes);

    uint64_t i = 0;
    for(; i < loopCount; i++){
        tensor + tensor2;
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, addition_001_control){

    const LinearContainer<uint64_t> dimensionSizes{16, 16, 16};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = additionMultiplier;

    Tensor<int> tensor = Tensor<int>(dimensionSizes);
    Tensor<int> tensor2 = Tensor<int>(dimensionSizes);

    uint64_t i = 0;
    for(; i < loopCount; i++){
        //tensor + tensor2;
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, addition_002){

    const LinearContainer<uint64_t> dimensionSizes{16, 16, 16};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = additionMultiplier;

    Tensor<int> tensor = Tensor<int>(dimensionSizes);
    Tensor<int> tensor2 = Tensor<int>(dimensionSizes);

    uint64_t i = 0;
    for(; i < loopCount; i++){
        tensor2 += tensor;
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, addition_002_control){

    const LinearContainer<uint64_t> dimensionSizes{16, 16, 16};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = additionMultiplier;

    Tensor<int> tensor = Tensor<int>(dimensionSizes);
    Tensor<int> tensor2 = Tensor<int>(dimensionSizes);

    uint64_t i = 0;
    for(; i < loopCount; i++){
        //tensor2 += tensor;
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, addition_003){

    const LinearContainer<uint64_t> dimensionSizes{4096, 4096, 64};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = additionMultiplier / (256 * 16);

    Tensor<int> tensor = Tensor<int>(dimensionSizes);
    Tensor<int> tensor2 = Tensor<int>(dimensionSizes);

    uint64_t i = 0;
    for(; i < loopCount; i++){
        tensor2 += tensor;
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, addition_003_control){

    const LinearContainer<uint64_t> dimensionSizes{4096, 4096, 64};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = additionMultiplier / (256 * 16);

    Tensor<int> tensor = Tensor<int>(dimensionSizes);
    Tensor<int> tensor2 = Tensor<int>(dimensionSizes);

    uint64_t i = 0;
    for(; i < loopCount; i++){
        //tensor2 += tensor;
        doNotOptimizeAway(&i);
    }
}



constexpr uint64_t divisionMultiplier = 256 * globalMultiplier;

TEST(tensor_stress_test, division_001){

    PoolUSM::freePool();

    const LinearContainer<uint64_t> dimensionSizes{16, 16, 16};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = divisionMultiplier;

    Tensor<double> tensor = Tensor<double>(dimensionSizes);
    Tensor<double> tensor2 = Tensor<double>(dimensionSizes);

    uint64_t i = 0;
    for(; i < loopCount; i++){
        tensor / tensor2;
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, division_001_control){

    PoolUSM::freePool();

    const LinearContainer<uint64_t> dimensionSizes{16, 16, 16};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = divisionMultiplier;

    Tensor<double> tensor = Tensor<double>(dimensionSizes);
    Tensor<double> tensor2 = Tensor<double>(dimensionSizes);

    uint64_t i = 0;
    for(; i < loopCount; i++){
        //tensor / tensor2;
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, division_002){

    PoolUSM::freePool();

    const LinearContainer<uint64_t> dimensionSizes{4096, 4096, 32};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = divisionMultiplier / (256);

    Tensor<double> tensor = Tensor<double>(dimensionSizes);
    Tensor<double> tensor2 = Tensor<double>(dimensionSizes);

    uint64_t i = 0;
    for(; i < loopCount; i++){
        tensor / tensor2;
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, division_002_control){

    PoolUSM::freePool();

    const LinearContainer<uint64_t> dimensionSizes{4096, 4096, 32};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = divisionMultiplier / (256);

    Tensor<double> tensor = Tensor<double>(dimensionSizes);
    Tensor<double> tensor2 = Tensor<double>(dimensionSizes);

    uint64_t i = 0;
    for(; i < loopCount; i++){
        //tensor / tensor2;
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, division_003){

    PoolUSM::freePool();

    const LinearContainer<uint64_t> dimensionSizes{4096, 4096, 16};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = divisionMultiplier / (256);

    Tensor<double> tensor = Tensor<double>(dimensionSizes);
    Tensor<double> tensor2 = Tensor<double>(dimensionSizes);

    uint64_t i = 0;
    for(; i < loopCount; i++){
        tensor / tensor2;
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, division_003_control){

    PoolUSM::freePool();

    const LinearContainer<uint64_t> dimensionSizes{4096, 4096, 16};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = divisionMultiplier / (256);

    Tensor<double> tensor = Tensor<double>(dimensionSizes);
    Tensor<double> tensor2 = Tensor<double>(dimensionSizes);

    uint64_t i = 0;
    for(; i < loopCount; i++){
        //tensor / tensor2;
        doNotOptimizeAway(&i);
    }
}


