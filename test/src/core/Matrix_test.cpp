#include <bitset>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "TestUtils.hpp"

#define protected public // Hack pro možnost otestovat i privátní funkce
#include "core/Matrix.hpp"
#undef protected

using gema::Matrix;
using gema::LinearContainer;

TEST(matrix_test, constructor_001){

    Matrix<double> matrix = Matrix<double>(2, 3);

    Matrix<double> matrix2 = Matrix<double>(2, 3);

    EXPECT_EQ(matrix.getDimensionSizes(), matrix2.getDimensionSizes());
}

TEST(matrix_test, constructor_002){

    auto matrix = std::make_unique<Matrix<double>>(2, 3);

    matrix->setItem(5,      0, 0);
    matrix->setItem(0.55,   0, 1);
    matrix->setItem(-0,     0, 2);
    matrix->setItem(-2,     1, 0);
    matrix->setItem(4.5,    1, 1);
    matrix->setItem(7,      1, 2);

    EXPECT_EQ(matrix->getNumberOfDimensions(), 2);
    EXPECT_EQ(matrix->getDimensionSizes()[0], 2);
    EXPECT_EQ(matrix->getDimensionSizes()[1], 3);
}

TEST(matrix_test, constructor_003){

    Matrix<double> matrix(2, 2);

    matrix.setData({2, -3, 5, 10});

    Matrix<double> result(matrix);

    EXPECT_EQ(matrix, result);
}

TEST(matrix_test, constructor_004){

    Matrix<double> matrix(2, 2);
    matrix.setData({2, -3, 5, 10});

    Matrix<double> result(std::move(matrix));

    Matrix<double> expected(2, 2);
    expected.setData({2, -3, 5, 10});

    EXPECT_EQ(result, expected);
}

TEST(matrix_test, getItem_001){

    Matrix<int> matrix(2, 3);
    matrix.setData({0, 5, -1, 100, -2, -16});

    EXPECT_EQ(matrix.getItem(0, 0), 0);
    EXPECT_EQ(matrix.getItem(0, 1), 5);
    EXPECT_EQ(matrix.getItem(0, 2), -1);
    EXPECT_EQ(matrix.getItem(1, 0), 100);
    EXPECT_EQ(matrix.getItem(1, 1), -2);
    EXPECT_EQ(matrix.getItem(1, 2), -16);
}

TEST(matrix_test, setItem_001){

    Matrix<double> matrix(2, 2);

    matrix.setItem(5.5, 0, 0);
    matrix.setItem(-2.0, 0, 1);
    matrix.setItem(9.1, 1, 0);
    matrix.setItem(8.8, 1, 1);

    Matrix<double> expected(2, 2);
    expected.setData({5.5, -2.0, 9.1, 8.8});

    EXPECT_EQ(matrix, expected);
}

TEST(matrix_test, setData_001){

    Matrix<int> matrix(2, 2);
    matrix.setData({1, 2, 3, 4});

    Matrix<int> expected(2, 2);
    expected.setData({1, 2, 3, 4});

    EXPECT_EQ(matrix, expected);
}

TEST(matrix_test, fillWith_001){

    Matrix<int> matrix(2, 3);
    matrix.fillWith(69);

    Matrix<int> expected(2, 3);
    expected.setData({69, 69, 69, 69, 69, 69});

    EXPECT_EQ(matrix, expected);
}

TEST(matrix_test, toString_001){

    Matrix<int> matrix(2, 2);
    matrix.setData({1, 2, 3, 4});

    std::string expected = "{{1, 2}, {3, 4}}";

    EXPECT_EQ(matrix.toString(), expected);
}

TEST(matrix_test, transposition_001){

    Matrix<int> matrix(2, 3);
    matrix.setData({0, 5, -1, 100, -2, -16});

    matrix.transposition();

    Matrix<int> expected(3, 2);
    expected.setData({0, 100, 5, -2, -1, -16});

    EXPECT_EQ(matrix, expected);
}

TEST(matrix_test, resize_001){

    Matrix<int> matrix(2, 2);
    matrix.setData({1, 2, 3, 4});

    matrix.resize(3, 2);

    EXPECT_EQ(matrix.getDimensionSizes()[0], 3);
    EXPECT_EQ(matrix.getDimensionSizes()[1], 2);
    EXPECT_EQ(matrix.getNumberOfItems(), 6);
}

TEST(matrix_test, operatorAssign_001){

    Matrix<double> matrix(2, 2);
    matrix.setData({5, 0.55, -2, 4.5});

    Matrix<double> matrix2;
    matrix2 = matrix;

    matrix.setItem(8, 0, 0);

    Matrix<double> expected(2, 2);
    expected.setData({5, 0.55, -2, 4.5});

    EXPECT_EQ(matrix2, expected);
}

TEST(matrix_test, operatorAssign_002){

    Matrix<double> matrix(2, 2);
    matrix.setData({5, 0.55, -2, 4.5});

    Matrix<double> matrix2;
    matrix2 = std::move(matrix);

    Matrix<double> expected(2, 2);
    expected.setData({5, 0.55, -2, 4.5});

    EXPECT_EQ(matrix2, expected);
}

TEST(matrix_test, operatorEquals_001){

    Matrix<int> matrix(2, 2);
    matrix.setData({1, 2, 3, 4});

    Matrix<int> matrix2(2, 2);
    matrix2.setData({1, 2, 3, 4});

    EXPECT_TRUE(matrix == matrix2);
}

TEST(matrix_test, operatorEquals_002){

    Matrix<int> matrix(2, 2);
    matrix.setData({1, 2, 3, 4});

    Matrix<int> matrix2(2, 2);
    matrix2.setData({1, 2, 3, 5});

    EXPECT_TRUE(matrix != matrix2);
}

TEST(matrix_test, operatorSpaceship_001){

    Matrix<int> matrix(2, 2);
    matrix.setData({1, 2, 3, 4});

    Matrix<int> matrix2(2, 2);
    matrix2.setData({1, 2, 3, 4});

    EXPECT_TRUE((matrix <=> matrix2) == std::partial_ordering::equivalent);
}

TEST(matrix_test, getCoords_001){

    Matrix<int> matrix(2, 3);

    auto coords = matrix.getCoords(4);

    EXPECT_EQ(coords[0], 1);
    EXPECT_EQ(coords[1], 1);
}

TEST(matrix_test, getIndex_001){

    Matrix<int> matrix(2, 3);

    uint64_t index = matrix.getIndex({1, 1});

    EXPECT_EQ(index, 4);
}

TEST(matrix_test, matrixMultiplication_001){

    Matrix<int> matrix(2, 2);
    matrix.setData({1, 2, 3, 4});

    Matrix<int> matrix2(2, 2);
    matrix2.setData({5, 6, 7, 8});

    matrix.matrixMultiplication(matrix2);

    Matrix<int> expected(2, 2);
    expected.setData({
        19, 22,
        43, 50
    });

    EXPECT_EQ(matrix, expected);
}

TEST(matrix_test, inverse_001){

    Matrix<double> matrix(2, 2);
    matrix.setData({1, 2, 3, 4});

    Matrix<double> inverse = matrix.inverse();

    Matrix<double> expected(2, 2);
    expected.setData({
        -2.0,  1.0,
         1.5, -0.5
    });

    constexpr double epsilon = 1e-9;

    for(uint64_t i = 0; i < 2; ++i){
        for(uint64_t j = 0; j < 2; ++j){

            EXPECT_NEAR(
                inverse.getItem(i, j),
                expected.getItem(i, j),
                epsilon
            );
        }
    }
}

TEST(matrix_test, inverseInPlace_001){

    Matrix<double> matrix(2, 2);
    matrix.setData({1, 2, 3, 4});

    matrix.inverseInPlace();

    Matrix<double> expected(2, 2);
    expected.setData({
        -2.0,  1.0,
         1.5, -0.5
    });

    constexpr double epsilon = 1e-9;

    for(uint64_t i = 0; i < 2; ++i){
        for(uint64_t j = 0; j < 2; ++j){

            EXPECT_NEAR(
                matrix.getItem(i, j),
                expected.getItem(i, j),
                epsilon
            );
        }
    }
}