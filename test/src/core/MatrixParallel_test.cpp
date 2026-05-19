#include <bitset>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "TestUtils.hpp"

#define protected public // Hack pro možnost otestovat i privátní funkce
#include "core/MatrixParallel.hpp"
#undef protected

using gema::MatrixParallel;
using gema::LinearContainer;

TEST(matrixparallel_test, constructor_001){

    MatrixParallel<double> matrix = MatrixParallel<double>(2, 3);

    MatrixParallel<double> matrix2 = MatrixParallel<double>(2, 3);

    EXPECT_EQ(matrix.getDimensionSizes(), matrix2.getDimensionSizes());
}

TEST(matrixparallel_test, constructor_002){

    auto matrix = std::make_unique<MatrixParallel<double>>(2, 3);

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

TEST(matrixparallel_test, constructor_003){

    MatrixParallel<double> matrix(2, 2);

    matrix.setData({2, -3, 5, 10});

    MatrixParallel<double> result(matrix);

    EXPECT_EQ(matrix, result);
}

TEST(matrixparallel_test, constructor_004){

    MatrixParallel<double> matrix(2, 2);
    matrix.setData({2, -3, 5, 10});

    MatrixParallel<double> result(std::move(matrix));

    MatrixParallel<double> expected(2, 2);
    expected.setData({2, -3, 5, 10});

    EXPECT_EQ(result, expected);
}

TEST(matrixparallel_test, getItem_001){

    MatrixParallel<int> matrix(2, 3);
    matrix.setData({0, 5, -1, 100, -2, -16});

    EXPECT_EQ(matrix.getItem(0, 0), 0);
    EXPECT_EQ(matrix.getItem(0, 1), 5);
    EXPECT_EQ(matrix.getItem(0, 2), -1);
    EXPECT_EQ(matrix.getItem(1, 0), 100);
    EXPECT_EQ(matrix.getItem(1, 1), -2);
    EXPECT_EQ(matrix.getItem(1, 2), -16);
}

TEST(matrixparallel_test, setItem_001){

    MatrixParallel<double> matrix(2, 2);

    matrix.setItem(5.5, 0, 0);
    matrix.setItem(-2.0, 0, 1);
    matrix.setItem(9.1, 1, 0);
    matrix.setItem(8.8, 1, 1);

    MatrixParallel<double> expected(2, 2);
    expected.setData({5.5, -2.0, 9.1, 8.8});

    EXPECT_EQ(matrix, expected);
}

TEST(matrixparallel_test, setData_001){

    MatrixParallel<int> matrix(2, 2);
    matrix.setData({1, 2, 3, 4});

    MatrixParallel<int> expected(2, 2);
    expected.setData({1, 2, 3, 4});

    EXPECT_EQ(matrix, expected);
}

TEST(matrixparallel_test, fillWith_001){

    MatrixParallel<int> matrix(2, 3);
    matrix.fillWith(69);

    MatrixParallel<int> expected(2, 3);
    expected.setData({69, 69, 69, 69, 69, 69});

    EXPECT_EQ(matrix, expected);
}

TEST(matrixparallel_test, toString_001){

    MatrixParallel<int> matrix(2, 2);
    matrix.setData({1, 2, 3, 4});

    std::string expected = "{{1, 2}, {3, 4}}";

    EXPECT_EQ(matrix.toString(), expected);
}

TEST(matrixparallel_test, transposition_001){

    MatrixParallel<int> matrix(2, 3);
    matrix.setData({0, 5, -1, 100, -2, -16});

    matrix.transposition();

    MatrixParallel<int> expected(3, 2);
    expected.setData({0, 100, 5, -2, -1, -16});

    EXPECT_EQ(matrix, expected);
}

TEST(matrixparallel_test, resize_001){

    MatrixParallel<int> matrix(2, 2);
    matrix.setData({1, 2, 3, 4});

    matrix.resize(3, 2);

    EXPECT_EQ(matrix.getDimensionSizes()[0], 3);
    EXPECT_EQ(matrix.getDimensionSizes()[1], 2);
    EXPECT_EQ(matrix.getNumberOfItems(), 6);
}

TEST(matrixparallel_test, operatorAssign_001){

    MatrixParallel<double> matrix(2, 2);
    matrix.setData({5, 0.55, -2, 4.5});

    MatrixParallel<double> matrix2;
    matrix2 = matrix;

    matrix.setItem(8, 0, 0);

    MatrixParallel<double> expected(2, 2);
    expected.setData({5, 0.55, -2, 4.5});

    EXPECT_EQ(matrix2, expected);
}

TEST(matrixparallel_test, operatorAssign_002){

    MatrixParallel<double> matrix(2, 2);
    matrix.setData({5, 0.55, -2, 4.5});

    MatrixParallel<double> matrix2;
    matrix2 = std::move(matrix);

    MatrixParallel<double> expected(2, 2);
    expected.setData({5, 0.55, -2, 4.5});

    EXPECT_EQ(matrix2, expected);
}

TEST(matrixparallel_test, operatorEquals_001){

    MatrixParallel<int> matrix(2, 2);
    matrix.setData({1, 2, 3, 4});

    MatrixParallel<int> matrix2(2, 2);
    matrix2.setData({1, 2, 3, 4});

    EXPECT_TRUE(matrix == matrix2);
}

TEST(matrixparallel_test, operatorEquals_002){

    MatrixParallel<int> matrix(2, 2);
    matrix.setData({1, 2, 3, 4});

    MatrixParallel<int> matrix2(2, 2);
    matrix2.setData({1, 2, 3, 5});

    EXPECT_TRUE(matrix != matrix2);
}

TEST(matrixparallel_test, operatorSpaceship_001){

    MatrixParallel<int> matrix(2, 2);
    matrix.setData({1, 2, 3, 4});

    MatrixParallel<int> matrix2(2, 2);
    matrix2.setData({1, 2, 3, 4});

    EXPECT_TRUE((matrix <=> matrix2) == std::partial_ordering::equivalent);
}

TEST(matrixparallel_test, getCoords_001){

    MatrixParallel<int> matrix(2, 3);

    auto coords = matrix.getCoords(4);

    EXPECT_EQ(coords[0], 1);
    EXPECT_EQ(coords[1], 1);
}

TEST(matrixparallel_test, getIndex_001){

    MatrixParallel<int> matrix(2, 3);

    uint64_t index = matrix.getIndex({1, 1});

    EXPECT_EQ(index, 4);
}

TEST(matrixparallel_test, matrixMultiplication_001){

    MatrixParallel<int> matrix(2, 2);
    matrix.setData({1, 2, 3, 4});

    MatrixParallel<int> matrix2(2, 2);
    matrix2.setData({5, 6, 7, 8});

    matrix.matrixMultiplication(matrix2);

    MatrixParallel<int> expected(2, 2);
    expected.setData({
        19, 22,
        43, 50
    });

    EXPECT_EQ(matrix, expected);
}

TEST(matrixparallel_test, inverse_001){

    MatrixParallel<double> matrix(2, 2);
    matrix.setData({1, 2, 3, 4});

    // TODO: finish
}

TEST(matrixparallel_test, inverseInPlace_001){

    MatrixParallel<double> matrix(2, 2);
    matrix.setData({1, 2, 3, 4});

    // TODO: finish
}