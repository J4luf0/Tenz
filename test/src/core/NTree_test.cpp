#include <bitset>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "TestUtils.hpp"

#define protected public // Hack pro možnost otestovat i privátní funkce
#include "core/NTree.hpp"
#undef protected

using gema::NTree;

TEST(ntree_test, constructor_001){

    NTree<int> nTree{{2, 3, 3}, {{2}, {2, 3}, {3, 1, 2}}};
}