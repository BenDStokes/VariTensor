#ifndef TEST_BULK_H
#define TEST_BULK_H

#include <memory>

#include "TestSet.h"
#include "varitensor/impl/bulk.h"

// =================================================================================================
//                                                                                 broadcast_vec() |
// =================================================================================================

inline bool broadcast_vec_test(const size_t size1, const size_t size2) {
    // given
    const auto a = impl::allocate(size1);
    for (size_t i=0; i<size1; i++) a.get()[i] = static_cast<double>(i);
    const auto b = impl::allocate(size2);

    // when
    impl::broadcast_vec(a.get(), b.get(), size1, size2);

    // then
    bool result = true;
    for (size_t i=0; i<size2; i++) {
        if (b.get()[i] != static_cast<double>(i%size1)) result = false;
    }

    return result;
}

// -------------------------------------------------------------------------------------------------

struct BroadcastVec4to4 final: TestSet::Test {
    explicit BroadcastVec4to4(): Test("Broadcast Vec 4->4") {}

    bool run_test() override {
        // no padding, size1 == REG_WIDTH_256, size2 == REG_WIDTH_256
        return broadcast_vec_test(4, 4);
    }
};

struct BroadcastVec4to8 final: TestSet::Test {
    explicit BroadcastVec4to8(): Test("Broadcast Vec 4->8") {}

    bool run_test() override {
        // no padding, size1 == REG_WIDTH_256, size2 > REG_WIDTH_256
        return broadcast_vec_test(4, 8);
    }
};

struct BroadcastVec8to16 final: TestSet::Test {
    explicit BroadcastVec8to16(): Test("Broadcast Vec 8->16") {}

    bool run_test() override {
        // no padding, size1 > REG_WIDTH_256, size2 > REG_WIDTH_256
        return broadcast_vec_test(8, 16);
    }
};

struct BroadcastVec8to8 final: TestSet::Test {
    explicit BroadcastVec8to8(): Test("Broadcast Vec 8->8") {}

    bool run_test() override {
        // no padding, size1 > REG_WIDTH_256, size2 > REG_WIDTH_256, size2 == size1
        return broadcast_vec_test(8, 8);
    }
};

struct BroadcastVec3to3 final: TestSet::Test {
    explicit BroadcastVec3to3(): Test("Broadcast Vec 3->3") {}

    bool run_test() override {
        // padding, size1 < REG_WIDTH_256, size_2 < REG_WIDTH_256
        return broadcast_vec_test(3, 3);
    }
};

struct BroadcastVec2to4 final: TestSet::Test {
    explicit BroadcastVec2to4(): Test("Broadcast Vec 2->4") {}

    bool run_test() override {
        // padding, size1 < REG_WIDTH_256, size_2 == REG_WIDTH_256
        return broadcast_vec_test(2, 4);
    }
};

struct BroadcastVec3to6 final: TestSet::Test {
    explicit BroadcastVec3to6(): Test("Broadcast Vec 3->6") {}

    bool run_test() override {
        // padding, size1 < REG_WIDTH_256, size_2 > REG_WIDTH_256
        return broadcast_vec_test(3, 6);
    }
};

struct BroadcastVec9to27 final: TestSet::Test {
    explicit BroadcastVec9to27(): Test("Broadcast Vec 9->27") {}

    bool run_test() override {
        // padding, size1 > REG_WIDTH_256, size_2 > REG_WIDTH_256
        return broadcast_vec_test(9, 27);
    }
};

// =================================================================================================
//                                                                              broadcast_chunks() |
// =================================================================================================

inline bool broadcast_chunks_test(const size_t size1, const size_t size2) {
    // given
    auto a = impl::allocate(size1);
    for (size_t i=0; i<size1; i++) a.get()[i] = static_cast<double>(i);

    auto b = impl::allocate(size2);
    for (size_t i=0; i<size2; i++) b.get()[i] = 1;

    // when
    impl::broadcast_chunks(a.get(), b.get(), size1, size2/size1);

    // then
    bool result = true;
    for (size_t i=0; i<size2; i++) {
        if (b.get()[i] != static_cast<double>(i/size1)) result = false;
    }

    return result;
}

// -------------------------------------------------------------------------------------------------

struct BroadcastChunks2to4 final: TestSet::Test {
    explicit BroadcastChunks2to4(): Test("Broadcast chunks 2->4") {}

    bool run_test() override {
        // padding, size1 < REG_WIDTH_256, size_2 == REG_WIDTH_256
        return broadcast_chunks_test(2, 4);
    }
};

struct BroadcastChunks4to16 final: TestSet::Test {
    explicit BroadcastChunks4to16(): Test("Broadcast chunks 4->16") {}

    bool run_test() override {
        // padding, size1 == REG_WIDTH_256, size_2 >  REG_WIDTH_256
        return broadcast_chunks_test(4, 16);
    }
};

struct BroadcastChunks3to9 final: TestSet::Test {
    explicit BroadcastChunks3to9(): Test("Broadcast chunks 3->9") {}

    bool run_test() override {
        // padding, size1 < REG_WIDTH_256, size_2 >  REG_WIDTH_256
        return broadcast_chunks_test(3, 9);
    }
};

struct BroadcastChunks5to25 final: TestSet::Test {
    explicit BroadcastChunks5to25(): Test("Broadcast chunks 5->25") {}

    bool run_test() override {
        // padding, size1 > REG_WIDTH_256, size_2 >  REG_WIDTH_256
        return broadcast_chunks_test(5, 25);
    }
};

struct TestBulk final: TestSet {
    explicit TestBulk(): TestSet("Test Bulk Operations") {
        add_sub_test(std::make_unique<BroadcastVec4to4>());
        add_sub_test(std::make_unique<BroadcastVec4to8>());
        add_sub_test(std::make_unique<BroadcastVec8to16>());
        add_sub_test(std::make_unique<BroadcastVec8to8>());
        add_sub_test(std::make_unique<BroadcastVec3to3>());
        add_sub_test(std::make_unique<BroadcastVec2to4>());
        add_sub_test(std::make_unique<BroadcastVec3to6>());
        add_sub_test(std::make_unique<BroadcastVec9to27>());

        add_sub_test(std::make_unique<BroadcastChunks2to4>());
        add_sub_test(std::make_unique<BroadcastChunks4to16>());
        add_sub_test(std::make_unique<BroadcastChunks3to9>());
        add_sub_test(std::make_unique<BroadcastChunks5to25>());
    }
};

#endif