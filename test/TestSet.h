#ifndef TEST_SET_H
#define TEST_SET_H

#include <string>

#include "varitensor/Tensor.h"

using namespace varitensor;

struct TestSet {
    struct Test {
        std::string name;

        explicit Test(std::string name_): name{std::move(name_)} {}
        Test(const Test&) = default;
        virtual ~Test() = default;

        virtual bool run_test() = 0;
    };

    std::vector<std::unique_ptr<Test>> tests{};
    std::string name;

    explicit TestSet(std::string name_): name{std::move(name_)} {}
    TestSet(const TestSet&) = default;
    virtual ~TestSet() = default;

    void add_sub_test(std::unique_ptr<Test> test) {tests.push_back(std::move(test));}
};

#endif
