

#include <iostream>
#include <memory>

#include "TestAdvanced.h"
#include "TestAssignmentOperations.h"
#include "TestComparison.h"
#include "TestConstruction.h"
#include "TestIndexing.h"
#include "TestIndexManipulation.h"
#include "TestInformation.h"
#include "TestLinearOperators.h"
#include "TestManipulation.h"
#include "TestScalar.h"
#include "TestSet.h"
#include "TestTensorMultiplication.h"

int main(int, char**) {
    std::vector<std::unique_ptr<TestSet>> test_sets;
    test_sets.emplace_back(std::make_unique<TestScalar>());
    test_sets.emplace_back(std::make_unique<TestLinearOperations>());
    test_sets.emplace_back(std::make_unique<TestIndexing>());
    test_sets.emplace_back(std::make_unique<TestComparison>());
    test_sets.emplace_back(std::make_unique<TestTensorMultiplication>());
    test_sets.emplace_back(std::make_unique<TestAdvanced>());
    test_sets.emplace_back(std::make_unique<TestConstruction>());
    test_sets.emplace_back(std::make_unique<TestIndexManip>());
    test_sets.emplace_back(std::make_unique<TestAssignmentOperations>());
    test_sets.emplace_back(std::make_unique<TestInformation>());
    test_sets.emplace_back(std::make_unique<TestManipulation>());

    int passes{0};
    std::vector<std::string> failures{};

    for (const auto& test_set: test_sets) {
        std::cout << "Running \"" << test_set->name << "\"...\n";
        for (const auto& test: test_set->tests) {
            std::string result{};
            try {
                if (test->run_test()) {
                    result = "Pass";
                    ++passes;
                }
                else {
                    result = "FAIL (returned false)";
                    failures.push_back(test_set->name + ": " + test->name);
                }
            }
            catch(std::exception& exception) {
                std::string what(exception.what());
                result = "FAIL (with std::exception) - " + what;
                failures.push_back(test_set->name + ": " + test->name);
            }
            catch(...) {
                result = "FAIL (with unknown exception)";
                failures.push_back(test_set->name + ": " + test->name);
            }
            std::cout << "\t";
            std::cout.width(30);
            std::cout << std::left << test->name + ":";
            std::cout.width(0);
            std::cout << result << "\n";
        }
        std::cout << "\n";
    }

    std::cout << "-----------------------------------------\n";
    std::cout << "SUMMARY \n";
    std::cout << "-----------------------------------------\n";
    std::cout << passes << " tests passed." << std::endl;
    if (!failures.empty()) {
        std::cout << failures.size() << " tests failed:\n\n";
        for (auto& report_string: failures) std::cout << "\t" << report_string << "\n";
        std::cout << std::endl;
    }
}
