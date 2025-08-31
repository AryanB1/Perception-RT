#include <gtest/gtest.h>
#include <iostream>

// Test categories can be run individually using:
// ./unit_tests --gtest_filter="MetricsTest*"
// ./unit_tests --gtest_filter="PipelineTest*"
// etc.

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "Running FrameKeeper-RT Test Suite" << std::endl;
    std::cout << "=================================" << std::endl;
    
    // Set up any global test configuration here
    // For example, disable CUDA for testing if needed
    
    int result = RUN_ALL_TESTS();
    
    if (result == 0) {
        std::cout << "\nAll tests passed!" << std::endl;
    } else {
        std::cout << "\nTests failed!" << std::endl;
    }
    
    return result;
}
