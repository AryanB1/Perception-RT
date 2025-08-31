#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include "types.hpp"

// Test basic Frame structure
class FrameTest : public ::testing::Test {
protected:
    void SetUp() override {
        frame = Frame{};
    }

    Frame frame;
};

TEST_F(FrameTest, DefaultConstruction) {
    EXPECT_EQ(frame.id, 0);
    // t_capture should be default-initialized (epoch)
}

TEST_F(FrameTest, IdAssignment) {
    frame.id = 12345;
    EXPECT_EQ(frame.id, 12345);
}

TEST_F(FrameTest, TimestampAssignment) {
    auto now = Clock::now();
    frame.t_capture = now;
    EXPECT_EQ(frame.t_capture, now);
}

// Test StageTimings structure
class StageTimingsTest : public ::testing::Test {
protected:
    void SetUp() override {
        timings = StageTimings{};
    }

    StageTimings timings;
};

TEST_F(StageTimingsTest, DefaultConstruction) {
    EXPECT_DOUBLE_EQ(timings.pre_ms, 0.0);
    EXPECT_DOUBLE_EQ(timings.infer_ms, 0.0);
    EXPECT_DOUBLE_EQ(timings.post_ms, 0.0);
    EXPECT_DOUBLE_EQ(timings.e2e_ms, 0.0);
    EXPECT_FALSE(timings.missed);
}

TEST_F(StageTimingsTest, TimingAssignment) {
    timings.pre_ms = 1.5;
    timings.infer_ms = 2.8;
    timings.post_ms = 0.7;
    timings.e2e_ms = 5.0;
    timings.missed = true;

    EXPECT_DOUBLE_EQ(timings.pre_ms, 1.5);
    EXPECT_DOUBLE_EQ(timings.infer_ms, 2.8);
    EXPECT_DOUBLE_EQ(timings.post_ms, 0.7);
    EXPECT_DOUBLE_EQ(timings.e2e_ms, 5.0);
    EXPECT_TRUE(timings.missed);
}

// Test DeadlineProfile structure
class DeadlineProfileTest : public ::testing::Test {
protected:
    void SetUp() override {
        profile = DeadlineProfile{};
    }

    DeadlineProfile profile;
};

TEST_F(DeadlineProfileTest, DefaultValues) {
    EXPECT_EQ(profile.target_fps, 30);
    EXPECT_DOUBLE_EQ(profile.budget_ms, 33.0);
    EXPECT_DOUBLE_EQ(profile.switch_up_p95_ms, 28.0);
    EXPECT_DOUBLE_EQ(profile.switch_down_p95_ms, 22.0);
    EXPECT_EQ(profile.hysteresis_frames, 90);
}

TEST_F(DeadlineProfileTest, CustomValues) {
    profile.target_fps = 60;
    profile.budget_ms = 16.67;
    profile.switch_up_p95_ms = 14.0;
    profile.switch_down_p95_ms = 11.0;
    profile.hysteresis_frames = 180;

    EXPECT_EQ(profile.target_fps, 60);
    EXPECT_DOUBLE_EQ(profile.budget_ms, 16.67);
    EXPECT_DOUBLE_EQ(profile.switch_up_p95_ms, 14.0);
    EXPECT_DOUBLE_EQ(profile.switch_down_p95_ms, 11.0);
    EXPECT_EQ(profile.hysteresis_frames, 180);
}

TEST_F(DeadlineProfileTest, ConsistencyChecks) {
    // Ensure reasonable values for 30 FPS
    EXPECT_LE(profile.switch_up_p95_ms, profile.budget_ms);
    EXPECT_LE(profile.switch_down_p95_ms, profile.switch_up_p95_ms);
    EXPECT_GT(profile.target_fps, 0);
    EXPECT_GT(profile.hysteresis_frames, 0);
}

// Test ModelPrecision enum
TEST(ModelPrecisionTest, EnumValues) {
    ModelPrecision fp16 = ModelPrecision::FP16;
    ModelPrecision int8 = ModelPrecision::INT8;
    
    EXPECT_NE(fp16, int8);
    
    // Test assignment
    ModelPrecision precision = ModelPrecision::FP16;
    EXPECT_EQ(precision, ModelPrecision::FP16);
    
    precision = ModelPrecision::INT8;
    EXPECT_EQ(precision, ModelPrecision::INT8);
}

// Test PrecisionPlan structure
class PrecisionPlanTest : public ::testing::Test {
protected:
    void SetUp() override {
        plan = PrecisionPlan{};
    }

    PrecisionPlan plan;
};

TEST_F(PrecisionPlanTest, DefaultValues) {
    EXPECT_EQ(plan.precision, ModelPrecision::FP16);
    EXPECT_TRUE(plan.reason.empty());
}

TEST_F(PrecisionPlanTest, CustomValues) {
    plan.precision = ModelPrecision::INT8;
    plan.reason = "Performance optimization";
    
    EXPECT_EQ(plan.precision, ModelPrecision::INT8);
    EXPECT_EQ(plan.reason, "Performance optimization");
}

// Test StreamPlan structure
class StreamPlanTest : public ::testing::Test {
protected:
    void SetUp() override {
        plan = StreamPlan{};
    }

    StreamPlan plan;
};

TEST_F(StreamPlanTest, DefaultValues) {
    EXPECT_EQ(plan.preprocess_priority, -1);
    EXPECT_EQ(plan.infer_priority, -2);
}

TEST_F(StreamPlanTest, CustomPriorities) {
    plan.preprocess_priority = 10;
    plan.infer_priority = 20;
    
    EXPECT_EQ(plan.preprocess_priority, 10);
    EXPECT_EQ(plan.infer_priority, 20);
}

TEST_F(StreamPlanTest, ValidPriorityRange) {
    // Test extreme values
    plan.preprocess_priority = std::numeric_limits<int>::min();
    plan.infer_priority = std::numeric_limits<int>::max();
    
    EXPECT_EQ(plan.preprocess_priority, std::numeric_limits<int>::min());
    EXPECT_EQ(plan.infer_priority, std::numeric_limits<int>::max());
}

// Test Clock and TimePoint functionality
TEST(ClockTest, TimePointBasics) {
    auto t1 = Clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    auto t2 = Clock::now();
    
    EXPECT_LT(t1, t2);
    
    auto duration = t2 - t1;
    EXPECT_GE(duration.count(), 0);
}

TEST(ClockTest, TimePointComparison) {
    auto base_time = Clock::now();
    auto later_time = base_time + std::chrono::milliseconds(100);
    
    EXPECT_LT(base_time, later_time);
    EXPECT_GT(later_time, base_time);
    EXPECT_EQ(base_time, base_time);
}

// Integration tests for timing calculations
TEST(TimingIntegrationTest, EndToEndTiming) {
    StageTimings timings;
    timings.pre_ms = 1.2;
    timings.infer_ms = 3.5;
    timings.post_ms = 0.8;
    
    // Calculate expected e2e time
    double expected_e2e = timings.pre_ms + timings.infer_ms + timings.post_ms;
    timings.e2e_ms = expected_e2e;
    
    EXPECT_DOUBLE_EQ(timings.e2e_ms, 5.5);
    
    // Check if timing is within deadline
    DeadlineProfile profile;
    timings.missed = (timings.e2e_ms > profile.budget_ms);
    
    EXPECT_FALSE(timings.missed);  // 5.5ms < 33.0ms budget
}

TEST(TimingIntegrationTest, MissedDeadline) {
    StageTimings timings;
    timings.e2e_ms = 35.0;  // Over 33ms budget
    
    DeadlineProfile profile;
    timings.missed = (timings.e2e_ms > profile.budget_ms);
    
    EXPECT_TRUE(timings.missed);
}
