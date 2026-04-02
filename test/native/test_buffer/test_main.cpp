#include <gtest/gtest.h>
#include "buffer.h"

// Tests for RingBuffer<N> — the circular buffer that holds sensor readings.
// Each TEST block is independent; a fresh buffer is created for each one.

TEST(RingBuffer, EmptyBuffer) {
    RingBuffer<4> buf;
    EXPECT_EQ(0, buf.count());
    EXPECT_FALSE(buf.full());
}

TEST(RingBuffer, PushAndCount) {
    RingBuffer<4> buf;
    Reading r{};
    buf.push(r); buf.push(r);
    EXPECT_EQ(2, buf.count());
}

TEST(RingBuffer, Full) {
    RingBuffer<3> buf;
    Reading r{};
    buf.push(r); buf.push(r); buf.push(r);
    EXPECT_TRUE(buf.full());
}

TEST(RingBuffer, OrderingNoWrap) {
    // Verify oldest-to-newest ordering when the buffer hasn't wrapped yet
    RingBuffer<4> buf;
    Reading r{};
    r.pm2_5 = 1.0f; buf.push(r);
    r.pm2_5 = 2.0f; buf.push(r);
    r.pm2_5 = 3.0f; buf.push(r);
    EXPECT_FLOAT_EQ(1.0f, buf[0].pm2_5);  // oldest
    EXPECT_FLOAT_EQ(3.0f, buf[2].pm2_5);  // newest
}

TEST(RingBuffer, OrderingWithWrap) {
    // After the buffer wraps, index 0 should still be the oldest remaining value
    RingBuffer<3> buf;
    Reading r{};
    r.pm2_5 = 1.0f; buf.push(r);
    r.pm2_5 = 2.0f; buf.push(r);
    r.pm2_5 = 3.0f; buf.push(r);
    r.pm2_5 = 4.0f; buf.push(r); // overwrites 1.0 — oldest is now 2.0
    EXPECT_EQ(3, buf.count());
    EXPECT_FLOAT_EQ(2.0f, buf[0].pm2_5);  // oldest surviving value
    EXPECT_FLOAT_EQ(4.0f, buf[2].pm2_5);  // newest
}

TEST(RingBuffer, Clear) {
    RingBuffer<4> buf;
    Reading r{};
    buf.push(r); buf.push(r);
    buf.clear();
    EXPECT_EQ(0, buf.count());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    if (RUN_ALL_TESTS()) ;
    return 0;
}
