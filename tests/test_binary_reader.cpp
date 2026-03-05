#include <iostream>
#include <cstdint>
#include <cstdlib>
#include "util.h"

#define ASSERT(cond) \
    if (!(cond)) { \
        std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << ": " << #cond << std::endl; \
        std::exit(1); \
    }

void test_read_u64() {
    std::cout << "Testing read_u64..." << std::endl;

    // Case 1: 0
    uint8_t buf1[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    ASSERT(read_u64(buf1) == 0);

    // Case 2: 1
    uint8_t buf2[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    ASSERT(read_u64(buf2) == 1);

    // Case 3: 0x0102030405060708
    uint8_t buf3[8] = {0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01};
    ASSERT(read_u64(buf3) == 0x0102030405060708ULL);

    // Case 4: Max value
    uint8_t buf4[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    ASSERT(read_u64(buf4) == 0xFFFFFFFFFFFFFFFFULL);

    // Case 5: Pattern with high bits
    uint8_t buf5[8] = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11};
    ASSERT(read_u64(buf5) == 0x1100FFEEDDCCBBAAULL);

    std::cout << "read_u64 passed!" << std::endl;
}

void test_read_int() {
    std::cout << "Testing read_int..." << std::endl;

    // Case 1: 0
    uint8_t buf1[4] = {0, 0, 0, 0};
    ASSERT(read_int(buf1) == 0);

    // Case 2: 1
    uint8_t buf2[4] = {1, 0, 0, 0};
    ASSERT(read_int(buf2) == 1);

    // Case 3: 0x01020304
    uint8_t buf3[4] = {0x04, 0x03, 0x02, 0x01};
    ASSERT(read_int(buf3) == 0x01020304);

    // Case 4: Negative pattern (if treated as signed)
    uint8_t buf4[4] = {0xFF, 0xFF, 0xFF, 0xFF};
    ASSERT(read_int(buf4) == -1);

    // Case 5: 0x7FFFFFFF (max positive int32)
    uint8_t buf5[4] = {0xFF, 0xFF, 0xFF, 0x7F};
    ASSERT(read_int(buf5) == 2147483647);

    // Case 6: 0x80000000 (min negative int32)
    uint8_t buf6[4] = {0x00, 0x00, 0x00, 0x80};
    ASSERT(read_int(buf6) == (int32_t)0x80000000);

    std::cout << "read_int passed!" << std::endl;
}

void test_read_short() {
    std::cout << "Testing read_short..." << std::endl;

    // Case 1: 0
    uint8_t buf1[2] = {0, 0};
    ASSERT(read_short(buf1) == 0);

    // Case 2: 1
    uint8_t buf2[2] = {1, 0};
    ASSERT(read_short(buf2) == 1);

    // Case 3: 0x0102
    uint8_t buf3[2] = {0x02, 0x01};
    ASSERT(read_short(buf3) == 0x0102);

    // Case 4: Max value
    uint8_t buf4[2] = {0xFF, 0xFF};
    ASSERT(read_short(buf4) == 0xFFFF);

    std::cout << "read_short passed!" << std::endl;
}

int main() {
    test_read_u64();
    test_read_int();
    test_read_short();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
