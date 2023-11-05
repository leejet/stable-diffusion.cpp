#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"
#include <stdio.h>
#include <cstdlib>
#include "zip.h"

int32_t readInt(uint8_t* buffer) {
    int value = 0;
    value |= ((uint8_t)buffer[3] << 24); // Byte más significativo
    value |= ((uint8_t)buffer[2] << 16);
    value |= ((uint8_t)buffer[1] << 8);
    value |= (uint8_t)buffer[0]; // Byte menos significativo
    return value;
}

int16_t readShort(uint8_t* buffer) {
    int value = 0;
    value |= ((uint8_t)buffer[1] << 8);
    value |= (uint8_t)buffer[0]; // Byte menos significativo
    return value;
}

int8_t findChar(uint8_t* buffer, char c) {
    for(int8_t len = 0; len < 32; len++) {
        if(buffer[len] == c) {
            return len;
        }
    }
    return -1;
}

void readPickle(uint8_t* buffer, size_t size) {
    if(buffer[0] == 0x80) { // proto
        printf("Protocol: %i\n", buffer[1]);
        char string_buffer [32];
        bool finish = false;
        while(!finish) {
            uint8_t type = *buffer;
            buffer++;
            switch (type)
            {
            case '}':
                printf("Empty Dict\n");
                break;
            case ']':
                printf("Empty List\n");
                break;
            case 'h':
            case 'q':
                printf("Byte input: %i\n", *buffer);
                buffer++;
                break;
            case 0x95:
                buffer += 8;
            break;
            case 0x94:
                printf("memoize\n");
            break;
            case 'M':
                {
                    uint16_t val = readShort(buffer);
                    printf("Short input: %i\n", val);
                    buffer += 2;
                }
                break;
            case '(':
                printf("MARK\n");
                break;
            case 'X':
                {
                    const int32_t len = readInt(buffer);
                    buffer += 4;
                    memset(string_buffer, 0, 32);
                    memcpy(string_buffer, buffer, len);
                    buffer += len;
                    printf("String: '%s'\n", string_buffer);
                }
                break;
            case 0x8C:
                {
                    const int8_t len = *buffer;
                    buffer ++;
                    memset(string_buffer, 0, 32);
                    memcpy(string_buffer, buffer, len);
                    buffer += len;
                    printf("String: '%s'\n", string_buffer);
                }
                break;
            case 'c':
                {
                    int8_t len = findChar(buffer, '\n');
                    char module_[32];
                    memset(module_, 0, 32);
                    memcpy(module_, buffer, len);
                    buffer += len + 1;
                    char name_[32];
                    len = findChar(buffer, '\n');
                    memset(name_, 0, 32);
                    memcpy(name_, buffer, len);
                    buffer += len + 1;
                    printf("Global: %s.%s\n", module_, name_);
                }
                break;
            case 't':
                printf("truple\n");
                break;
            case '.':
            finish = true;
            return;
            default:
                break;
            }
        }
    }
}

// support safetensors and ckpt (pikle)

int main(int argc, const char* argv[]) {
    const char* data_pkl = "pickle.pkl";
    struct zip_t *zip = zip_open("pickle.pt", 0, 'r');
    {
        int i, n = zip_entries_total(zip);
        for (i = 0; i < n; ++i) {
            zip_entry_openbyindex(zip, i);
            {
                const char *name = zip_entry_name(zip);
                int isdir = zip_entry_isdir(zip);
                unsigned long long size = zip_entry_size(zip);
                unsigned int crc32 = zip_entry_crc32(zip);
                const char* res = strstr(name, data_pkl);
                if(res) {
                    void* pkl_data = NULL;
                    size_t pkl_size;
                    zip_entry_read(zip, &pkl_data, &pkl_size);
                    printf("Reading pickle: %s\n", name);
                    readPickle((uint8_t*)pkl_data, pkl_size);
                }
            }
            zip_entry_close(zip);
        }
    }
    zip_close(zip);
}