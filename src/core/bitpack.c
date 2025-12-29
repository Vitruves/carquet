/**
 * @file bitpack.c
 * @brief Bit packing and unpacking implementation
 *
 * This file contains scalar implementations of bit packing operations.
 * SIMD-optimized versions are in src/simd/
 */

#include "bitpack.h"
#include <string.h>

/* ============================================================================
 * Bit Unpacking - Specialized Functions (1-8 bits)
 * ============================================================================
 */

void carquet_bitunpack8_1bit(const uint8_t* input, uint32_t* values) {
    uint8_t byte = input[0];
    values[0] = (byte >> 0) & 1;
    values[1] = (byte >> 1) & 1;
    values[2] = (byte >> 2) & 1;
    values[3] = (byte >> 3) & 1;
    values[4] = (byte >> 4) & 1;
    values[5] = (byte >> 5) & 1;
    values[6] = (byte >> 6) & 1;
    values[7] = (byte >> 7) & 1;
}

void carquet_bitunpack8_2bit(const uint8_t* input, uint32_t* values) {
    uint16_t v;
    memcpy(&v, input, 2);
    values[0] = (v >> 0) & 0x3;
    values[1] = (v >> 2) & 0x3;
    values[2] = (v >> 4) & 0x3;
    values[3] = (v >> 6) & 0x3;
    values[4] = (v >> 8) & 0x3;
    values[5] = (v >> 10) & 0x3;
    values[6] = (v >> 12) & 0x3;
    values[7] = (v >> 14) & 0x3;
}

void carquet_bitunpack8_3bit(const uint8_t* input, uint32_t* values) {
    uint32_t v;
    memcpy(&v, input, 3);
    values[0] = (v >> 0) & 0x7;
    values[1] = (v >> 3) & 0x7;
    values[2] = (v >> 6) & 0x7;
    values[3] = (v >> 9) & 0x7;
    values[4] = (v >> 12) & 0x7;
    values[5] = (v >> 15) & 0x7;
    values[6] = (v >> 18) & 0x7;
    values[7] = (v >> 21) & 0x7;
}

void carquet_bitunpack8_4bit(const uint8_t* input, uint32_t* values) {
    uint32_t v;
    memcpy(&v, input, 4);
    values[0] = (v >> 0) & 0xF;
    values[1] = (v >> 4) & 0xF;
    values[2] = (v >> 8) & 0xF;
    values[3] = (v >> 12) & 0xF;
    values[4] = (v >> 16) & 0xF;
    values[5] = (v >> 20) & 0xF;
    values[6] = (v >> 24) & 0xF;
    values[7] = (v >> 28) & 0xF;
}

void carquet_bitunpack8_5bit(const uint8_t* input, uint32_t* values) {
    uint64_t v;
    memcpy(&v, input, 5);
    values[0] = (v >> 0) & 0x1F;
    values[1] = (v >> 5) & 0x1F;
    values[2] = (v >> 10) & 0x1F;
    values[3] = (v >> 15) & 0x1F;
    values[4] = (v >> 20) & 0x1F;
    values[5] = (v >> 25) & 0x1F;
    values[6] = (v >> 30) & 0x1F;
    values[7] = (v >> 35) & 0x1F;
}

void carquet_bitunpack8_6bit(const uint8_t* input, uint32_t* values) {
    uint64_t v;
    memcpy(&v, input, 6);
    values[0] = (v >> 0) & 0x3F;
    values[1] = (v >> 6) & 0x3F;
    values[2] = (v >> 12) & 0x3F;
    values[3] = (v >> 18) & 0x3F;
    values[4] = (v >> 24) & 0x3F;
    values[5] = (v >> 30) & 0x3F;
    values[6] = (v >> 36) & 0x3F;
    values[7] = (v >> 42) & 0x3F;
}

void carquet_bitunpack8_7bit(const uint8_t* input, uint32_t* values) {
    uint64_t v;
    memcpy(&v, input, 7);
    values[0] = (v >> 0) & 0x7F;
    values[1] = (v >> 7) & 0x7F;
    values[2] = (v >> 14) & 0x7F;
    values[3] = (v >> 21) & 0x7F;
    values[4] = (v >> 28) & 0x7F;
    values[5] = (v >> 35) & 0x7F;
    values[6] = (v >> 42) & 0x7F;
    values[7] = (v >> 49) & 0x7F;
}

void carquet_bitunpack8_8bit(const uint8_t* input, uint32_t* values) {
    values[0] = input[0];
    values[1] = input[1];
    values[2] = input[2];
    values[3] = input[3];
    values[4] = input[4];
    values[5] = input[5];
    values[6] = input[6];
    values[7] = input[7];
}

/* ============================================================================
 * Bit Unpacking - General Functions
 * ============================================================================
 */

void carquet_bitunpack8_32(const uint8_t* input, int bit_width, uint32_t* values) {
    if (bit_width == 0) {
        memset(values, 0, 8 * sizeof(uint32_t));
        return;
    }

    /* Use specialized functions for common bit widths */
    switch (bit_width) {
        case 1: carquet_bitunpack8_1bit(input, values); return;
        case 2: carquet_bitunpack8_2bit(input, values); return;
        case 3: carquet_bitunpack8_3bit(input, values); return;
        case 4: carquet_bitunpack8_4bit(input, values); return;
        case 5: carquet_bitunpack8_5bit(input, values); return;
        case 6: carquet_bitunpack8_6bit(input, values); return;
        case 7: carquet_bitunpack8_7bit(input, values); return;
        case 8: carquet_bitunpack8_8bit(input, values); return;
    }

    /* General case for 9-32 bits */
    uint32_t mask = (1U << bit_width) - 1;
    int bit_pos = 0;
    int byte_pos = 0;

    for (int i = 0; i < 8; i++) {
        /* Read enough bytes to cover the value */
        uint64_t bits = 0;
        int bits_needed = bit_width;
        int bits_in_buffer = 0;

        while (bits_needed > 0) {
            int bits_from_byte = 8 - (bit_pos % 8);
            if (bits_from_byte > bits_needed) {
                bits_from_byte = bits_needed;
            }

            uint8_t byte_val = input[byte_pos];
            int shift_down = bit_pos % 8;
            uint64_t extracted = (byte_val >> shift_down) & ((1U << bits_from_byte) - 1);
            bits |= extracted << bits_in_buffer;

            bit_pos += bits_from_byte;
            bits_in_buffer += bits_from_byte;
            bits_needed -= bits_from_byte;

            if (bit_pos % 8 == 0) {
                byte_pos++;
            }
        }

        values[i] = (uint32_t)(bits & mask);
    }
}

size_t carquet_bitunpack_32(const uint8_t* input, size_t count,
                            int bit_width, uint32_t* values) {
    if (bit_width == 0) {
        memset(values, 0, count * sizeof(uint32_t));
        return 0;
    }

    size_t bytes_consumed = 0;
    size_t i = 0;

    /* Process groups of 8 */
    for (; i + 8 <= count; i += 8) {
        carquet_bitunpack8_32(input + bytes_consumed, bit_width, values + i);
        bytes_consumed += bit_width;  /* 8 values * bit_width bits = bit_width bytes */
    }

    /* Handle remaining values */
    if (i < count) {
        uint32_t temp[8];
        carquet_bitunpack8_32(input + bytes_consumed, bit_width, temp);
        for (size_t j = 0; j < count - i; j++) {
            values[i + j] = temp[j];
        }
        bytes_consumed += carquet_packed_size(count - i, bit_width);
    }

    return bytes_consumed;
}

/* ============================================================================
 * Bit Packing - General Functions
 * ============================================================================
 */

void carquet_bitpack8_32(const uint32_t* values, int bit_width, uint8_t* output) {
    if (bit_width == 0) {
        return;
    }

    if (bit_width == 8) {
        for (int i = 0; i < 8; i++) {
            output[i] = (uint8_t)values[i];
        }
        return;
    }

    /* General packing */
    memset(output, 0, bit_width);

    uint32_t mask = (1U << bit_width) - 1;
    int bit_pos = 0;

    for (int i = 0; i < 8; i++) {
        uint32_t val = values[i] & mask;
        int byte_pos = bit_pos / 8;
        int bit_offset = bit_pos % 8;

        /* Write value across bytes */
        output[byte_pos] |= (uint8_t)(val << bit_offset);

        int bits_written = 8 - bit_offset;
        if (bits_written < bit_width) {
            val >>= bits_written;
            byte_pos++;

            while (bits_written < bit_width) {
                output[byte_pos] |= (uint8_t)val;
                val >>= 8;
                bits_written += 8;
                byte_pos++;
            }
        }

        bit_pos += bit_width;
    }
}

size_t carquet_bitpack_32(const uint32_t* values, size_t count,
                          int bit_width, uint8_t* output) {
    if (bit_width == 0 || count == 0) {
        return 0;
    }

    size_t bytes_written = 0;
    size_t i = 0;

    /* Process groups of 8 */
    for (; i + 8 <= count; i += 8) {
        carquet_bitpack8_32(values + i, bit_width, output + bytes_written);
        bytes_written += bit_width;
    }

    /* Handle remaining values (pad with zeros) */
    if (i < count) {
        uint32_t temp[8] = {0};
        for (size_t j = 0; j < count - i; j++) {
            temp[j] = values[i + j];
        }
        size_t remaining_bytes = carquet_packed_size(count - i, bit_width);
        carquet_bitpack8_32(temp, bit_width, output + bytes_written);
        bytes_written += remaining_bytes;
    }

    return bytes_written;
}

/* ============================================================================
 * Function Dispatch
 * ============================================================================
 */

static carquet_bitunpack8_fn unpack_functions[33] = {
    NULL,  /* 0 bits */
    carquet_bitunpack8_1bit,
    carquet_bitunpack8_2bit,
    carquet_bitunpack8_3bit,
    carquet_bitunpack8_4bit,
    carquet_bitunpack8_5bit,
    carquet_bitunpack8_6bit,
    carquet_bitunpack8_7bit,
    carquet_bitunpack8_8bit,
    /* 9-32 bits use general function, return NULL */
};

carquet_bitunpack8_fn carquet_get_bitunpack8_fn(int bit_width) {
    if (bit_width < 1 || bit_width > 8) {
        return NULL;
    }
    return unpack_functions[bit_width];
}

carquet_bitpack8_fn carquet_get_bitpack8_fn(int bit_width) {
    /* For now, return NULL - callers should use carquet_bitpack8_32 */
    (void)bit_width;
    return NULL;
}

/* ============================================================================
 * Bit Reader
 * ============================================================================
 */

void carquet_bit_reader_init(carquet_bit_reader_t* reader,
                              const uint8_t* data, size_t size) {
    reader->data = data;
    reader->size = size;
    reader->byte_pos = 0;
    reader->bit_pos = 0;
    reader->buffer = 0;
    reader->buffer_bits = 0;
}

static void refill_buffer(carquet_bit_reader_t* reader) {
    while (reader->buffer_bits <= 56 && reader->byte_pos < reader->size) {
        reader->buffer |= (uint64_t)reader->data[reader->byte_pos++] << reader->buffer_bits;
        reader->buffer_bits += 8;
    }
}

int carquet_bit_reader_read_bit(carquet_bit_reader_t* reader) {
    if (reader->buffer_bits == 0) {
        refill_buffer(reader);
    }
    if (reader->buffer_bits == 0) {
        return -1;  /* No more data */
    }

    int bit = reader->buffer & 1;
    reader->buffer >>= 1;
    reader->buffer_bits--;
    return bit;
}

uint32_t carquet_bit_reader_read_bits(carquet_bit_reader_t* reader, int num_bits) {
    if (num_bits == 0) return 0;
    if (num_bits > 32) num_bits = 32;

    if (reader->buffer_bits < num_bits) {
        refill_buffer(reader);
    }

    uint32_t result = (uint32_t)(reader->buffer & ((1ULL << num_bits) - 1));
    reader->buffer >>= num_bits;
    reader->buffer_bits -= num_bits;
    return result;
}

uint64_t carquet_bit_reader_read_bits64(carquet_bit_reader_t* reader, int num_bits) {
    if (num_bits == 0) return 0;
    if (num_bits > 64) num_bits = 64;

    if (num_bits <= 32) {
        return carquet_bit_reader_read_bits(reader, num_bits);
    }

    /* Read in two parts */
    uint64_t low = carquet_bit_reader_read_bits(reader, 32);
    uint64_t high = carquet_bit_reader_read_bits(reader, num_bits - 32);
    return low | (high << 32);
}

bool carquet_bit_reader_has_more(const carquet_bit_reader_t* reader) {
    return reader->buffer_bits > 0 || reader->byte_pos < reader->size;
}

size_t carquet_bit_reader_remaining_bits(const carquet_bit_reader_t* reader) {
    return (size_t)reader->buffer_bits +
           (reader->size - reader->byte_pos) * 8;
}

/* ============================================================================
 * Bit Writer
 * ============================================================================
 */

void carquet_bit_writer_init(carquet_bit_writer_t* writer,
                              uint8_t* data, size_t capacity) {
    writer->data = data;
    writer->capacity = capacity;
    writer->byte_pos = 0;
    writer->bit_pos = 0;
    writer->buffer = 0;
    writer->buffer_bits = 0;
}

static void flush_buffer(carquet_bit_writer_t* writer) {
    while (writer->buffer_bits >= 8 && writer->byte_pos < writer->capacity) {
        writer->data[writer->byte_pos++] = (uint8_t)(writer->buffer);
        writer->buffer >>= 8;
        writer->buffer_bits -= 8;
    }
}

void carquet_bit_writer_write_bit(carquet_bit_writer_t* writer, int bit) {
    writer->buffer |= (uint64_t)(bit & 1) << writer->buffer_bits;
    writer->buffer_bits++;

    if (writer->buffer_bits >= 56) {
        flush_buffer(writer);
    }
}

void carquet_bit_writer_write_bits(carquet_bit_writer_t* writer,
                                    uint32_t value, int num_bits) {
    if (num_bits == 0) return;
    if (num_bits > 32) num_bits = 32;

    uint32_t mask = num_bits == 32 ? ~0U : (1U << num_bits) - 1;
    writer->buffer |= (uint64_t)(value & mask) << writer->buffer_bits;
    writer->buffer_bits += num_bits;

    if (writer->buffer_bits >= 56) {
        flush_buffer(writer);
    }
}

void carquet_bit_writer_write_bits64(carquet_bit_writer_t* writer,
                                      uint64_t value, int num_bits) {
    if (num_bits == 0) return;
    if (num_bits > 64) num_bits = 64;

    if (num_bits <= 32) {
        carquet_bit_writer_write_bits(writer, (uint32_t)value, num_bits);
        return;
    }

    /* Write in two parts */
    carquet_bit_writer_write_bits(writer, (uint32_t)value, 32);
    carquet_bit_writer_write_bits(writer, (uint32_t)(value >> 32), num_bits - 32);
}

void carquet_bit_writer_flush(carquet_bit_writer_t* writer) {
    /* Flush complete bytes */
    flush_buffer(writer);

    /* Write any remaining partial byte */
    if (writer->buffer_bits > 0 && writer->byte_pos < writer->capacity) {
        writer->data[writer->byte_pos++] = (uint8_t)(writer->buffer);
        writer->buffer = 0;
        writer->buffer_bits = 0;
    }
}

size_t carquet_bit_writer_bytes_written(const carquet_bit_writer_t* writer) {
    return writer->byte_pos;
}
