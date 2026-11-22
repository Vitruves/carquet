/**
 * @file writer_sink.h
 * @brief Where a file writer's bytes go: a stdio stream or a memory buffer
 *
 * Every byte of a Parquet file is appended through carquet_writer_sink_write():
 * to the FILE* of a path / FILE* / append writer, or, for a buffer writer
 * (carquet_writer_create_buffer), to a growable buffer that becomes the block
 * carquet_writer_get_buffer() hands out, with no temporary file in between.
 *
 * The sink is strictly append-only and has a single writer at a time: the
 * background I/O thread while its jobs run, the caller's thread once they are
 * drained (writer_io_drain() in file_writer.c). The file writer keeps the
 * current offset itself (writer->file_offset); only the append path seeks,
 * and it only exists for files.
 */

#ifndef CARQUET_WRITER_SINK_H
#define CARQUET_WRITER_SINK_H

#include <carquet/error.h>
#include "core/buffer.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

typedef struct carquet_writer_sink {
    FILE* file;             /* destination stream, NULL for a memory sink */
    bool in_memory;         /* bytes go to `mem` */
    carquet_buffer_t mem;   /* memory sink (carquet_mem_* allocated) */
} carquet_writer_sink_t;

static inline void carquet_writer_sink_init(carquet_writer_sink_t* sink, FILE* file) {
    sink->file = file;
    sink->in_memory = false;
    carquet_buffer_init(&sink->mem);
}

/* Append `size` bytes. A stream reports CARQUET_ERROR_FILE_WRITE, the memory
 * sink CARQUET_ERROR_OUT_OF_MEMORY when it cannot grow. A sink with neither
 * (a NULL stream passed to a file writer) fails rather than buffer. */
static inline carquet_status_t carquet_writer_sink_write(
    carquet_writer_sink_t* sink, const void* data, size_t size) {
    if (size == 0) {
        return CARQUET_OK;
    }
    if (sink->file) {
        return fwrite(data, 1, size, sink->file) == size
            ? CARQUET_OK : CARQUET_ERROR_FILE_WRITE;
    }
    if (!sink->in_memory) {
        return CARQUET_ERROR_FILE_WRITE;
    }
    return carquet_buffer_append(&sink->mem, data, size);
}

#endif /* CARQUET_WRITER_SINK_H */
