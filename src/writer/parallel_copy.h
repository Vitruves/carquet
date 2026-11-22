#ifndef CARQUET_WRITER_PARALLEL_COPY_H
#define CARQUET_WRITER_PARALLEL_COPY_H

#include <stddef.h>

/* memcpy(dst, src, n), cut into pieces copied by a small helper pool and the
 * calling thread when n is large (see parallel_copy.c). The regions must not
 * overlap. Returns when all of it is copied. */
void carquet_parallel_copy(void* dst, const void* src, size_t n);

/* Stop the helper pool (carquet_cleanup); a later copy starts it again. */
void carquet_parallel_copy_shutdown(void);

#endif /* CARQUET_WRITER_PARALLEL_COPY_H */
