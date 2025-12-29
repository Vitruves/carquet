/**
 * @file detect.c
 * @brief CPU feature detection
 */

#include <carquet/carquet.h>
#include <string.h>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#endif

/* Linux ARM SVE detection via getauxval */
#if defined(__linux__) && defined(__aarch64__)
#include <sys/auxv.h>
#ifndef HWCAP_SVE
#define HWCAP_SVE (1 << 22)
#endif
#ifndef HWCAP2_SVE2
#define HWCAP2_SVE2 (1 << 1)
#endif
#ifndef AT_HWCAP
#define AT_HWCAP 16
#endif
#endif

static carquet_cpu_info_t g_cpu_info = {0};
static int g_initialized = 0;

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)

static void detect_x86_features(void) {
#if defined(_MSC_VER)
    int info[4];
    __cpuid(info, 0);
    int max_leaf = info[0];

    if (max_leaf >= 1) {
        __cpuid(info, 1);
        g_cpu_info.has_sse2 = (info[3] >> 26) & 1;
        g_cpu_info.has_sse41 = (info[2] >> 19) & 1;
        g_cpu_info.has_sse42 = (info[2] >> 20) & 1;
        g_cpu_info.has_avx = (info[2] >> 28) & 1;
    }

    if (max_leaf >= 7) {
        __cpuidex(info, 7, 0);
        g_cpu_info.has_avx2 = (info[1] >> 5) & 1;
        g_cpu_info.has_avx512f = (info[1] >> 16) & 1;
        g_cpu_info.has_avx512bw = (info[1] >> 30) & 1;
        g_cpu_info.has_avx512vl = (info[1] >> 31) & 1;
        g_cpu_info.has_avx512vbmi = (info[2] >> 1) & 1;
    }
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int eax, ebx, ecx, edx;

    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        g_cpu_info.has_sse2 = (edx >> 26) & 1;
        g_cpu_info.has_sse41 = (ecx >> 19) & 1;
        g_cpu_info.has_sse42 = (ecx >> 20) & 1;
        g_cpu_info.has_avx = (ecx >> 28) & 1;
    }

    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        g_cpu_info.has_avx2 = (ebx >> 5) & 1;
        g_cpu_info.has_avx512f = (ebx >> 16) & 1;
        g_cpu_info.has_avx512bw = (ebx >> 30) & 1;
        g_cpu_info.has_avx512vl = (ebx >> 31) & 1;
        g_cpu_info.has_avx512vbmi = (ecx >> 1) & 1;
    }
#endif
}

#elif defined(__aarch64__) || defined(_M_ARM64)

static void detect_arm_features(void) {
    /* NEON is always available on AArch64 */
    g_cpu_info.has_neon = 1;

    /* SVE detection */
    g_cpu_info.has_sve = 0;
    g_cpu_info.sve_vector_length = 0;

#if defined(__linux__)
    /* Linux: use getauxval to detect SVE */
    unsigned long hwcap = getauxval(AT_HWCAP);
    if (hwcap & HWCAP_SVE) {
        g_cpu_info.has_sve = 1;

        /* Get SVE vector length using RDVL instruction via inline asm */
#if defined(__GNUC__) || defined(__clang__)
#ifdef __ARM_FEATURE_SVE
        uint64_t vl;
        __asm__ volatile("rdvl %0, #1" : "=r"(vl));
        g_cpu_info.sve_vector_length = (int)(vl * 8);  /* Convert bytes to bits */
#else
        /* SVE detected but not compiled with SVE support */
        g_cpu_info.sve_vector_length = 128;  /* Minimum SVE vector length */
#endif
#endif
    }
#elif defined(__APPLE__)
    /* macOS/Apple Silicon: SVE is not available on Apple M-series chips */
    g_cpu_info.has_sve = 0;
    g_cpu_info.sve_vector_length = 0;
#endif
}

#elif defined(__arm__) || defined(_M_ARM)

static void detect_arm_features(void) {
    /* ARMv7 NEON detection would require runtime checks */
    g_cpu_info.has_neon = 0;  /* Conservative default */
}

#endif

carquet_status_t carquet_init(void) {
    if (g_initialized) {
        return CARQUET_OK;
    }

    memset(&g_cpu_info, 0, sizeof(g_cpu_info));

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    detect_x86_features();
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
    detect_arm_features();
#endif

    g_initialized = 1;
    return CARQUET_OK;
}

const carquet_cpu_info_t* carquet_get_cpu_info(void) {
    if (!g_initialized) {
        carquet_init();
    }
    return &g_cpu_info;
}
