/* Linux counterparts of the Windows process queries that more than one unit makes. Windows units keep their own
 * calls; a C unit that includes this header defines _GNU_SOURCE before its first include (C++ units have it). */
#ifndef CHAINNER_CN_PLATFORM_H
#define CHAINNER_CN_PLATFORM_H
#ifdef __linux__
#include <sched.h>
#include <stdint.h>
#include <sys/sysinfo.h>

/* The CPUs the calling thread may run on (sched_getaffinity), as GetProcessAffinityMask's count; 0 when unknown. */
static inline unsigned int cn_affinity_cpus(void)
{
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof set, &set) != 0) return 0;
    return (unsigned int)CPU_COUNT(&set);
}

/* Total physical memory in bytes (sysinfo), as GlobalMemoryStatusEx's ullTotalPhys; 0 with errno on failure. */
static inline uint64_t cn_physical_memory(void)
{
    struct sysinfo info;
    if (sysinfo(&info) != 0) return 0;
    return (uint64_t)info.totalram * info.mem_unit;
}
#endif
#endif
