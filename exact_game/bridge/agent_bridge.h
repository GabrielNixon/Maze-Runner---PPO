#pragma once

#include <stdint.h>

#if defined(_WIN32)
#define MR_API __declspec(dllexport)
#else
#define MR_API __attribute__((visibility("default")))
#endif

#define MR_OBSERVATION_SIZE 19
#define MR_GRID_CHANNELS 7
#define MR_STATS_SIZE 14

#define MR_EVENT_ORB      1
#define MR_EVENT_SPIKE    2
#define MR_EVENT_BLOCKED  4

#ifdef __cplusplus
extern "C" {
#endif

MR_API void mr_set_curriculum(int level);
MR_API int mr_reset(uint32_t seed);
MR_API int mr_step(int action, int frame_skip);
MR_API void mr_get_observation(float *grid_out, float *stats_out);
MR_API int mr_get_events(void);
MR_API int mr_is_done(void);
MR_API int mr_get_death_reason(void);
MR_API float mr_get_survival_time(void);
MR_API float mr_get_hunger(void);
MR_API int mr_get_orbs(void);
MR_API int mr_get_enemy_count(void);
MR_API float mr_get_spike_phase(void);
MR_API uint32_t mr_upstream_abi_version(void);

#ifdef __cplusplus
}
#endif
