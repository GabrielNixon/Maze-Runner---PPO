#include "agent_observation.h"

#include "raylib.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define HALF (AGENT_OBSERVATION_SIZE / 2)

void AgentObservation_Build(
    const MazeBuffer *maze,
    const Player *player,
    const EnemyList *enemies,
    float hunger,
    int last_action,
    float *out
) {
    const int plane = AGENT_OBSERVATION_SIZE * AGENT_OBSERVATION_SIZE;
    float *stats = out + AGENT_GRID_CHANNELS * plane;
    memset(out, 0, sizeof(float) * AGENT_OBSERVATION_FLOATS);

    int player_tx = (int)floorf(player->x / TILE_SIZE);
    int player_ty = (int)floorf(player->y / TILE_SIZE);
    float phase_seconds = fmodf((float)GetTime(), SPIKE_PERIOD);
    int spikes_up = phase_seconds >= SPIKE_UP_START;
    int warning = phase_seconds >= SPIKE_WARN_START && !spikes_up;

    for (int oy = -HALF; oy <= HALF; oy++) {
        for (int ox = -HALF; ox <= HALF; ox++) {
            int local_row = oy + HALF;
            int local_col = ox + HALF;
            int index = local_row * AGENT_OBSERVATION_SIZE + local_col;
            int col = player_tx + ox - maze->origin_x;
            int row = player_ty + oy - maze->origin_y;
            if (col < 0 || col >= BUF_W || row < 0 || row >= BUF_H) {
                out[index] = 1.0f;
                continue;
            }
            const TileCell *cell = &maze->cells[row][col];
            out[index] = cell->is_wall ? 1.0f : 0.0f;
            out[plane + index] = cell->has_orb ? 1.0f : 0.0f;
            out[2 * plane + index] = cell->has_spike ? 1.0f : 0.0f;
            if (spikes_up && cell->has_spike)
                out[3 * plane + index] = 1.0f;
            else if (warning && cell->has_spike)
                out[4 * plane + index] = 1.0f;
        }
    }

    int active_count = 0;
    float nearest = 1.0f;
    for (int i = 0; i < MAX_ENEMIES; i++) {
        const Enemy *enemy = &enemies->enemies[i];
        if (!enemy->active || enemy->dying) continue;
        active_count++;
        int enemy_tx = (int)floorf(enemy->x / TILE_SIZE);
        int enemy_ty = (int)floorf(enemy->y / TILE_SIZE);
        int dx = enemy_tx - player_tx;
        int dy = enemy_ty - player_ty;
        int distance = abs(dx) > abs(dy) ? abs(dx) : abs(dy);
        float normalized = (float)distance / (float)HALF;
        if (normalized < nearest) nearest = normalized;
        int local_col = dx + HALF;
        int local_row = dy + HALF;
        if (local_col >= 0 && local_col < AGENT_OBSERVATION_SIZE &&
                local_row >= 0 && local_row < AGENT_OBSERVATION_SIZE) {
            int index = local_row * AGENT_OBSERVATION_SIZE + local_col;
            int channel = enemy->freeze_timer > 0.0f ? 6 : 5;
            out[channel * plane + index] = 1.0f;
        }
    }
    if (nearest > 1.0f) nearest = 1.0f;

    float phase = phase_seconds / SPIKE_PERIOD;
    stats[0] = hunger;
    stats[1] = (sinf(2.0f * PI * phase) + 1.0f) * 0.5f;
    stats[2] = (cosf(2.0f * PI * phase) + 1.0f) * 0.5f;
    stats[3] = (float)active_count / (float)MAX_ENEMIES;
    stats[4] = nearest;
    if (last_action >= 0 && last_action <= 8)
        stats[5 + last_action] = 1.0f;
}
