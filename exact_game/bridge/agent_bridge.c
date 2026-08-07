#include "agent_bridge.h"

#include "draw_tool.h"
#include "enemy.h"
#include "maze.h"
#include "player.h"
#include "raylib.h"
#include "wfc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define HUNGER_MAX 1.0f
#define HUNGER_DECAY_RATE 0.1f
#define FIXED_DT (1.0f / 60.0f)
#define OBS_HALF (MR_OBSERVATION_SIZE / 2)
#define ABI_VERSION 1u

static DrawTool g_draw;
static WFCData g_wfc;
static MazeBuffer g_maze;
static Player g_player;
static EnemyList g_enemies;

static float g_hunger = HUNGER_MAX;
static float g_score_time = 0.0f;
static float g_spike_last_damage_time = -999.0f;
static int g_score_orbs = 0;
static int g_done = 0;
static int g_death_reason = 0;
static int g_last_action = 0;
static int g_last_events = 0;
static int g_curriculum = 3;

static int enemy_count(void) {
    int count = 0;
    for (int i = 0; i < MAX_ENEMIES; i++)
        if (g_enemies.enemies[i].active) count++;
    return count;
}

static void spawn_visible_enemies(void) {
    if (g_curriculum < 3) return;
    float spx[MAX_ENEMIES], spy[MAX_ENEMIES];
    int n = Maze_DrainEnemySpawns(
        &g_maze, g_player.x, g_player.y, VISION_RADIUS,
        spx, spy, MAX_ENEMIES
    );
    for (int i = 0; i < n; i++)
        EnemyList_Spawn(&g_enemies, spx[i], spy[i]);
}

void mr_set_curriculum(int level) {
    if (level < 0) level = 0;
    if (level > 3) level = 3;
    g_curriculum = level;
}

int mr_reset(uint32_t seed) {
    srand((unsigned int)seed);
    MRStub_SetTime(0.0);
    MRStub_SetAction(0);

    DrawTool_Init(&g_draw);
    DrawTool_Randomize(&g_draw);
    DrawTool_Randomize(&g_draw);
    DrawTool_Randomize(&g_draw);

    WFC_Init(&g_wfc, &g_draw.pixels[0][0], CANVAS_SIZE, CANVAS_SIZE);
    if (!WFC_HasFloorPattern(&g_wfc)) {
        DrawTool_FillDefault(&g_draw);
        WFC_Init(&g_wfc, &g_draw.pixels[0][0], CANVAS_SIZE, CANVAS_SIZE);
    }

    float start_x = 0.0f, start_y = 0.0f;
    Maze_Init(&g_maze, &g_wfc, start_x, start_y);
    Maze_GetStartPos(&g_maze, &start_x, &start_y);
    Player_Init(&g_player, start_x, start_y);
    EnemyList_Init(&g_enemies);
    spawn_visible_enemies();

    g_hunger = HUNGER_MAX;
    g_score_time = 0.0f;
    g_score_orbs = 0;
    g_spike_last_damage_time = -999.0f;
    g_done = 0;
    g_death_reason = 0;
    g_last_action = 0;
    g_last_events = 0;
    return 1;
}

static void simulate_frame(int action) {
    if (g_done) return;

    float before_x = g_player.x;
    float before_y = g_player.y;
    MRStub_SetAction(action);
    MRStub_AdvanceTime(FIXED_DT);

    Player_Update(&g_player, &g_maze, FIXED_DT);
    Maze_Update(&g_maze, g_player.x, g_player.y);
    spawn_visible_enemies();

    if (g_curriculum >= 3) {
        EnemyList_CullOutOfBounds(
            &g_enemies, &g_maze, g_player.x, g_player.y, VISION_RADIUS
        );
        EnemyList_Update(&g_enemies, &g_maze, g_player.x, g_player.y, FIXED_DT);
    } else {
        EnemyList_Init(&g_enemies);
    }

    g_score_time += FIXED_DT;
    if (g_curriculum >= 1) {
        g_hunger -= HUNGER_DECAY_RATE * FIXED_DT;
        if (g_hunger < 0.0f) g_hunger = 0.0f;
    }

    int player_tx = (int)floorf(g_player.x / TILE_SIZE);
    int player_ty = (int)floorf(g_player.y / TILE_SIZE);
    if (g_curriculum >= 1 && Maze_TryCollectOrb(&g_maze, player_tx, player_ty)) {
        g_hunger += 0.5f;
        if (g_hunger > HUNGER_MAX) g_hunger = HUNGER_MAX;
        g_score_orbs++;
        g_last_events |= MR_EVENT_ORB;
    }

    if (g_curriculum >= 2 && Maze_IsSpikeUp()) {
        int col = player_tx - g_maze.origin_x;
        int row = player_ty - g_maze.origin_y;
        if (col >= 0 && col < BUF_W && row >= 0 && row < BUF_H &&
                g_maze.cells[row][col].has_spike) {
            float now = (float)GetTime();
            if (now - g_spike_last_damage_time > 1.5f) {
                g_hunger -= 0.5f;
                if (g_hunger < 0.0f) g_hunger = 0.0f;
                g_spike_last_damage_time = now;
                g_last_events |= MR_EVENT_SPIKE;
            }
        }
        if (g_curriculum >= 3)
            EnemyList_KillOnSpikes(&g_enemies, &g_maze);
    }

    if (g_curriculum >= 3 &&
            EnemyList_CheckPlayerCollision(&g_enemies, g_player.x, g_player.y)) {
        g_done = 1;
        g_death_reason = 1;
    } else if (g_curriculum >= 1 && g_hunger <= 0.0f) {
        g_done = 1;
        g_death_reason = 2;
    }

    float moved_x = g_player.x - before_x;
    float moved_y = g_player.y - before_y;
    if (action != 0 && moved_x * moved_x + moved_y * moved_y < 0.0001f)
        g_last_events |= MR_EVENT_BLOCKED;
}

int mr_step(int action, int frame_skip) {
    if (action < 0 || action > 8) return -1;
    if (frame_skip < 1) frame_skip = 1;
    if (frame_skip > 120) frame_skip = 120;
    g_last_action = action;
    g_last_events = 0;
    for (int frame = 0; frame < frame_skip && !g_done; frame++)
        simulate_frame(action);
    MRStub_SetAction(0);
    return g_done;
}

static int tile_index(int world_tx, int world_ty, int *row, int *col) {
    *col = world_tx - g_maze.origin_x;
    *row = world_ty - g_maze.origin_y;
    return *col >= 0 && *col < BUF_W && *row >= 0 && *row < BUF_H;
}

void mr_get_observation(float *grid_out, float *stats_out) {
    const int plane = MR_OBSERVATION_SIZE * MR_OBSERVATION_SIZE;
    memset(grid_out, 0, sizeof(float) * MR_GRID_CHANNELS * plane);
    memset(stats_out, 0, sizeof(float) * MR_STATS_SIZE);

    int player_tx = (int)floorf(g_player.x / TILE_SIZE);
    int player_ty = (int)floorf(g_player.y / TILE_SIZE);
    float phase_seconds = fmodf((float)GetTime(), SPIKE_PERIOD);
    int spikes_up = phase_seconds >= SPIKE_UP_START;
    int spikes_warning = phase_seconds >= SPIKE_WARN_START && !spikes_up;

    for (int oy = -OBS_HALF; oy <= OBS_HALF; oy++) {
        for (int ox = -OBS_HALF; ox <= OBS_HALF; ox++) {
            int local_row = oy + OBS_HALF;
            int local_col = ox + OBS_HALF;
            int index = local_row * MR_OBSERVATION_SIZE + local_col;
            int row = 0, col = 0;
            if (!tile_index(player_tx + ox, player_ty + oy, &row, &col)) {
                grid_out[index] = 1.0f;
                continue;
            }
            TileCell *cell = &g_maze.cells[row][col];
            grid_out[index] = cell->is_wall ? 1.0f : 0.0f;
            grid_out[plane + index] = cell->has_orb ? 1.0f : 0.0f;
            grid_out[2 * plane + index] = cell->has_spike ? 1.0f : 0.0f;
            if (spikes_up && cell->has_spike)
                grid_out[3 * plane + index] = 1.0f;
            else if (spikes_warning && cell->has_spike)
                grid_out[4 * plane + index] = 1.0f;
        }
    }

    float nearest = 1.0f;
    for (int i = 0; i < MAX_ENEMIES; i++) {
        Enemy *enemy = &g_enemies.enemies[i];
        if (!enemy->active || enemy->dying) continue;
        int enemy_tx = (int)floorf(enemy->x / TILE_SIZE);
        int enemy_ty = (int)floorf(enemy->y / TILE_SIZE);
        int dx = enemy_tx - player_tx;
        int dy = enemy_ty - player_ty;
        int distance = abs(dx) > abs(dy) ? abs(dx) : abs(dy);
        float normalized = (float)distance / (float)OBS_HALF;
        if (normalized < nearest) nearest = normalized;
        int local_col = dx + OBS_HALF;
        int local_row = dy + OBS_HALF;
        if (local_col >= 0 && local_col < MR_OBSERVATION_SIZE &&
                local_row >= 0 && local_row < MR_OBSERVATION_SIZE) {
            int index = local_row * MR_OBSERVATION_SIZE + local_col;
            int channel = enemy->freeze_timer > 0.0f ? 6 : 5;
            grid_out[channel * plane + index] = 1.0f;
        }
    }
    if (nearest > 1.0f) nearest = 1.0f;

    float phase = phase_seconds / SPIKE_PERIOD;
    stats_out[0] = g_hunger;
    stats_out[1] = (sinf(2.0f * PI * phase) + 1.0f) * 0.5f;
    stats_out[2] = (cosf(2.0f * PI * phase) + 1.0f) * 0.5f;
    stats_out[3] = (float)enemy_count() / (float)MAX_ENEMIES;
    stats_out[4] = nearest;
    if (g_last_action >= 0 && g_last_action <= 8)
        stats_out[5 + g_last_action] = 1.0f;
}

int mr_get_events(void) { return g_last_events; }
int mr_is_done(void) { return g_done; }
int mr_get_death_reason(void) { return g_death_reason; }
float mr_get_survival_time(void) { return g_score_time; }
float mr_get_hunger(void) { return g_hunger; }
int mr_get_orbs(void) { return g_score_orbs; }
int mr_get_enemy_count(void) { return enemy_count(); }
float mr_get_spike_phase(void) { return fmodf((float)GetTime(), SPIKE_PERIOD); }
uint32_t mr_upstream_abi_version(void) { return ABI_VERSION; }
