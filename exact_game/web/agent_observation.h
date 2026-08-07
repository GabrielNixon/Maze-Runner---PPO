#pragma once

#include "enemy.h"
#include "maze.h"
#include "player.h"

#define AGENT_OBSERVATION_SIZE 19
#define AGENT_GRID_CHANNELS 7
#define AGENT_STATS_SIZE 14
#define AGENT_OBSERVATION_FLOATS \
    (AGENT_GRID_CHANNELS * AGENT_OBSERVATION_SIZE * AGENT_OBSERVATION_SIZE + AGENT_STATS_SIZE)

void AgentObservation_Build(
    const MazeBuffer *maze,
    const Player *player,
    const EnemyList *enemies,
    float hunger,
    int last_action,
    float *out
);
