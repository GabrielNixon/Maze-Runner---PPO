#include "agent_input.h"
#include "raylib.h"

static int g_agent_action = 0;

void AgentInput_SetAction(int action) {
    if (action < 0 || action > 8) action = 0;
    g_agent_action = action;
}

static int north(void) {
    return g_agent_action == 1 || g_agent_action == 2 || g_agent_action == 8;
}
static int east(void) {
    return g_agent_action == 2 || g_agent_action == 3 || g_agent_action == 4;
}
static int south(void) {
    return g_agent_action == 4 || g_agent_action == 5 || g_agent_action == 6;
}
static int west(void) {
    return g_agent_action == 6 || g_agent_action == 7 || g_agent_action == 8;
}

int AgentInput_IsKeyDown(int key) {
    if (key == KEY_W || key == KEY_UP) return north();
    if (key == KEY_D || key == KEY_RIGHT) return east();
    if (key == KEY_S || key == KEY_DOWN) return south();
    if (key == KEY_A || key == KEY_LEFT) return west();
    return 0;
}
