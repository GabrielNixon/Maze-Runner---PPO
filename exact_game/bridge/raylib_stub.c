#include "raylib.h"

static double g_time_seconds = 0.0;
static int g_action = 0;

void MRStub_SetAction(int action) {
    g_action = action;
}

void MRStub_SetTime(double seconds) {
    g_time_seconds = seconds;
}

void MRStub_AdvanceTime(double seconds) {
    g_time_seconds += seconds;
}

double GetTime(void) {
    return g_time_seconds;
}

float GetFrameTime(void) {
    return 1.0f / 60.0f;
}

static int action_has_north(void) {
    return g_action == 1 || g_action == 2 || g_action == 8;
}

static int action_has_east(void) {
    return g_action == 2 || g_action == 3 || g_action == 4;
}

static int action_has_south(void) {
    return g_action == 4 || g_action == 5 || g_action == 6;
}

static int action_has_west(void) {
    return g_action == 6 || g_action == 7 || g_action == 8;
}

int IsKeyDown(int key) {
    if (key == KEY_W || key == KEY_UP) return action_has_north();
    if (key == KEY_D || key == KEY_RIGHT) return action_has_east();
    if (key == KEY_S || key == KEY_DOWN) return action_has_south();
    if (key == KEY_A || key == KEY_LEFT) return action_has_west();
    return 0;
}
