#pragma once

#include <math.h>
#include <stdint.h>

#ifndef PI
#define PI 3.14159265358979323846f
#endif
#ifndef RAD2DEG
#define RAD2DEG (180.0f / PI)
#endif

typedef struct Vector2 { float x, y; } Vector2;
typedef struct Rectangle { float x, y, width, height; } Rectangle;
typedef struct Color { uint8_t r, g, b, a; } Color;
typedef struct Texture2D {
    unsigned int id;
    int width;
    int height;
    int mipmaps;
    int format;
} Texture2D;

#define WHITE     ((Color){255, 255, 255, 255})
#define BLACK     ((Color){0, 0, 0, 255})
#define LIGHTGRAY ((Color){200, 200, 200, 255})
#define GRAY      ((Color){130, 130, 130, 255})
#define DARKGRAY  ((Color){80, 80, 80, 255})
#define GREEN     ((Color){0, 228, 48, 255})
#define RED       ((Color){230, 41, 55, 255})

#define KEY_W      87
#define KEY_A      65
#define KEY_S      83
#define KEY_D      68
#define KEY_G      71
#define KEY_UP     265
#define KEY_DOWN   264
#define KEY_LEFT   263
#define KEY_RIGHT  262
#define KEY_ESCAPE 256
#define KEY_ENTER  257
#define KEY_SPACE  32

#define MOUSE_BUTTON_LEFT  0
#define MOUSE_BUTTON_RIGHT 1

#ifdef __cplusplus
extern "C" {
#endif

double GetTime(void);
float GetFrameTime(void);
int IsKeyDown(int key);
void MRStub_SetAction(int action);
void MRStub_SetTime(double seconds);
void MRStub_AdvanceTime(double seconds);

#ifdef __cplusplus
}
#endif

static inline int IsKeyPressed(int key) { (void)key; return 0; }
static inline int IsMouseButtonPressed(int button) { (void)button; return 0; }
static inline int IsMouseButtonDown(int button) { (void)button; return 0; }
static inline Vector2 GetMousePosition(void) { return (Vector2){0.0f, 0.0f}; }
static inline int CheckCollisionPointRec(Vector2 point, Rectangle rec) {
    return point.x >= rec.x && point.x <= rec.x + rec.width &&
           point.y >= rec.y && point.y <= rec.y + rec.height;
}
static inline Texture2D LoadTexture(const char *path) {
    (void)path;
    return (Texture2D){0};
}
static inline void UnloadTexture(Texture2D texture) { (void)texture; }
static inline int MeasureText(const char *text, int font_size) {
    int n = 0;
    while (text && text[n]) n++;
    return (n * font_size) / 2;
}

static inline void DrawRectangle(int x, int y, int w, int h, Color c) {
    (void)x; (void)y; (void)w; (void)h; (void)c;
}
static inline void DrawRectangleRec(Rectangle r, Color c) { (void)r; (void)c; }
static inline void DrawRectangleLines(int x, int y, int w, int h, Color c) {
    (void)x; (void)y; (void)w; (void)h; (void)c;
}
static inline void DrawRectangleLinesEx(Rectangle r, float thick, Color c) {
    (void)r; (void)thick; (void)c;
}
static inline void DrawRectanglePro(Rectangle r, Vector2 o, float rot, Color c) {
    (void)r; (void)o; (void)rot; (void)c;
}
static inline void DrawCircle(int x, int y, float radius, Color c) {
    (void)x; (void)y; (void)radius; (void)c;
}
static inline void DrawCircleLines(int x, int y, float radius, Color c) {
    (void)x; (void)y; (void)radius; (void)c;
}
static inline void DrawCircleGradient(int x, int y, float radius, Color c1, Color c2) {
    (void)x; (void)y; (void)radius; (void)c1; (void)c2;
}
static inline void DrawTriangle(Vector2 a, Vector2 b, Vector2 c, Color color) {
    (void)a; (void)b; (void)c; (void)color;
}
static inline void DrawLine(int x1, int y1, int x2, int y2, Color c) {
    (void)x1; (void)y1; (void)x2; (void)y2; (void)c;
}
static inline void DrawRing(Vector2 center, float inner, float outer,
                            float start, float end, int segments, Color c) {
    (void)center; (void)inner; (void)outer; (void)start; (void)end;
    (void)segments; (void)c;
}
static inline void DrawTexturePro(Texture2D t, Rectangle src, Rectangle dst,
                                  Vector2 origin, float rotation, Color tint) {
    (void)t; (void)src; (void)dst; (void)origin; (void)rotation; (void)tint;
}
static inline void DrawText(const char *text, int x, int y, int size, Color c) {
    (void)text; (void)x; (void)y; (void)size; (void)c;
}
