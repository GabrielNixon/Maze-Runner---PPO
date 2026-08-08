from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

UPSTREAM_URL = "https://github.com/nadarenator/mazerunner.git"
UPSTREAM_COMMIT = "95f83d1a35717fe4dacbb50fc12a9b57f1575754"
MARKER = "MAZERUNNER_PPO_AGENT_HOOK"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} match, found {count}")
    return text.replace(old, new, 1)


def prepare_clone(path: Path, force: bool) -> None:
    if force and path.exists():
        shutil.rmtree(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--no-checkout", UPSTREAM_URL, str(path)])
    run(["git", "fetch", "--depth", "1", "origin", UPSTREAM_COMMIT], cwd=path)
    run(["git", "checkout", "--detach", UPSTREAM_COMMIT], cwd=path)


def patch_player(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '#include "raylib.h"\n#include <math.h>',
        '#include "raylib.h"\n#include "agent_input.h"\n#include <math.h>',
        "player include",
    )
    replacements = {
        "if (IsKeyDown(KEY_W) || IsKeyDown(KEY_UP))": (
            "if (AgentInput_IsKeyDown(KEY_W) || IsKeyDown(KEY_W) || IsKeyDown(KEY_UP))"
        ),
        "if (IsKeyDown(KEY_S) || IsKeyDown(KEY_DOWN))": (
            "if (AgentInput_IsKeyDown(KEY_S) || IsKeyDown(KEY_S) || IsKeyDown(KEY_DOWN))"
        ),
        "if (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT))": (
            "if (AgentInput_IsKeyDown(KEY_A) || IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT))"
        ),
        "if (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT))": (
            "if (AgentInput_IsKeyDown(KEY_D) || IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT))"
        ),
    }
    for old, new in replacements.items():
        text = replace_once(text, old, new, old)
    path.write_text(text, encoding="utf-8")


def patch_main(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '#include "enemy.h"\n',
        '#include "enemy.h"\n#include "agent_input.h"\n#include "agent_observation.h"\n',
        "main include",
    )
    text = replace_once(
        text,
        "static int        g_lb_modal_pending;\n",
        "static int        g_lb_modal_pending;\n"
        "#if defined(PLATFORM_WEB) && defined(AGENT_MODE)\n"
        f"// {MARKER}\n"
        "static float g_agent_obs[AGENT_OBSERVATION_FLOATS];\n"
        "static float g_agent_accum;\n"
        "static int   g_agent_last_action;\n"
        "static int   g_agent_qualified;\n"
        "#endif\n",
        "agent globals",
    )
    text = replace_once(
        text,
        "static void TransitionToGameOver(void) {\n"
        "    g_state            = STATE_GAMEOVER;\n"
        "    g_lb_modal_pending = 0;",
        "static void TransitionToGameOver(void) {\n"
        "    g_state            = STATE_GAMEOVER;\n"
        "    g_lb_modal_pending = 0;\n"
        "#if defined(PLATFORM_WEB) && defined(AGENT_MODE)\n"
        "    AgentInput_SetAction(0);\n"
        "    EM_ASM({ if (window.agentBridge) window.agentBridge.episodeDone(); });\n"
        "#endif",
        "game-over hook",
    )
    original_leaderboard_block = """        if (EM_ASM_INT({
                return (window.lb_isTop10 && window.lb_isTop10($0, $1)) ? 1 : 0;
            }, time_ms, g_score_orbs)) {
            g_lb_modal_pending = 1;
            EM_ASM({
                if (window.lb_showEntryModal) {
                    var pix = [];
                    for (var i = 0; i < 64; i++) pix.push(HEAPU8[$2 + i]);
                    window.lb_showEntryModal($0, $1, pix);
                }
            }, time_ms, g_score_orbs, &g_draw.pixels[0][0]);
        }
"""
    record_aware_leaderboard_block = """        int should_prompt = EM_ASM_INT({
            return (window.lb_isTop10 && window.lb_isTop10($0, $1)) ? 1 : 0;
        }, time_ms, g_score_orbs);
#if defined(AGENT_MODE)
        if (should_prompt) {
            should_prompt = EM_ASM_INT({
                return (window.lb_isRecord && window.lb_isRecord($0, $1)) ? 1 : 0;
            }, time_ms, g_score_orbs);
            if (should_prompt) g_agent_qualified = 1;
        }
#endif
        if (should_prompt) {
            g_lb_modal_pending = 1;
            EM_ASM({
                if (window.lb_showEntryModal) {
                    var pix = [];
                    for (var i = 0; i < 64; i++) pix.push(HEAPU8[$2 + i]);
                    window.lb_showEntryModal($0, $1, pix);
                }
            }, time_ms, g_score_orbs, &g_draw.pixels[0][0]);
        }
"""
    text = replace_once(
        text,
        original_leaderboard_block,
        record_aware_leaderboard_block,
        "record-aware leaderboard hook",
    )
    text = replace_once(
        text,
        "    g_state            = STATE_SURVIVAL_INTRO;",
        "    g_state            = STATE_SURVIVAL_INTRO;\n"
        "#if defined(PLATFORM_WEB) && defined(AGENT_MODE)\n"
        "    g_agent_accum = 0.0f;\n"
        "    g_agent_last_action = 0;\n"
        "    g_agent_qualified = 0;\n"
        "    AgentInput_SetAction(0);\n"
        "#endif",
        "episode reset hook",
    )
    text = replace_once(
        text,
        "    } else if (g_state == STATE_PLAY) {\n"
        "        Player_Update(&g_player, &g_maze, dt);",
        "    } else if (g_state == STATE_PLAY) {\n"
        "#if defined(PLATFORM_WEB) && defined(AGENT_MODE)\n"
        "        g_agent_accum += dt;\n"
        "        if (g_agent_accum >= 0.10f) {\n"
        "            AgentObservation_Build(&g_maze, &g_player, &g_enemies,\n"
        "                                   g_hunger, g_agent_last_action, g_agent_obs);\n"
        "            EM_ASM({\n"
        "                if (window.agentBridge) window.agentBridge.send($0, $1);\n"
        "            }, g_agent_obs, AGENT_OBSERVATION_FLOATS);\n"
        "            g_agent_last_action = EM_ASM_INT({\n"
        "                return window.agentBridge ? (window.agentBridge.action | 0) : 0;\n"
        "            });\n"
        "            AgentInput_SetAction(g_agent_last_action);\n"
        "            g_agent_accum = 0.0f;\n"
        "        }\n"
        "#endif\n"
        "        Player_Update(&g_player, &g_maze, dt);",
        "play-loop hook",
    )
    text = replace_once(
        text,
        "        {\n"
        "            if (IsKeyPressed(KEY_ENTER) || IsKeyPressed(KEY_SPACE)) {\n"
        "                // Retry: survival replays full intro; custom replays same map instantly\n",
        "        {\n"
        "#if defined(PLATFORM_WEB) && defined(AGENT_MODE)\n"
        "            if (g_game_mode == MODE_SURVIVAL && !g_agent_qualified) {\n"
        "                enter_survival_intro();\n"
        "                return;\n"
        "            }\n"
        "#endif\n"
        "            if (IsKeyPressed(KEY_ENTER) || IsKeyPressed(KEY_SPACE)) {\n"
        "                // Retry: survival replays full intro; custom replays same map instantly\n",
        "agent auto retry",
    )
    path.write_text(text, encoding="utf-8")


def patch_makefile(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "SRCS = src/main.c src/wfc.c src/draw_tool.c src/maze.c src/player.c src/enemy.c",
        "SRCS = src/main.c src/wfc.c src/draw_tool.c src/maze.c src/player.c src/enemy.c "
        "src/agent_input.c src/agent_observation.c",
        "source list",
    )
    text = replace_once(
        text,
        "WASM_CFLAGS  = -O2 -I$(RAYLIB_SRC) -Isrc -DPLATFORM_WEB -s USE_GLFW=3",
        "WASM_CFLAGS  = -O2 -I$(RAYLIB_SRC) -Isrc -DPLATFORM_WEB -DAGENT_MODE -s USE_GLFW=3",
        "agent build flag",
    )
    path.write_text(text, encoding="utf-8")


def patch_shell(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    top10_block = """    window.lb_isTop10 = function(time_ms, orbs) {
      if (!_lb_fetched) return false;
      if (_lb_cache.length < LB_MAX) return true;
      var worst = _lb_cache[_lb_cache.length - 1];
      return time_ms > worst.time_ms ||
             (time_ms === worst.time_ms && orbs > worst.orbs);
    };
"""
    record_block = top10_block + """
    window.lb_isRecord = function(time_ms, orbs) {
      if (!_lb_fetched || _lb_cache.length === 0) return false;
      var best = _lb_cache[0];
      return time_ms > best.time_ms ||
             (time_ms === best.time_ms && orbs > best.orbs);
    };
"""
    text = replace_once(text, top10_block, record_block, "record leaderboard helper")

    insertion_point = "    // ---- Emscripten Module ----\n"
    bridge_js = f"""    // {MARKER}
    window.agentBridge = {{
      action: 0,
      socket: null,
      connect: function() {{
        var self = this;
        try {{
          var socket = new WebSocket('ws://127.0.0.1:8765');
          socket.binaryType = 'arraybuffer';
          socket.onopen = function() {{
            console.log('[MazeRunner agent] policy server connected');
          }};
          socket.onmessage = function(event) {{
            if (event.data instanceof ArrayBuffer) {{
              var bytes = new Uint8Array(event.data);
              if (bytes.length) self.action = bytes[0] | 0;
            }} else {{
              self.action = parseInt(event.data, 10) | 0;
            }}
          }};
          socket.onclose = function() {{
            self.socket = null;
            self.action = 0;
            setTimeout(function() {{ self.connect(); }}, 1000);
          }};
          socket.onerror = function() {{ socket.close(); }};
          this.socket = socket;
        }} catch (error) {{
          setTimeout(function() {{ self.connect(); }}, 1000);
        }}
      }},
      send: function(pointer, count) {{
        if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return;
        var start = pointer >> 2;
        var values = HEAPF32.slice(start, start + count);
        this.socket.send(values.buffer);
      }},
      episodeDone: function() {{
        this.action = 0;
        if (this.socket && this.socket.readyState === WebSocket.OPEN)
          this.socket.send('reset');
      }}
    }};
    window.agentBridge.connect();

"""
    text = replace_once(text, insertion_point, bridge_js + insertion_point, "shell insertion")
    path.write_text(text, encoding="utf-8")


def apply_patches(root: Path, clone: Path) -> None:
    src = root / "exact_game" / "web"
    names = ("agent_input.h", "agent_input.c", "agent_observation.h", "agent_observation.c")
    for name in names:
        shutil.copy2(src / name, clone / "src" / name)
    patch_player(clone / "src" / "player.c")
    patch_main(clone / "src" / "main.c")
    patch_makefile(clone / "Makefile")
    patch_shell(clone / "web" / "shell.html")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Prepare a local web build controlled by the trained policy server"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / ".exact_game" / "leaderboard",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--build", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    clone = args.output.resolve()
    prepare_clone(clone, args.force)
    apply_patches(root, clone)
    print(f"Prepared leaderboard build at: {clone}")
    if args.build:
        run(["make", "web"], cwd=clone)
        print(f"Serve {clone / 'web'} over HTTP and keep the policy server running.")


if __name__ == "__main__":
    main()
