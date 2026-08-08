from __future__ import annotations

import argparse
from pathlib import Path

import prepare_leaderboard_build as base

TOP10_MARKER = "MAZERUNNER_PPO_TOP10_GRINDER"
DEFAULT_GRINDER_NAME = "Gabriel"
MAX_BROWSER_DT = 0.05


def patch_top10_main(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    # requestAnimationFrame / Raylib can return a huge frame delta when Chrome
    # suspends or throttles a background WASM tab. Upstream adds that dt directly
    # to both score time and hunger decay, which can manufacture multi-minute
    # scores in a single resumed frame. Clamp only the agent browser build; normal
    # 60 Hz frames (~0.0167 s) are untouched.
    frame_start = """static void UpdateDrawFrame(void) {
    float dt = GetFrameTime();
"""
    frame_start_clamped = f"""static void UpdateDrawFrame(void) {{
    float dt = GetFrameTime();
#if defined(PLATFORM_WEB) && defined(AGENT_MODE)
    if (dt < 0.0f) dt = 0.0f;
    if (dt > {MAX_BROWSER_DT:.2f}f) dt = {MAX_BROWSER_DT:.2f}f;
#endif
"""
    text = base.replace_once(
        text,
        frame_start,
        frame_start_clamped,
        "browser frame-delta clamp",
    )

    # The normal agent build pauses only for a new #1. For the grinder we pause
    # for every score that qualifies for the live top 10. The actual submission
    # still goes through the game's original name-entry modal / Save Score button.
    record_gate = """#if defined(AGENT_MODE)
        if (should_prompt) {
            should_prompt = EM_ASM_INT({
                return (window.lb_isRecord && window.lb_isRecord($0, $1)) ? 1 : 0;
            }, time_ms, g_score_orbs);
            if (should_prompt) g_agent_qualified = 1;
        }
#endif
"""
    top10_gate = f"""#if defined(AGENT_MODE)
        // {TOP10_MARKER}: every genuine top-10 score pauses for the normal modal.
        if (should_prompt) g_agent_qualified = 1;
#endif
"""
    text = base.replace_once(
        text,
        record_gate,
        top10_gate,
        "record-only qualification gate",
    )

    # Upstream returns to the menu after the modal is saved/skipped. In grinder
    # mode, immediately launch another normal Survival run instead. This keeps
    # retries going forever while still waiting for the user to explicitly Save
    # or Skip every qualifying leaderboard score.
    modal_done = """        if (g_lb_modal_pending) {
            if (EM_ASM_INT({ return window.lb_checkDone ? window.lb_checkDone() : 0; })) {
                g_lb_modal_pending = 0;
                EM_ASM({ if (window.lb_refresh) window.lb_refresh(); });
                g_state = STATE_MENU;
            }
        } else
"""
    modal_done_grinder = """        if (g_lb_modal_pending) {
            if (EM_ASM_INT({ return window.lb_checkDone ? window.lb_checkDone() : 0; })) {
                g_lb_modal_pending = 0;
                EM_ASM({ if (window.lb_refresh) window.lb_refresh(); });
#if defined(AGENT_MODE)
                if (g_game_mode == MODE_SURVIVAL) {
                    g_agent_qualified = 0;
                    enter_survival_intro();
                    return;
                }
#endif
                g_state = STATE_MENU;
            }
        } else
"""
    text = base.replace_once(
        text,
        modal_done,
        modal_done_grinder,
        "leaderboard modal completion",
    )

    path.write_text(text, encoding="utf-8")


def patch_top10_shell(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    saved_name = """    function lb_getSavedName() {
      try { return localStorage.getItem(LB_NAME_KEY) || ''; } catch (e) { return ''; }
    }
"""
    grinder_name = f"""    function lb_getSavedName() {{
      try {{ return localStorage.getItem(LB_NAME_KEY) || '{DEFAULT_GRINDER_NAME}'; }}
      catch (e) {{ return '{DEFAULT_GRINDER_NAME}'; }}
    }}
"""
    text = base.replace_once(
        text,
        saved_name,
        grinder_name,
        "grinder default leaderboard name",
    )
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the policy-controlled browser build that retries Survival forever "
            "and pauses on every genuine top-10 score"
        )
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

    base.prepare_clone(clone, args.force)
    base.apply_patches(root, clone)
    patch_top10_main(clone / "src" / "main.c")
    patch_top10_shell(clone / "web" / "shell.html")

    print(f"Prepared endless top-10 grinder at: {clone}")
    print(f"Browser frame delta capped at {MAX_BROWSER_DT:.2f}s to reject tab-resume timing spikes.")
    print(f"Leaderboard name defaults to {DEFAULT_GRINDER_NAME!r}.")
    print("Non-qualifying deaths restart automatically.")
    print("Top-10 runs pause on the game's normal Save Score / Skip modal.")
    print("After Save or Skip, the next Survival run starts automatically.")

    if args.build:
        base.run(["make", "web"], cwd=clone)
        print(f"Serve {clone / 'web'} over HTTP and keep the policy server running.")


if __name__ == "__main__":
    main()
