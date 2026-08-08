from __future__ import annotations

import argparse
from pathlib import Path

import prepare_leaderboard_build as base

TOP10_MARKER = "MAZERUNNER_PPO_TOP10_GRINDER"


def patch_top10_main(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

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

    print(f"Prepared endless top-10 grinder at: {clone}")
    print("Non-qualifying deaths restart automatically.")
    print("Top-10 runs pause on the game's normal Save Score / Skip modal.")
    print("After Save or Skip, the next Survival run starts automatically.")

    if args.build:
        base.run(["make", "web"], cwd=clone)
        print(f"Serve {clone / 'web'} over HTTP and keep the policy server running.")


if __name__ == "__main__":
    main()
