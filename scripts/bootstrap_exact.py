from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

UPSTREAM_URL = "https://github.com/nadarenator/mazerunner.git"
UPSTREAM_COMMIT = "95f83d1a35717fe4dacbb50fc12a9b57f1575754"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def library_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return "libmazerunner_exact.dylib"
    if system == "Linux":
        return "libmazerunner_exact.so"
    raise RuntimeError("The exact backend currently supports macOS and Linux")


def prepare_upstream(path: Path, force: bool) -> None:
    if force and path.exists():
        shutil.rmtree(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--no-checkout", UPSTREAM_URL, str(path)])
    run(["git", "fetch", "--depth", "1", "origin", UPSTREAM_COMMIT], cwd=path)
    run(["git", "checkout", "--detach", UPSTREAM_COMMIT], cwd=path)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError(f"Expected upstream {UPSTREAM_COMMIT}, got {actual}")


def compile_library(root: Path, upstream: Path, output: Path) -> None:
    cc = os.environ.get("CC", "cc")
    bridge = root / "exact_game" / "bridge"
    sources = [
        bridge / "raylib_stub.c",
        bridge / "agent_bridge.c",
        upstream / "src" / "wfc.c",
        upstream / "src" / "draw_tool.c",
        upstream / "src" / "maze.c",
        upstream / "src" / "player.c",
        upstream / "src" / "enemy.c",
    ]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing build inputs: " + ", ".join(missing))

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        cc,
        "-std=c11",
        "-O3",
        "-DNDEBUG",
        "-fPIC",
        "-fvisibility=hidden",
        "-Wall",
        "-Wextra",
        "-Wno-unused-parameter",
        f"-I{bridge}",
        f"-I{upstream / 'src'}",
    ]
    if platform.system() == "Darwin":
        command.append("-dynamiclib")
    else:
        command.append("-shared")
    command.extend(str(path) for path in sources)
    command.extend(["-lm", "-o", str(output)])
    run(command)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Clone the pinned MazeRunner source and build the headless exact backend"
    )
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=root / ".exact_game" / "mazerunner",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "mazerunner_ppo" / "lib" / library_name(),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    upstream = args.upstream_dir.resolve()
    output = args.output.resolve()
    prepare_upstream(upstream, args.force)
    compile_library(root, upstream, output)
    metadata = {
        "upstream_url": UPSTREAM_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "library": str(output),
        "platform": platform.platform(),
        "python": sys.version,
    }
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Built exact backend: {output}")


if __name__ == "__main__":
    main()
