"""Cross-platform Poe entry points. No Git operations or global environment edits."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([str(arg) for arg in args], cwd=ROOT, check=True)


def tool(name):
    executable = shutil.which(name)
    if executable is None:
        candidate = Path(sys.executable).parent / (
            name + (".exe" if sys.platform == "win32" else "")
        )
        if not candidate.exists():
            raise RuntimeError(f"Missing {name}; install the native dependency group")
        executable = str(candidate)
    return executable


def configure():
    run(tool("cmake"), "--preset", "dev", f"-DPython_EXECUTABLE={sys.executable}")


def build():
    configure()
    run(tool("cmake"), "--build", "--preset", "dev")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=["check", "configure", "build", "install", "test", "wheel"]
    )
    action = parser.parse_args().action
    if action == "check":
        run(tool("cmake"), "--version")
        run(sys.executable, "-m", "pybind11", "--cmakedir")
        if sys.platform == "win32":
            vswhere = Path(
                "C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
            )
            result = subprocess.run(
                [
                    str(vswhere),
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            if not result.stdout.strip():
                raise RuntimeError(
                    "Install Visual Studio Build Tools with Desktop development with C++"
                )
            print(result.stdout.strip())
        elif not any(shutil.which(name) for name in ("c++", "g++", "clang++")):
            raise RuntimeError("Install a C++17 compiler")
    elif action == "configure":
        configure()
    elif action in ("build", "install", "test"):
        build()
        if action == "install":
            # Local checkout takes precedence over site-packages when running its tests.
            run(
                tool("cmake"),
                "--install",
                "build/native",
                "--config",
                "Release",
                "--component",
                "python",
                "--prefix",
                ROOT,
            )
        elif action == "test":
            run(tool("ctest"), "--preset", "dev")
    else:
        run("uv", "build", "--wheel", "--no-build-isolation")


if __name__ == "__main__":
    main()
