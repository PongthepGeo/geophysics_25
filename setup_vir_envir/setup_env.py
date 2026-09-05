"""Create the class virtual environment (Windows-focused).

For students: this is the ONE line you need to edit before running the
script -- give your environment any name you like (letters, digits,
underscores; no spaces).
"""

ENVIR_NAME = "geophysics_env"

# ---------------------------------------------------------------------------
# Everything below this line does not need to be edited.
# ---------------------------------------------------------------------------

import subprocess
import sys
from pathlib import Path

# Packages the class code needs. Installed together in one pip call so pip
# can resolve compatible versions for all of them at once.
REQUIREMENTS = [
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "seaborn",
    "pillow",
    "opencv-python",
    "tqdm",
    "readgssi",
    "torch",
    "deepwave",
]

# pygimli (used by 07_resis.py) is deliberately NOT installed here -- it's a
# C++/Boost-heavy package that upstream recommends installing via
# conda/mamba from conda-forge, not pip. If you need it, install
# Miniforge/Anaconda separately and run:
#     conda install -c conda-forge pygimli

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = REPO_ROOT / ENVIR_NAME


MIN_PYTHON = (3, 9)


def venv_python(venv_dir: Path) -> Path:
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run(cmd, **kwargs):
    print(">", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def check_python() -> None:
    """Confirm the Python running this script is a real, complete install.

    The most common way a student's setup breaks is not this script -- it's
    Python itself: the Microsoft Store stub, a copy missing ensurepip/venv,
    or a version too old for the packages we need. Catch that here, with a
    clear message, instead of failing later with a cryptic pip/venv error.
    """
    print(f"Checking Python installation...\n  {sys.executable}")

    if sys.version_info < MIN_PYTHON:
        got = ".".join(map(str, sys.version_info[:3]))
        need = ".".join(map(str, MIN_PYTHON))
        sys.exit(
            f"Python {got} is too old (need {need}+).\n"
            "Install a current Python from https://www.python.org/downloads/ "
            "(check 'Add python.exe to PATH' during install) and run this "
            "script again with that Python."
        )

    try:
        import ensurepip  # noqa: F401
        import venv  # noqa: F401
    except ImportError:
        sys.exit(
            "This Python installation is missing the 'venv'/'ensurepip' "
            "modules, so it cannot create a virtual environment.\n"
            "This usually means Python came from the Microsoft Store or a "
            "stripped-down install. Install the full Python from "
            "https://www.python.org/downloads/ instead, then run this "
            "script again with that Python."
        )

    print("Python looks good -- continuing.\n")


def main() -> None:
    if not ENVIR_NAME.strip():
        sys.exit("ENVIR_NAME is empty -- set it at the top of this script first.")

    check_python()

    print(f"Creating virtual environment '{ENVIR_NAME}' in:\n  {REPO_ROOT}\n")

    try:
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    except subprocess.CalledProcessError:
        sys.exit(
            "Could not create the virtual environment.\n"
            "Make sure Python was installed from python.org with "
            "'Add python.exe to PATH' checked, then try again."
        )

    py = venv_python(VENV_DIR)
    if not py.exists():
        sys.exit(f"Expected the new environment's Python at {py}, but it is not there.")

    print("\nUpgrading pip...")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip"])

    print("\nInstalling required packages (this can take a few minutes)...")
    try:
        run([str(py), "-m", "pip", "install", *REQUIREMENTS])
    except subprocess.CalledProcessError:
        sys.exit(
            "Package installation failed. Check your internet connection and "
            "the error message above, then run this script again."
        )

    print("\n" + "=" * 70)
    print("Done! Your environment is ready.")
    print("=" * 70)
    print(
        "\nNote: pygimli (needed only by 07_resis.py) is NOT included -- "
        "install it separately via Miniforge/Anaconda:\n"
        "    conda install -c conda-forge pygimli"
    )
    print(f"\nTo use it from now on, open Command Prompt or PowerShell in:\n  {REPO_ROOT}")
    print("\nWindows Command Prompt:")
    print(f"    {ENVIR_NAME}\\Scripts\\activate")
    print("\nWindows PowerShell:")
    print(f"    {ENVIR_NAME}\\Scripts\\Activate.ps1")
    print("\n(macOS/Linux: source " + ENVIR_NAME + "/bin/activate)")
    print("\nYou'll know it worked when your prompt starts with "
          f"'({ENVIR_NAME})'.")


if __name__ == "__main__":
    main()
