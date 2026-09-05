# Setting up your Python environment (Windows)

This folder has one script, `setup_env.py`, that creates a **virtual
environment** for the class: an isolated folder holding its own copy of
Python's packages, so installing things for this class can't break or be
broken by anything else on your computer.

The two words you need to know are **activate** and **deactivate**:

- **Activate** = switch on the class environment. Do this every time you
  open a new terminal to run class code.
- **Deactivate** = switch it back off, returning to your normal Python.

## 1) One-time setup

1. Open `setup_env.py` in a text editor.
2. At the top, set your environment name if you want something other than
   the default:

   ```python
   ENVIR_NAME = "geophysics_env"
   ```

3. Open **Command Prompt** (or PowerShell) in this folder and run:

   ```bash
   python setup_env.py
   ```

   This creates the environment (named `ENVIR_NAME`) in `01_code`, one
   level above `setup_vir_envir`, and installs all the packages the
   class code needs. It can take a few minutes.

   If the script stops with a message about your Python installation,
   fix that first (it tells you exactly what's wrong) and run the script
   again -- don't skip this step, nothing else will work until it passes.

You only need to do this once.

## 2) Every time you work on class code: activate

Open Command Prompt (or PowerShell) in `01_code` (the folder that
contains `setup_vir_envir`, `lib`, `excercise`, `final`, etc.) and run:

**Command Prompt:**
```bash
geophysics_env\Scripts\activate
```

**PowerShell:**
```bash
geophysics_env\Scripts\Activate.ps1
```

(Replace `geophysics_env` with whatever you set `ENVIR_NAME` to.)

You'll know it worked when your prompt starts with `(geophysics_env)`.
From here on, `python` and `pip` in this terminal use the class
environment -- run your scripts as usual, e.g.:

```bash
python ch_02_02_gravity_matrix.py
```

## 3) When you're done: deactivate

Just run:

```bash
deactivate
```

Your prompt goes back to normal, and `python`/`pip` stop pointing at the
class environment. You can close the terminal instead if you're done for
the day -- activation doesn't carry over between terminal windows, so
you'll need to activate again next time.

## Quick reference

| I want to...                          | Command                                |
|----------------------------------------|-----------------------------------------|
| Set up the environment (once)          | `python setup_env.py`                   |
| Turn it on (start of every session)    | `geophysics_env\Scripts\activate`       |
| Turn it off (end of session, optional) | `deactivate`                            |

## Troubleshooting

- **`'python' is not recognized...`** -- Python isn't on your PATH.
  Reinstall from [python.org](https://www.python.org/downloads/) and
  check "Add python.exe to PATH" during install.
- **The activate command "is not recognized" / does nothing** -- make sure
  you're running it from `01_code` (one level above `setup_vir_envir`),
  and that step 1 finished without errors first.
- **PowerShell says running scripts is disabled** -- use the Command
  Prompt `activate` command instead, or ask your instructor about
  `Set-ExecutionPolicy`.
- **Need `pygimli`** (for `07_resis.py`) -- it isn't installed by
  `setup_env.py`. Install
  [Miniforge/Anaconda](https://github.com/conda-forge/miniforge) and run
  `conda install -c conda-forge pygimli` separately.
