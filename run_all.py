#!/usr/bin/env python3
"""
Run every experiment in experiments/ sequentially, saving all figures to
outputs/ and printing each experiment's numerical verification to the
console. Any single experiment failing does not stop the others.

Usage:
    python run_all.py            # run everything
    python run_all.py 01 05 12   # run only experiments 01, 05, 12
"""
import glob
import importlib.util
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(HERE, "experiments")


def discover():
    files = sorted(glob.glob(os.path.join(EXP_DIR, "exp*.py")))
    return files


def main():
    only = sys.argv[1:]
    files = discover()
    if only:
        files = [f for f in files if any(os.path.basename(f).startswith(f"exp{o.zfill(2)}") for o in only)]

    passed, failed = [], []
    for f in files:
        name = os.path.basename(f)
        print("\n" + "=" * 78)
        print(f"Running {name}")
        print("=" * 78)
        spec = importlib.util.spec_from_file_location(name[:-3], f)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            mod.run()
            passed.append(name)
        except Exception:
            print(f"[FAILED] {name}")
            traceback.print_exc()
            failed.append(name)

    print("\n" + "#" * 78)
    print(f"# Done. {len(passed)} passed, {len(failed)} failed.")
    if failed:
        print("# Failed:", ", ".join(failed))
    print("#" * 78)


if __name__ == "__main__":
    main()
