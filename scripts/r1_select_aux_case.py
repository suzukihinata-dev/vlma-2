"""Select one valid example not already proved by bare DDAR for R1 search smoke."""

from pathlib import Path

from newclid.api import GeometricSolverBuilder


SOURCE = Path("benchmarks/examples.txt")
DESTINATION = Path("/artifacts/genesisgeo/eval-smoke-problems.txt")
CANDIDATES = (
    "two_paths_problem",
    "ar_example_paper_angle_chasing",
    "worlds_hardest_easy_geometry_problem1",
    "imo_2004_p1_generalized",
)


def main() -> None:
    lines = SOURCE.read_text().splitlines()
    for name in CANDIDATES:
        solver = (
            GeometricSolverBuilder(seed=123)
            .load_problem_from_file(str(SOURCE), name)
            .build()
        )
        solved = solver.run(timeout=10)
        print(f"{name}: solved={solved}, infos={solver.run_infos}", flush=True)
        if solved or solver.run_infos.get("error"):
            continue
        index = lines.index(name)
        DESTINATION.write_text(name + "\n" + lines[index + 1] + "\n")
        print(f"Selected {name}: {DESTINATION}")
        return
    raise RuntimeError("No valid unsolved candidate found")


if __name__ == "__main__":
    main()
