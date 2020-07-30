import sys
import subprocess


def execute() -> str:
    maturin_list_interpreters = ["maturin", "list-python"]

    results = subprocess.run(maturin_list_interpreters, capture_output=True, encoding="utf8", errors="ignore")

    if results.returncode != 0:
        print(
            f"Something happened when executing {' '.join(maturin_list_interpreters)}\n\tSTDERR: {results.stderr}\n\tSTDOUT: {results.stdout}",
            file=sys.stderr
        )
        exit(-1)
    return results.stdout


def detect(
    python_major_minor: str,
    maturin_output: str
) -> str:
    search_sequence = f" - CPython {python_major_minor} at "
    lines = maturin_output.split("\n")
    matches = []
    for line in lines:
        if line.startswith(search_sequence):
            matches.append(line.replace(search_sequence, "").strip())

    if len(matches) > 1:
        print(f"We found more than one match meeting our major.minor criteria: {python_major_minor}", file=sys.stderr)
        print(f"Interpreters found that match: {matches}")
        exit(-1)
    elif len(matches) == 0:
        print(f"We were not able to extract any matches meeting our major.minor criteria: {python_major_minor} from {maturin_output}", file=sys.stderr)
        exit(-1)
    return matches[0]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit(-1)
    interpreter = detect(sys.argv[1], execute())
    results = subprocess.run(
        ["maturin", "build", "--release", "-i", interpreter],
        capture_output=True,
        encoding="utf8",
        errors="ignore"
    )

    build_stdout = results.stdout.encode("utf8", errors="ignore")
    build_stderr = results.stderr.encode("utf8", errors="ignore")
    if results.returncode == 0:
        print(build_stdout, file=sys.stdout)
    else:
        if results.stdout == results.stderr:
            print(build_stderr, file=sys.stderr)
        else:
            print(f"STDOUT: {build_stdout}", file=sys.stdout)
            print(f"STDERR: {build_stderr}", file=sys.stderr)
    exit(results.returncode)


# import unittest
#
#
# class TestInterpreters(unittest.TestCase):
#
#     def test_windows(self):
#         maturin_output_capture = """🐍 4 python interpreter found:
#  - CPython 3.8 at C:\hostedtoolcache\windows\Python\3.8.5\x64\python.exe
#  - CPython 3.7 at C:\hostedtoolcache\windows\Python\3.7.8\x64\python.exe
#  - CPython 3.6 at C:\hostedtoolcache\windows\Python\3.6.8\x64\python.exe
#  - CPython 3.5 at C:\hostedtoolcache\windows\Python\3.5.4\x64\python.exe
# """
#         expected = "C:\hostedtoolcache\windows\Python\3.8.5\x64\python.exe"
#         results = detect("3.8", maturin_output_capture)
#         self.assertEqual(expected, results)
#
#     def test_macos(self):
#         maturin_output_capture = """🐍 1 python interpreter found:
#  - CPython 3.8 at python3.8
#
# """
#         expected = "python3.8"
#         results = detect("3.8", maturin_output_capture)
#         self.assertEqual(expected, results)
#
#     def test_linux(self):
#         maturin_output_capture = """🐍 2 python interpreter found:
#  - CPython 3.6m at python3.6
#  - CPython 3.8 at python3.8
#
# """
#         expected = "python3.8"
#         results = detect("3.8", maturin_output_capture)
#         self.assertEqual(expected, results)
