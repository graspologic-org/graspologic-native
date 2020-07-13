import sys
import toml

if len(sys.argv) != 4:
    print("Called with wrong arguments, requires: `python .devops/manifest_version.py PATH BRANCH BUILDID`")
    exit(-1)

_, cargo_file, branch, build = sys.argv

if branch != "main":
    cargo_conf = toml.load(cargo_file)
    build_date, build_id = build.split(".")
    original = cargo_conf["package"]["version"]
    snapshot = f"{original}-dev{build_date}{build_id.rjust(3,'0')}"
    cargo_conf["package"]["version"] = snapshot
    with open(sys.argv[1], "w") as cargo_file:
        toml.dump(cargo_conf, cargo_file)
