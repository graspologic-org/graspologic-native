import sys
import toml

if len(sys.argv) != 3:
    print("Called with wrong arguments, requires: `python .devops/create_snapshot_version.py PATH BUILDID`")
    exit(-1)

cargo_conf = toml.load(sys.argv[1])
build_date, build_id = sys.argv[2].split(".")
original = cargo_conf["package"]["version"]
snapshot = f"{original}-dev{build_date}{build_id.rjust(3,'0')}"
cargo_conf["package"]["version"] = snapshot
with open(sys.argv[1], "w") as cargo_file:
    toml.dump(cargo_conf, cargo_file)
