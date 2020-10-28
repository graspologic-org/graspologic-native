import sys
import os
import toml
from datetime import datetime

"""
Github Env Variables:
 - GITHUB_REF
 - GITHUB_RUN_ID
Devops Env Variables:
 - BUILD_SOURCEBRANCH
 - BUILD_BUILDNUMBER
"""

RELEASE_BRANCH = "refs/heads/main"

BUILD_SYSTEM = "github"
REF_KEY = "GITHUB_REF"
BUILD_ID = "GITHUB_RUN_ID"

if len(sys.argv) != 4:
    print(f"Called with wrong arguments, requires: `python {sys.argv[0]} TOML_PATH VERSION_FILE_PATH")
    sys.exit(-1)

if REF_KEY not in os.environ:
    print(f"{REF_KEY} not found in environment variable, adjust script for Azure Devops Pipelines or Github Actions")
    sys.exit(-1)

if BUILD_ID not in os.environ:
    print(f"{BUILD_ID} not found in environment variable, adjust script for Azure Devops Pipelines or Github Actions")
    sys.exit(-1)

branch = os.environ[REF_KEY]
build_id = os.environ[BUILD_ID]
build_date = datetime.now().strftime("%Y%m%d")

if BUILD_SYSTEM == "devops":
    build_id = build_id.split(".")[1]
_, cargo_file, version_file = sys.argv

cargo_conf = toml.load(cargo_file)
if branch != RELEASE_BRANCH:
    # if it's a release branch, don't touch the TOML - it's fine
    original = cargo_conf["package"]["version"]
    snapshot = f"{original}-dev{build_date}{build_id.rjust(3,'0')}"
    cargo_conf["package"]["version"] = snapshot
    with open(cargo_file, "w") as cargo_file_io:
        toml.dump(cargo_conf, cargo_file_io)

version = cargo_conf["package"]["version"]

with open(version_file, "w") as version_file_io:
    version_file_io.write(version)
