"""Print the third-party dependencies declared in pyproject.toml files, one
per line, for `pip install -r`.

Used by the Dockerfiles to install every heavy dependency from the
manifests alone, before any of this repo's source code is copied in — so
editing that code doesn't invalidate (and re-download) the dependency
layers. The pins stay in pyproject.toml; this only reads them. Internal
ktp-* packages are skipped: they're installed from source afterwards, with
--no-deps. Optional-dependency groups are included (e.g. services/ocr's
`onnx` extra).
"""

import sys
import tomllib

for path in sys.argv[1:]:
    with open(path, "rb") as f:
        project = tomllib.load(f)["project"]
    for group in [project.get("dependencies", []), *project.get("optional-dependencies", {}).values()]:
        for dep in group:
            if not dep.startswith("ktp-"):
                print(dep)
