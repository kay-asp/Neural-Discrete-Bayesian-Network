"""Load notebooks' settings cells, so other notebooks can share them.

Deprecated: notebooks now import shared settings from ndbn.settings directly.
This module is kept for backward compatibility with test scripts.
"""
import json
import types


def load_notebook_config(*paths, tag="config"):
    """Run the cells tagged `tag` in the notebook(s) at `paths`; return their names.

    .. deprecated::
        Notebooks now import from ``ndbn.settings`` instead of using this.
    """
    ns = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            nb = json.load(f)
        cells = [c for c in nb["cells"]
                 if c["cell_type"] == "code" and tag in c.get("metadata", {}).get("tags", [])]
        if not cells:
            raise ValueError(f"no code cells tagged {tag!r} in {path}")
        for c in cells:
            src = "".join(c["source"])
            code = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith(("%", "!")))
            exec(compile(code, f"{path} [{tag} cell {c.get('id', '?')}]", "exec"), ns)
    return {k: v for k, v in ns.items()
            if not k.startswith("_") and not isinstance(v, types.ModuleType)}
