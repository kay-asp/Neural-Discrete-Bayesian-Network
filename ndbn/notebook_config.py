"""Load a notebook's settings cells, so another notebook can share them."""
import json
import types


def load_notebook_config(path, tag="config"):
    """Run the cells tagged `tag` in the notebook at `path`; return their names.

    The settings stay defined in one notebook (e.g. NN_estimation.ipynb, whose
    settings cells carry the "config" tag) and other notebooks read them with
        globals().update(load_notebook_config("NN_estimation.ipynb"))
    Tagged cells run in order in a fresh namespace; lines starting with % or !
    (notebook magics) are skipped, so each tagged cell must do its own imports.
    Returns the public names they define (no leading _, no modules).
    This reads the *saved* file: save the notebook after editing its settings.
    """
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    cells = [c for c in nb["cells"]
             if c["cell_type"] == "code" and tag in c.get("metadata", {}).get("tags", [])]
    if not cells:
        raise ValueError(f"no code cells tagged {tag!r} in {path}")
    ns = {}
    for c in cells:
        src = "".join(c["source"])
        code = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith(("%", "!")))
        exec(compile(code, f"{path} [{tag} cell {c.get('id', '?')}]", "exec"), ns)
    return {k: v for k, v in ns.items()
            if not k.startswith("_") and not isinstance(v, types.ModuleType)}
