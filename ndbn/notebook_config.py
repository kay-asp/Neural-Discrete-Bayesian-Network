"""Load notebooks' settings cells, so other notebooks can share them."""
import json
import types


def load_notebook_config(*paths, tag="config"):
    """Run the cells tagged `tag` in the notebook(s) at `paths`; return their names.

    Each setting is defined once, in the notebook it belongs to (cells tagged
    "config"), and later notebooks load them, e.g.
        globals().update(load_notebook_config("1_mle.ipynb", "2_hyperparameter_tuning.ipynb"))
    The notebooks' tagged cells run in the given order in one shared
    namespace, so a later notebook's settings can use an earlier one's (e.g.
    RESULTS_DIR). Lines starting with % or ! (notebook magics) are skipped, so
    each tagged cell must do its own imports. Returns the public names they
    define (no leading _, no modules).
    This reads the *saved* files: save a notebook after editing its settings.
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
