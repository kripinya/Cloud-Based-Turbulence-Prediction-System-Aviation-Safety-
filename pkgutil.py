# local shim for pkgutil.get_loader for environments where stdlib pkgutil lacks it
# This file intentionally shadows the stdlib pkgutil (picked from project root).
# It implements only get_loader which Flask expects.

import importlib.util

def get_loader(fullname):
    """
    Return a loader for a module name, or None if not found.
    Uses importlib.util.find_spec under the hood.
    """
    try:
        spec = importlib.util.find_spec(fullname)
        if spec is None:
            return None
        return spec.loader
    except Exception:
        return None
