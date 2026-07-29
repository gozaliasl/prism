"""Backward-compatibility shim.

The simulation driver module was renamed from ``jwst_lens_simulator`` to
``simulator`` (it drives all five supported telescopes, not just JWST).
Import from :mod:`prism.core.simulator` going forward.
"""
from prism.core.simulator import *  # noqa: F401,F403
from prism.core.simulator import main  # noqa: F401

if __name__ == "__main__":
    main()
