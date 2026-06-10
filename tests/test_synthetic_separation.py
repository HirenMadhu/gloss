"""DPI separation (impl §9): values+structure-only model bounded below planted no-doc ceiling; a
doc-using model exceeds it. Validates the CMI estimator against known ground truth.

Lightweight cross-variant version runs in Phase 2 (gloss.proxy). Skipped until that probe lands.
"""
import pytest

pytest.skip("Phase 2/3: needs the cross-variant proxy / model. See run_proxy_gate.", allow_module_level=True)
