# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Imports every scenario module so the decorator registers them.

The imports are EXPLICIT and ORDERED. Never ``pkgutil``, never ``glob``:
automatic discovery registers in whatever order the filesystem returns, which
varies between platforms, so two runs of the same commit would emit
certificates differing only in line order, and a diff that moves by itself
destroys the value of ``git log -p`` over a fixed path.

Modules are added here as their phase lands. A module present on disk and
missing from this list never registers, and the reconciliation cannot see it,
so ``ScenarioModuleImportTests`` compares the two.
"""

from smoke.scenarios import workspace  # noqa: F401
from smoke.scenarios import vault  # noqa: F401
from smoke.scenarios import config_fatal  # noqa: F401
from smoke.scenarios import hygiene  # noqa: F401
