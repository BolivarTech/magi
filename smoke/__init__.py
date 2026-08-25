# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""The smoke harness: the product exercised as a consumer exercises it.

Two constants live here rather than beside the code that uses them, because
both have readers in more than one place and one of those readers comes first.
``CERTIFICATE_PATH`` names the only file the harness may leave in the tree: the
scenario that asserts nothing else was left behind needs it, and so does the
writer of the certificate itself.
"""

VERSION = "1.0.0"
CERTIFICATE_PATH = "docs/test/smoke-certificate.md"
