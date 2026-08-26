# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""The smoke harness: the product exercised as a consumer exercises it.

``CERTIFICATE_PATH`` lives here rather than beside the code that writes it,
because it has readers in two places and one of them comes first: the scenario
that asserts nothing else was left behind needs the name, and so does the
writer of the certificate.

A ``VERSION`` sat here too and nothing ever read it. The version that matters
is the PRODUCT's, and the certificate takes that from the binary itself.
"""

CERTIFICATE_PATH = "docs/test/smoke-certificate.md"
