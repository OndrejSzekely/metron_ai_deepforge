# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

import pytest


def pytest_collection_finish(session: pytest.Session) -> None:
    if not session.items:
        pytest.exit("No tests collected.", returncode=0)
