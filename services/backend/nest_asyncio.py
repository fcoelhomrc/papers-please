"""Shim redirecting `import nest_asyncio` to the maintained `nest_asyncio2`
fork.

The real `nest_asyncio` on PyPI is archived/dead upstream (github.com/erdewit/
nest_asyncio, archived March 2024) and was never fixed for Python 3.13+:
its patched event loop breaks `asyncio.current_task()` tracking, which makes
`asyncio.wait_for()`/`asyncio.timeout()` raise "RuntimeError: Timeout should
be used inside a task" even when genuinely running inside a task (confirmed
via nest_asyncio's own test suite failing the same way on 3.14 - Red Hat
Bugzilla #2353141). `ragas/executor.py` does `import nest_asyncio;
nest_asyncio.apply()` unconditionally at import time, so this bites us
every time ragas is imported.

`nest_asyncio2` (github.com/Chaoses-Ib/nest-asyncio2) is a maintained fork
fixing exactly this, but ships under a different import name, so it isn't
a transparent drop-in for code that does `import nest_asyncio` by name (like
ragas does). This file makes it one: since /app is ahead of site-packages
on sys.path, `import nest_asyncio` resolves to this shim instead of the
real (broken) package, and forwards to the fork.
"""
from nest_asyncio2 import *  # noqa: F401,F403
from nest_asyncio2 import apply  # noqa: F401
