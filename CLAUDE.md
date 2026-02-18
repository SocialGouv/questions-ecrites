# CLAUDE.md

## Project conventions

- **No `__init__.py` files.** This project uses implicit namespace packages (PEP 420). Do not create `__init__.py` files. Use direct imports to submodules (e.g. `from qe.clients.qdrant import QdrantClient`, not `from qe.clients import QdrantClient`).
