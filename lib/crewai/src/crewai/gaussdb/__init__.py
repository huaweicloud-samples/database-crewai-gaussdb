"""GaussDB storage support for crewAI.

Shared connection/config layer used by the GaussDB persistence backends
(flow state, kickoff outputs, checkpoints, and -- in a follow-up plan --
vector storage). Enabled via ``CREWAI_STORAGE_BACKEND=gaussdb`` plus the
``GAUSSDB_*`` connection environment variables.
"""
