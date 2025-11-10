"""
Thread-local context for tracking file/chunk information for continuous RoPE.

This allows us to pass file metadata to TTT layers without modifying
the entire model forward call chain.
"""

import threading
from typing import Optional

# Thread-local storage for current file/chunk context
_rope_context = threading.local()


def set_rope_context(file_id: Optional[str], chunk_index: Optional[int]):
    """Set the current file/chunk context for this thread."""
    _rope_context.file_id = file_id
    _rope_context.chunk_index = chunk_index


def get_rope_context() -> tuple[Optional[str], Optional[int]]:
    """Get the current file/chunk context for this thread."""
    file_id = getattr(_rope_context, 'file_id', None)
    chunk_index = getattr(_rope_context, 'chunk_index', None)
    return file_id, chunk_index


def clear_rope_context():
    """Clear the current file/chunk context."""
    _rope_context.file_id = None
    _rope_context.chunk_index = None
