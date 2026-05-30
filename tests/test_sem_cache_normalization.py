"""Cache-key normalization: redactor placeholders + Anthropic block content.

The semantic cache embeds the *last user message* for similarity lookup.
Two cross-cutting concerns this test file exercises:

1. **Redactor placeholders carry per-request tags** (``⟨EMAIL_1·a1b2c3d4⟩``).
   If we keyed the cache on the raw redacted text, every request would
   hash differently and the cache would never hit.  ``_extract_cache_text``
   strips the ``·tag`` segment so PII-equivalent queries collapse.

2. **Anthropic content is a list of blocks** (``[{"type": "text",
   "text": "..."}]``).  The cache must read the text out of those blocks,
   not stringify the whole list.
"""

from __future__ import annotations

from local_splitter.pipeline.sem_cache import (
    _extract_cache_text,
    _normalize_redactor_placeholders,
    cache_embed_source,
)


def test_strip_redactor_per_request_tag() -> None:
    raw = "send to ⟨EMAIL_1·a1b2c3d4⟩ and ⟨EMAIL_2·a1b2c3d4⟩"
    out = _normalize_redactor_placeholders(raw)
    assert out == "send to ⟨EMAIL_1⟩ and ⟨EMAIL_2⟩"


def test_normalize_is_idempotent() -> None:
    out = _normalize_redactor_placeholders("⟨SSN_3⟩ already normalized")
    assert out == "⟨SSN_3⟩ already normalized"


def test_two_redacted_variants_collapse() -> None:
    msgs_a = [
        {
            "role": "user",
            "content": "ping ⟨EMAIL_1·session1⟩ about ⟨PHONE_1·session1⟩",
        }
    ]
    msgs_b = [
        {
            "role": "user",
            "content": "ping ⟨EMAIL_1·different_session⟩ about ⟨PHONE_1·different_session⟩",
        }
    ]
    assert _extract_cache_text(msgs_a) == _extract_cache_text(msgs_b)


def test_anthropic_block_content_extracted() -> None:
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first part"},
                {"type": "text", "text": "second part"},
            ],
        }
    ]
    out = _extract_cache_text(msgs)
    assert "first part" in out
    assert "second part" in out


def test_anthropic_block_with_redactor_placeholder() -> None:
    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "send to ⟨EMAIL_1·abc12345⟩ please",
                }
            ],
        }
    ]
    out = _extract_cache_text(msgs)
    assert "⟨EMAIL_1⟩" in out
    assert "abc12345" not in out


def test_cache_embed_source_uses_normalized_text() -> None:
    msgs = [{"role": "user", "content": "hi ⟨NAME_1·s1⟩"}]
    out = cache_embed_source(msgs, params=None, meta=None)
    assert out == "hi ⟨NAME_1⟩"


def test_extract_empty_when_no_user_message() -> None:
    msgs = [{"role": "assistant", "content": "nothing"}]
    assert _extract_cache_text(msgs) == ""


def test_extract_uses_last_user_only() -> None:
    msgs = [
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second turn ⟨EMAIL_1·tag⟩"},
    ]
    out = _extract_cache_text(msgs)
    assert "second turn" in out
    assert "first turn" not in out
    assert "tag" not in out
