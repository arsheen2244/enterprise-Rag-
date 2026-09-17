from app.chunking import recursive_split


def test_short_text_returns_single_chunk():
    text = "This is short."
    assert recursive_split(text, chunk_size=100, chunk_overlap=10) == [text]


def test_long_text_is_split_into_multiple_chunks():
    text = "Paragraph one. " * 50 + "\n\n" + "Paragraph two. " * 50
    chunks = recursive_split(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks)  # allow small slack for overlap join


def test_overlap_preserves_boundary_context():
    text = "A" * 300 + "B" * 300
    chunks = recursive_split(text, chunk_size=300, chunk_overlap=50)
    assert len(chunks) >= 2
    # the tail of chunk 1 should reappear at the start of chunk 2
    assert chunks[0][-10:] in chunks[1]


def test_empty_text_returns_no_chunks():
    assert recursive_split("   ", chunk_size=100, chunk_overlap=10) == []
