"""
Text chunking.

Why we write this ourselves instead of importing LangChain's splitter:
The whole point of building this "from scratch to learn from it" is
understanding WHY chunking is hard, not just calling a library. This is a
recursive character splitter: try to split on paragraph breaks first: if a
chunk is still too big, fall back to sentences, then words, then raw
characters. That keeps semantically related text together as long as
possible, which is what makes retrieval quality good or bad.
"""
from __future__ import annotations

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_on(text: str, sep: str) -> list[str]:
    if sep == "":
        return list(text)
    return text.split(sep)


def recursive_split(text: str, chunk_size: int, chunk_overlap: int, separators: list[str] = SEPARATORS) -> list[str]:
    """Split `text` into chunks of at most `chunk_size` characters.

    Recursively tries each separator in order until pieces are small enough,
    then reassembles pieces into chunks up to chunk_size, adding overlap
    between consecutive chunks so context isn't lost at a chunk boundary.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep, *rest_seps = separators
    pieces = _split_on(text, sep)

    # If this separator didn't actually break anything up, try the next one.
    if len(pieces) == 1 and rest_seps:
        return recursive_split(text, chunk_size, chunk_overlap, rest_seps)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = (current + sep + piece) if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # piece itself might still be too big -> recurse with remaining separators
            if len(piece) > chunk_size and rest_seps:
                chunks.extend(recursive_split(piece, chunk_size, chunk_overlap, rest_seps))
                current = ""
            else:
                current = piece
    if current:
        chunks.append(current)

    return _add_overlap(chunks, chunk_overlap)


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap:]
        overlapped.append((tail + " " + chunks[i]).strip())
    return overlapped
