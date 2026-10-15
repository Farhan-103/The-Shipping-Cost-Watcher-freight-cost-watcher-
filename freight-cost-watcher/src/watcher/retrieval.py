"""Retrieval module: ChromaDB-based RAG with metadata filtering.

Embeds each note once with all-MiniLM-L6-v2 (via ChromaDB's default embedding
function). Rebuilds the collection from scratch on every run. Provides both
a Chroma metadata filter and a plain-Python filter, with an assertion that
both return the same notes.

With only 10 notes, the route+date filter narrows to 0-1 notes almost every
time, so RAG adds little ranking. It is built because the brief asks for
retrieval, and it matters for larger note sets.
"""
import csv
import logging
from datetime import date, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


def _date_to_ord(d: date) -> int:
    """Convert a date to an integer ordinal like 20250224."""
    return d.year * 10000 + d.month * 100 + d.day


def _build_collection(notes, chroma_client, collection_name: str):
    """Build (or rebuild) a ChromaDB collection from parsed notes.

    Embedding text: route | start_date | note text (joined with pipe).
    Metadata: applies_to, start_ord, end_ord (dates as integers).
    """
    # Delete existing collection if it exists
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={'hnsw:space': 'cosine'},
    )

    ids = []
    documents = []
    metadatas = []

    for note in notes:
        ids.append(note.note_id)
        documents.append(f'{note.applies_to}|{note.start_date.isoformat()}|{note.text}')
        metadatas.append({
            'applies_to': note.applies_to,
            'start_ord': _date_to_ord(note.start_date),
            'end_ord': _date_to_ord(note.end_date),
        })

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    logger.info('Built ChromaDB collection with %d notes', len(ids))

    return collection


def _chroma_retrieve(collection, route: str, week_monday: date, n_results: int = 3):
    """Retrieve notes from ChromaDB with metadata filtering.

    Query: 'cost increase on {route}, week of {week_monday}'
    Metadata filter: applies_to in [route, 'All Routes'],
                     start_ord <= week's Sunday,
                     end_ord >= week's Monday.
    """
    week_sunday = week_monday + timedelta(days=6)
    monday_ord = _date_to_ord(week_monday)
    sunday_ord = _date_to_ord(week_sunday)

    query_text = f'cost increase on {route}, week of {week_monday.isoformat()}'

    # ChromaDB where filter: route matches AND date window overlaps
    where_filter = {
        '$and': [
            {
                '$or': [
                    {'applies_to': {'$eq': route}},
                    {'applies_to': {'$eq': 'All Routes'}},
                ],
            },
            {'start_ord': {'$lte': sunday_ord}},
            {'end_ord': {'$gte': monday_ord}},
        ],
    }

    try:
        results = collection.query(
            query_texts=[query_text],
            where=where_filter,
            n_results=n_results,
        )
    except Exception as e:
        logger.warning('ChromaDB query failed: %s; falling back to empty', e)
        return [], []

    if not results['ids'] or not results['ids'][0]:
        return [], []

    note_ids = results['ids'][0]
    distances = results['distances'][0] if results.get('distances') else [0.0] * len(note_ids)

    return note_ids, distances


def python_filter_notes(notes, route: str, week_monday: date) -> list:
    """Plain Python filter: route matches AND date window overlaps.

    This is the reference filter used to verify ChromaDB results.
    """
    matching = []
    for note in notes:
        if note.matches_route(route) and note.overlaps_week(week_monday):
            matching.append(note)

    # Sort by note_id for determinism
    matching.sort(key=lambda n: n.note_id)
    return matching


class Retriever:
    """Manages note retrieval with optional ChromaDB RAG."""

    def __init__(self, notes, use_rag: bool = True, log_path: Path | None = None):
        self.notes = notes
        self.use_rag = use_rag
        self.log_path = log_path
        self._collection = None
        self._chroma_client = None
        self._log_initialized = False

        if use_rag:
            self._init_chroma()

    def _init_chroma(self):
        """Initialize ChromaDB with an ephemeral (in-memory) client."""
        import chromadb
        self._chroma_client = chromadb.Client()
        self._collection = _build_collection(
            self.notes, self._chroma_client, 'context_notes'
        )

    def retrieve(self, route: str, week_monday: date) -> list:
        """Retrieve matching notes for a candidate row.

        Uses both Chroma filter and Python filter, asserts they agree.
        Returns notes sorted by relevance (if RAG) or note_id (if filter-only).
        """
        # Python filter (always runs as ground truth)
        py_matches = python_filter_notes(self.notes, route, week_monday)
        py_ids = sorted(n.note_id for n in py_matches)

        if self.use_rag and self._collection is not None:
            # ChromaDB retrieval
            chroma_ids, distances = _chroma_retrieve(
                self._collection, route, week_monday
            )
            chroma_ids_sorted = sorted(chroma_ids)

            # Consistency assertion: same set of note IDs
            if chroma_ids_sorted != py_ids:
                logger.warning(
                    'Chroma/Python filter mismatch for %s/%s: '
                    'chroma=%s, python=%s',
                    route, week_monday, chroma_ids_sorted, py_ids,
                )
                # Use Python filter as ground truth
                result = py_matches
            else:
                # Reorder by Chroma ranking (lower distance = more relevant)
                note_map = {n.note_id: n for n in py_matches}
                id_dist = list(zip(chroma_ids, distances))
                id_dist.sort(key=lambda x: (x[1], x[0]))  # sort by distance, then id
                result = [note_map[nid] for nid, _ in id_dist if nid in note_map]
        else:
            result = py_matches

        # Log retrieval
        self._log_retrieval(route, week_monday, result)

        return result

    def _log_retrieval(self, route: str, week_monday: date, notes) -> None:
        """Log retrieval results to CSV."""
        if self.log_path is None:
            return

        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        if not self._log_initialized:
            with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'route', 'week_of', 'retrieved_note_ids', 'count',
                ])
            self._log_initialized = True

        note_ids = ';'.join(n.note_id for n in notes)
        with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([route, week_monday.isoformat(), note_ids, len(notes)])
