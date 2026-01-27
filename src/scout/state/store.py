# scout/store.py

from scout.index.inverted import InvertedIndex
from scout.index.stats import IndexStats
from scout.storage import paths, serializer


class Store:
    """
    Persistent storage for index and stats.
    """

    @staticmethod
    def save(index: InvertedIndex) -> None:
        serializer.save(index, str(paths.INDEX_FILE))
        serializer.save(index.stats, str(paths.STATS_FILE))

    @staticmethod
    def load() -> InvertedIndex:
        index: InvertedIndex = serializer.load(str(paths.INDEX_FILE))
        stats: IndexStats = serializer.load(str(paths.STATS_FILE))
        index.stats = stats
        return index
