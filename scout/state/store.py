# scout/store.py

from scout.index.inverted import InvertedIndex
from scout.index.stats import IndexStats
from scout.storage import serializer, paths

class Store:
    """
    Persistent storage for index and stats.
    """

    @staticmethod
    def save_index(index: InvertedIndex) -> None:
        serializer.save(index, str(paths.INDEX_FILE))

    @staticmethod
    def load_index() -> InvertedIndex:
        return serializer.load(str(paths.INDEX_FILE))

    @staticmethod
    def save_stats(stats: IndexStats) -> None:
        serializer.save(stats, str(paths.STATS_FILE))

    @staticmethod
    def load_stats() -> IndexStats:
        return serializer.load(str(paths.STATS_FILE))
