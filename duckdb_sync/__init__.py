"""
duckdb_sync — synchronisation GOLD + TRENDS vers dashboard.duckdb.

Usage minimal :
    from duckdb_sync import DuckDBSync

    sync = DuckDBSync.from_env()
    sync.run()

Usage avec paramètres explicites :
    from duckdb_sync import DuckDBSync

    sync = DuckDBSync(
        duckdb_path="data/dashboard.duckdb",
        gold_path="data/GOLD",
        trends_path="data/TRENDS",
    )
    results = sync.run(
        categories=["stat_ML", "cs_AI"],
        display_names={"stat_ML": "Statistics - ML", "cs_AI": "Computer Science - AI"},
    )
"""

from .sync import DuckDBSync

__all__ = ["DuckDBSync"]
