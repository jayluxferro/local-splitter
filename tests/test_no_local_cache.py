def test_cache_namespace_is_per_chain(tmp_path):
    """Each manifold chain shares one cache DB but must partition by its
    cloud endpoint — a shared 'default' namespace let one chain serve
    another chain's cached answer (found by the config-matrix audit)."""
    import pathlib

    from local_splitter.cli import _build_pipeline
    from local_splitter.config import load_config

    preset = pathlib.Path(__file__).parents[1] / "configs/proxy/no-local.yaml"
    cfg_a = load_config(str(preset))
    cfg_b = load_config(str(preset))
    from dataclasses import replace

    cfg_a = replace(cfg_a, cloud=replace(cfg_a.cloud, endpoint="http://127.0.0.1:11132"))
    cfg_b = replace(cfg_b, cloud=replace(cfg_b.cloud, endpoint="http://127.0.0.1:22243"))

    from conftest import TEST_DB_URL as db

    pipe_a = _build_pipeline(cfg_a, db)
    pipe_b = _build_pipeline(cfg_b, db)
    assert pipe_a.cache_store._namespace != pipe_b.cache_store._namespace
    assert pipe_a.cache_store._namespace == "chain:http://127.0.0.1:11132"


def test_concurrent_store_init_on_fresh_db_single_flight():
    """Regression (hostile review M2): five chains share one cache DB, and
    the snapshot-then-INSERT migration race killed 3-4 of 5 processes at
    first deploy.  The advisory lock must make initialization
    single-flight — every constructor returns alive."""
    import threading

    import psycopg
    from local_splitter.pipeline.sem_cache import LexicalCacheStore

    from conftest import TEST_DB_URL

    # Fresh schema: drop what migrations create (index/table cascade).
    with psycopg.connect(TEST_DB_URL, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS cache_entry")
        conn.execute("DROP TABLE IF EXISTS schema_migrations")

    errors: list[Exception] = []
    barrier = threading.Barrier(5)

    def open_store():
        try:
            barrier.wait()  # maximize the race window
            LexicalCacheStore(TEST_DB_URL)
        except Exception as exc:  # noqa: BLE001 — record every failure
            errors.append(exc)

    threads = [threading.Thread(target=open_store) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, f"concurrent init crashed: {[type(e).__name__ for e in errors]}"
    with psycopg.connect(TEST_DB_URL, autocommit=True) as conn:
        versions = {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}
    assert versions == {1, 2}
