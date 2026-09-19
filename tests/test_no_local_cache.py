

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
