# Extensions and plugins

`local_splitter.plugins` defines a small **protocol** (`TacticHook`) for
third-party code that wants to post-process messages after T2 compress.

The bundled pipeline **does not** dynamically import user modules from
YAML (that would be a foot-gun for deployments). Integrate hooks from
your application entrypoint by wrapping `Pipeline` or by forking the
orchestrator in `src/local_splitter/pipeline/pre_cloud.py`.

Future work may add an explicit `LOCAL_SPLITTER_EXTENSIONS=1` gated
import path; until then, treat this package as API documentation for
fork-friendly extension points.
