# Migrating from DuckDB to chDB: ClickHouse in your Python process — drop-in pandas, 8× faster JSON

chDB embeds the **full ClickHouse engine** in your Python process. Same three-line dev experience as DuckDB (single Python wheel, query a Parquet file, get a DataFrame back) — but with ClickHouse SQL (1000+ functions), typed `JSON` sub-columns, native vectors, `MergeTree` storage, ~80 data formats / 12+ source connectors in core, drop-in pandas via `import datastore as pd`, and one-statement federation to a remote ClickHouse cluster. The surface is the same; the engines underneath are built for very different workloads.

We ran 16 typical agent and notebook queries on chDB 4.1 vs DuckDB 1.4, head to head. The numbers:

- **Typed JSON path access** — chDB **8.0× faster**
- **Drop-in pandas-compatible API** — `import datastore as pd` covers **~300 pandas-shaped methods**, compiles to ClickHouse SQL
- **Multi-percentile (`p50` / `p95` / `p99`)** — chDB **2.35× faster**
- **Funnel and sequence pattern aggregates** — chDB **2.5–4.5× faster**
- **Parquet → DataFrame export** — chDB **1.3–3× faster**

DuckDB stays ahead on plain Parquet `count` / `sum` / `GROUP BY` scans, on lightweight one-shot `CREATE TABLE AS` workflows (chDB's `MergeTree` pays a sort + index build upfront), and on the isolated raw cosine kernel — see §5 for the full picture.

This guide is for engineers and data scientists choosing between, or already on, DuckDB. It helps you decide **which workloads belong on chDB vs DuckDB**, shows the concrete wins (and where DuckDB still has the edge), and lays out exactly how to migrate when you decide to — about a day of work for a mid-sized notebook codebase.

---

## 1. chDB vs DuckDB at a glance

| Dimension | DuckDB | chDB |
|---|---|---|
| Lineage | Postgres-flavored optimizer built from scratch | Embedded ClickHouse query engine |
| SQL dialect | PostgreSQL-compatible | ClickHouse SQL |
| Built-in functions | ~600 | **1000+** (full ClickHouse function library) |
| Specialty SQL | `PIVOT` / `UNPIVOT`, `ASOF JOIN`, `QUALIFY`, `GROUP BY ALL`, `COLUMNS(*)` star expressions, list lambdas, `STRUCT`, `approx_top_k` | `windowFunnel`, `uniqHLL12` / `uniqCombined`, `quantilesTDigest`, `sequenceCount` / `sequenceMatch`, `retention`, `geoToH3`, the `-If` / `-Merge` / `-State` combinator family |
| Data formats | CSV, JSON, Parquet, Excel; Avro / Delta / Iceberg via extensions (~10 total) | **~80, everything can be queried in place** — CSV, JSON, Parquet, Avro, ORC, Arrow, Protobuf, CapnProto, MsgPack, BSON, Npy, MySQLDump, Native, … all in core |
| Remote source connectors | `postgres`, `mysql`, `sqlite`, `httpfs`, `aws`, `azure`, `spatial` in the official core matrix; `mongo` / `redis` / `tributary` (Kafka) / `airport` (Arrow Flight) via the community-extension registry | `postgresql()`, `mysql()`, `sqlite()`, `s3()`, `url()`, `hdfs()`, `mongodb()`, `redis()`, `iceberg()`, `deltaLake()`, `hudi()`, `arrowFlight()` — **all in core, no `INSTALL/LOAD` step** |
| Streaming sources | None | `Kafka`, `RabbitMQ`, `NATS` storage engines |
| Pandas API | SQL-on-DataFrame via replacement scan (`duckdb.sql("SELECT … FROM pdf")`) | **Drop-in pandas: change `import pandas as pd` to `import datastore as pd`, rest of the code stays the same** (~300 DataFrame / `.str` / `.dt` methods compiled to ClickHouse SQL); `Python(df)` for raw SQL access |
| Streaming results (large) | `cursor.fetchmany(n)` — paged fetch with engine-side cursor state | `sess.send_query()` push-based chunked iterator — **constant memory** for unbounded result sets (80 % RSS reduction in v4.1.6); critical for streaming millions of rows to a downstream consumer |
| Concurrent writes | Single-writer + WAL — concurrent inserts serialise | **`MergeTree` part-based inserts** — multiple writers append separate parts, background merges run async; designed for many-producer telemetry |
| Event-stream storage | Single-file `.db` columnar — designed for batch analytics on settled data | **`MergeTree` family** with async background merges + `AggregatingMergeTree` rollups + `TTL` auto-ageing + time-series codecs (`Delta`, `Gorilla`, `DoubleDelta`) — designed for append-only telemetry |
| Remote federation | `ATTACH` to Postgres / MySQL / SQLite / Iceberg / DuckLake, plus DuckDB's hosted cloud variant | `remoteSecure('host:9440', db.table, ...)` to a ClickHouse cluster — distributed query planning across local + remote in one optimiser |
| Vector indexing | HNSW via the `vss` extension | Vector skip-indexes for ANN, in core (raw `cosineDistance` / `L2Distance` are in core on both sides) |
| Browser / WASM | Supported (DuckDB-WASM) | Not supported |

---

## 2. Nine reasons to migrate

The reasons below drive most AI-agent and notebook adoption today. Each one is a concrete chDB capability that either has no DuckDB equivalent at all, or requires significant glue to replicate. They are ordered top-to-bottom by frequency in real agent codebases.

1. **You work with typed JSON columns.**

    **Use case:** any agent that persists tool-call outputs, API responses, or LLM replies as JSON and queries them repeatedly — LLM trace dashboards, multi-tenant user-event analytics, log-analysis copilots.

    **Why it matters:** both engines now ship a typed `JSON` type (DuckDB since v1.2, chDB throughout). The architectural difference is **when the sub-column extraction happens**: chDB materialises each JSON path as its own typed sub-column at load time, so `payload.user.tier.:String` compiles to a direct **O(1) column read** with its own min/max stats and compression; DuckDB's typed `JSON` uses on-demand path access against the encoded value. On 1 M agent events the gap is **8×** (Q1; see cookbook [BENCHMARK.md](https://github.com/chdb-io/cookbook/blob/main/migration-from-duckdb/BENCHMARK.md) Case A) — the single highest-leverage win in the benchmark.

2. **You want a pandas-compatible API that compiles to SQL.**

    **Use case:** existing pandas-heavy notebooks, Hex / Jupyter workflows, ML feature-engineering pipelines, or any agent codebase that already speaks pandas (most of them).

    **Why it matters:** `import datastore as pd` covers **~300 pandas-shaped methods** (209 `DataFrame` + 56 `.str` + 42+ `.dt` accessors) — every common pandas idiom (filters, group-bys, joins, time bucketing, window functions, accessor chains) compiles to ClickHouse SQL with the same call shape, plus a further 334 ClickHouse SQL functions exposed as DataStore methods for things pandas doesn't have native names for (`quantilesTDigest`, `uniqHLL12`, `windowFunnel`, …). DuckDB's Python API stays on the SQL surface — `duckdb.sql("SELECT … FROM pdf")` reads a DataFrame via replacement scan with no `register()` step, but **chained pandas-method syntax compiles only on the chDB side**.

3. **You're building AI-agent retrieval (RAG, memory, similarity).**

    **Use case:** chatbots with long-term memory, document Q&A pipelines, copilot session state, agents that combine vector retrieval with structured filters and analytical JOINs.

    **Why it matters:** The chDB win is *integration*: vectors, session memory (`chdb.session.Session('path')`), and analytical SQL live in **one engine, one cache, one transaction boundary** — a single SQL statement runs `JSON metadata filter → cosine top-K → SQL JOIN session history → DataFrame`. The DuckDB equivalent still works but pays the JSON-metadata-filter gap (see cookbook [BENCHMARK.md](https://github.com/chdb-io/cookbook/blob/main/migration-from-duckdb/BENCHMARK.md) Case A), has no native session abstraction (external SQLite / Postgres / Redis), and HNSW indexing requires the `vss` extension.

4. **You log LLM traces, agent events, or any append-only telemetry.**

    **Use case:** self-hosted LLM observability (prompt / response / tokens / latency / tool calls), SLO dashboards for production agents, time-series analytics over Kafka / RabbitMQ / NATS event streams.

    **Why it matters:** chDB inherits ClickHouse's full **append-only event-stream stack**: `MergeTree` with async background merges and near-zero write amplification, `AggregatingMergeTree` / `SummingMergeTree` for pre-aggregated rollups, incremental materialized views, `TTL` for auto-ageing, time-series codecs (`Delta`, `Gorilla`, `DoubleDelta`), and `Kafka` / `RabbitMQ` / `NATS` storage engines. The storage format is the same one consumed by ClickStack (ClickHouse's open-source observability platform, formerly HyperDX) and Langfuse (LLM-focused), so a chDB-persisted trace store is queryable by either without schema translation. DuckDB's single-file model and lack of streaming engines aren't a fit for the same append-only telemetry pattern.

5. **You read from heterogeneous data sources and formats.**

    **Use case:** agents pulling from MySQL / Postgres operational stores, MongoDB document collections, S3 data lakes, Iceberg / Delta tables, Kafka streams, or any mix of Protobuf / Avro / MsgPack serialised data — data-engineering agents, ETL copilots, federated analytics workflows.

    **Why it matters:** chDB ships ~80 data formats and 12+ source connectors in the core binary — no extension management required:

    | Layer | What's in core |
    |---|---|
    | Data formats (~80) | Protobuf · CapnProto · Avro · MsgPack · ORC · Arrow · Native · BSON · Npy · MySQLDump · plus the usual CSV / JSON / Parquet |
    | Remote source connectors (12+) | `mysql()` · `postgresql()` · `mongodb()` · `redis()` · `s3()` · `hdfs()` · `iceberg()` · `deltaLake()` · `hudi()` · `arrowFlight()` · `url()` · `file()` |
    | Streaming engines | `Kafka` · `RabbitMQ` · `NATS` |

    The function-level shape of this is what matters in agent code: `SELECT * FROM mongodb('host', 'db', 'coll', 'user', 'pw')` or `SELECT … FROM Kafka(broker, topic)` — every source is just another table function alongside `file()` and `s3()`, optimised together inside one query plan. DuckDB splits into two camps:

    - **Not in DuckDB Labs' first-party core extensions**: `rabbitmq()` and `nats()` are genuine coverage gaps. `mongodb()` / `redis()` / `kafka()` / `arrowFlight()` are available as **community extensions** (`mongo` / `redis` / `tributary` / `airport`) installable via `INSTALL ... FROM community` — third-party-maintained.
    - **DuckDB ships an official first-party extension** that loads via `INSTALL/LOAD` (functionally equivalent once loaded): `iceberg`, `delta`, `postgres`, `mysql`, `sqlite`, `httpfs`, `aws`, `azure`, `spatial`, `ducklake`.

    chDB's actual differentiator here is **packaging** — every source lives in the core binary, so there is no `INSTALL/LOAD` chain in your container image, no extension version pinning, no per-extension threading/auth model to learn. For an agent runtime that has to talk to four or five external systems, that collapses real operational glue.

    The agent payoff is **glue you don't have to write**. The typical chDB retrieval chain:

        tool-call JSON  →  session memory  →  local / remote embedding  →  online log  →  JOIN warehouse  →  DataFrame

    is one SQL statement against one in-process engine. **Data sources, data formats, SQL analytics, session state, and ClickHouse federation all live inside the same Python process.**

6. **You federate to a remote ClickHouse cluster.**

    **Use case:** customer-support copilots, BI assistants, cross-tenant analytics agents — anything that combines a small local working set with warehouse-scale history in ClickHouse Cloud.

    **Why it matters:** `remoteSecure('host:9440', 'db.table', 'user', 'pass')` joins a local Parquet file with a remote ClickHouse table in **one SQL statement**, planned across both sides by a single optimiser. DuckDB has its own hosted cloud variant for hybrid execution, so if your warehouse is already a different system the choice tracks that. **chDB is the only in-process engine that federates natively to a ClickHouse cluster**, with distributed query planning across local + remote in one optimiser.

7. **You're already in the ClickHouse stack.**

    **Use case:** teams running ClickHouse Cloud or on-prem ClickHouse who want local notebooks, sandboxes, CI fixtures, and edge jobs to speak the same SQL as the warehouse — copy-paste from production query to notebook to edge agent without dialect translation.

    **Why it matters:** chDB lets you run a ClickHouse query verbatim in a Python script. **No SQL dialect bridge to maintain.** This is a dialect-compatibility property — it applies even when no federation is involved (CI fixtures, edge devices, sandbox replay).

8. **You write funnel / cohort / session / pattern analytics.**

    **Use case:** product-analytics dashboards (signup → activation → conversion funnels), user-behavior agents, session-level sequence detection, retention reports.

    **Why it matters:** `windowFunnel`, `sequenceCount`, `sequenceMatch`, and `retention` are **single-aggregate SQL expressions** in chDB. The DuckDB equivalent requires multi-CTE pipelines with `LAG` windows; in our benchmark they were **2.5–4.5× slower in DuckDB**.

    The same aggregates compose with chDB's `-State` / `-Merge` / `-If` combinator family — `windowFunnelStateIf(...)` produces a serialisable partial aggregate, store it in an `AggregatingMergeTree`, and `Merge` it later across hourly / per-tenant / cross-shard slices without re-scanning the raw events. DuckDB aggregates have no equivalent intermediate state, so the same use case forces a re-run from raw rows each time.

9. **You need many percentiles in one pass.**

    **Use case:** SLO reports (p50 / p95 / p99 in one query), application performance dashboards, statistical observability over high-cardinality metrics.

    **Why it matters:** both engines can compute multiple percentiles from one TDigest sketch — chDB with `quantilesTDigest(0.5, 0.95, 0.99)(x)`, DuckDB with the list form `approx_quantile(x, [0.5, 0.95, 0.99])`. On p50/p95/p99 over 18 M rows, the chDB sketch implementation is **2.35× faster** (64 ms → 27 ms). The chDB family also extends to `quantilesExact`, `quantilesGK`, `quantilesBFloat16Weighted` — same API shape, different precision / memory trade-offs.

---

## 3. When to stay on DuckDB

These are honest "stay on DuckDB" signals — not every workload should migrate.

1. **You depend on `PIVOT` / `UNPIVOT`.** chDB has no direct equivalent. Rewriting with `sumIf` / `countIf` conditional aggregation works but is more verbose. If your warehouse contract is "wide reports out of long fact tables," DuckDB is the better fit.

2. **You ship a browser / WASM bundle.** DuckDB-WASM is mature; chDB has no WASM target.

3. **You depend on DuckDB-specific extensions** — `spatial` for PostGIS-style geometric operations, FUGUE-style distributed Python workflows, or the experimental ML scoring functions.

4. **You have heavy `STRUCT` / nested-list pipelines** that lean on DuckDB's list lambdas. ClickHouse `Array` / `Tuple` / `Map` cover the same ground but with different ergonomics; existing list-lambda code will need rewriting.

5. **You run in a RAM-constrained sandbox** (small Lambda, small Modal container) and your workload is dominated by simple aggregations on small files. chDB has a larger startup baseline RSS — roughly 100–200 MB more than DuckDB before any meaningful work — for a 5-second Lambda doing a 10 MB query, that matters. (On heavier workloads, DuckDB's CTE materialisations can push its peak RSS above chDB's — see §5.)

6. **Your tooling is locked to PostgreSQL dialect.** Tools that auto-generate PostgreSQL DDL or rely on the `pg_catalog` introspection surface will need adapters.

If you fall into more than one of the above, defer migration. chDB can sit alongside DuckDB in the same project — the engines do not conflict.

---

## 4. Migration SOP

The migration is mechanical and lossless for most plain analytical SQL. Plan for about a day of work for a mid-sized notebook codebase, plus more for SQL dialect cleanup if you use advanced functions.

### Step 1 — Audit

Run the static helper that ships with this guide. Grab it from the [cookbook entry](https://github.com/chdb-io/cookbook/tree/main/migration-from-duckdb) — [`migrate.py`](https://github.com/chdb-io/cookbook/blob/main/migration-from-duckdb/migrate.py) is a single self-contained script.

```bash
python migrate.py path/to/your/project
```

`migrate.py` scans `.py`, `.sql`, and `.ipynb` files for DuckDB API calls and SQL constructs that need attention. It produces a checklist of touch points: one-line mechanical replacements (import, `read_parquet`, `date_trunc`), and "dialect review" flags for harder cases (`PIVOT`, `INSTALL`, `STRUCT`).

A representative run on a small notebook-style script:

```text
$ python migrate.py /path/to/project
Found 7 migration touch points across 1 files:

--- /path/to/project/analytics.py
  L   1  import
        before:  import duckdb
        after :  import chdb
                 from chdb import session as chdb_session
  L   4  connect
        before:  con = duckdb.connect()
        after :  con = chdb_session.Session()
  L   7  date_trunc('hour', x)
        before:  SELECT date_trunc('hour', ts) AS h,
        after :  SELECT toStartOfHour(ts) AS h,
  L   8  approx_count_distinct
        before:  approx_count_distinct(user_id) AS uniq,
        after :  uniqHLL12(user_id) AS uniq,
  L  10  read_parquet
        before:  FROM read_parquet('events.parquet')
        after :  FROM file('events.parquet', 'Parquet')
  L  16  register(name, df) → Python(df) table function
        before:  con.register('users', df)
        after :  reference the DataFrame as Python(df) directly in the SQL
  L  17  execute().df() → query(.., 'DataFrame')
        before:  top = con.execute("SELECT * FROM users LIMIT 10").df()
        after :  top = con.query("SELECT * FROM users LIMIT 10", 'DataFrame')
```

Treat the output as a checklist for human review, not an auto-correct. `--apply` will rewrite the unambiguous mechanical replacements in place (originals saved to `*.bak`), but the `Python(df)` migration and any `PIVOT` / `INSTALL` / `STRUCT` flagged by `--dialect-only` need your judgment about call-site shape.

### Step 2 — Install chDB

```bash
pip install chdb
```

chDB ships as a single wheel; no system packages, no daemon. Confirm it works:

```python
import chdb
print(chdb.query("SELECT version()", "CSV"))
```

### Step 3 — Mechanical Python rewrites

| DuckDB | chDB |
|---|---|
| `import duckdb` | `import chdb` + `from chdb import session as chdb_session` |
| `con = duckdb.connect()` | `sess = chdb_session.Session()` |
| `con = duckdb.connect('path.db')` | `sess = chdb_session.Session('path/dir')` (a directory, not a file) |
| `con.execute(sql).fetchall()` | `sess.query(sql, 'CSV')` (review return shape) |
| `con.execute(sql).df()` | `sess.query(sql, 'DataFrame')` |
| `con.register('df', pdf)`<br>`con.execute("SELECT * FROM df")` | `sess.query("SELECT * FROM Python(pdf)", 'DataFrame')` — no register step; `pdf` is picked up from caller scope |

One thing to know about `Session`: the argument is a **directory**, not a single file — ClickHouse storage is directory-based.

### Step 4 — SQL dialect rewrites

The most common ones (`migrate.py` flags these):

| DuckDB SQL | chDB SQL |
|---|---|
| `read_parquet(...)` / `read_csv(...)` / `read_json(...)` | `file('path', 'Parquet' \| 'CSV' \| 'JSONEachRow')` |
| `date_trunc('hour' \| 'day' \| 'month', ts)` | `toStartOfHour(ts)` / `toStartOfDay(ts)` / `toStartOfMonth(ts)` |
| `strftime(ts, '%Y-%m-%d')` | `formatDateTime(ts, '%Y-%m-%d')` |
| `approx_count_distinct(x)` | `uniqHLL12(x)` (or `uniq(x)`; `uniqExact(x)` for exact) |
| `regexp_matches(x, 'pat')` | `match(x, 'pat')` |
| `LIST(x)` / `ARRAY_AGG(x)` / `UNNEST(arr)` | `groupArray(x)` / `groupArrayInsertAt(...)` / `arrayJoin(arr)` |
| `PIVOT` | Conditional aggregation: `sumIf(x, key='A') AS A, sumIf(x, key='B') AS B` |

Approximation precision differs slightly between `approx_count_distinct` and `uniqHLL12`, and across the quantile families — results agree within their error bounds, not bit-identical. Use `uniqExact` / `quantileExact` when exact agreement matters.

### Step 5 — Pre-load for complex repeated analytics

For one-shot queries against Parquet, `file('path', 'Parquet')` is fine. For repeated complex aggregations (`windowFunnel`, `uniq` over very large tables, joins between two large files), pre-load into a MergeTree-backed table in the Session:

```python
sess.query("""
    CREATE TABLE trips
    ENGINE = MergeTree ORDER BY tpep_pickup_datetime AS
    SELECT
        PULocationID,
        assumeNotNull(toDateTime(tpep_pickup_datetime)) AS tpep_pickup_datetime,
        fare_amount
    FROM file('yellow_tripdata_*.parquet', 'Parquet')
""")
```

Two pitfalls worth knowing up front:

- **`ORDER BY` cannot include nullable columns** by default. Wrap with `assumeNotNull` (if the column is genuinely never null) or set `SETTINGS allow_nullable_key = 1` on the table.
- **Type widening on Parquet load.** Parquet `timestamp[us]` becomes `Nullable(DateTime64(6, 'UTC'))` in chDB. Some aggregate functions (notably `windowFunnel`) only accept non-nullable `DateTime` — cast with `assumeNotNull(toDateTime(ts))`.

### Step 6 — Validation

Run both engines side by side on a few representative queries. Compare:

1. **Row counts.** Should match exactly.
2. **Aggregate values.** Numeric exact matches expected; HLL/quantile aggregates within their stated error bounds.
3. **NULL handling.** Arithmetic and comparison semantics are the same on both sides (`NULL + 1` is `NULL`, `NULL = NULL` is `NULL`), but the helper-function names differ: DuckDB's `COALESCE` works on both; ClickHouse adds `ifNull(x, default)` and `assumeNotNull(x)` as the idiomatic spellings. Watch for these during the dialect rewrite.

The two workload files in the cookbook entry — [`workload_aligned_duckdb.py`](https://github.com/chdb-io/cookbook/blob/main/migration-from-duckdb/benchmark/workload_aligned_duckdb.py) and [`workload_aligned_chdb.py`](https://github.com/chdb-io/cookbook/blob/main/migration-from-duckdb/benchmark/workload_aligned_chdb.py) — are a worked example: the migrated workload on both engines, side-by-side syntax differences.

---

## 5. The benchmark: 16-query head-to-head

We ran this migration on a real workload to validate the SOP. The full code (`workload_aligned_duckdb.py`, `workload_aligned_chdb.py`, `run_aligned.py`, `migrate.py`, `gen_data.py`) lives in the [`chdb-io/cookbook` migration-from-duckdb entry](https://github.com/chdb-io/cookbook/tree/main/migration-from-duckdb); you can reproduce the numbers in roughly 5 minutes from a clean checkout.

### Machine

| | |
|---|---|
| Hardware | MacBook Pro, Apple M5 Max, 18 cores (6P + 12E), 36 GB RAM |
| OS | macOS 26.4.1 (Build 25E253) |
| Python | 3.9.6 |
| DuckDB | 1.4.4 |
| chDB | 4.1.6 |

### Dataset

| | |
|---|---|
| Source | NYC TLC Yellow Taxi |
| Months | 2024-01 through 2024-06 |
| Files | 6 Parquet files |
| Size | 326 MB total |
| Rows | ~18 million |

Downloaded directly from `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-{01..06}.parquet`.

### Workload — sixteen queries

The 16 kernel queries cover three categories. Q1–Q8 exercise the §2 chDB use cases. Q9–Q15 are baseline analytical SQL and reference comparisons — the conventional OLAP-on-Parquet kernel. Q16 replicates chDB's published "DataFrame export" claim. §2.4 / §2.5 / §2.6 / §2.7 are integration / workflow / storage features rather than per-query timings, so they don't appear as benchmark rows.

| # | §2 reason | Query | What it stresses |
|---|---|---|---|
| Q1 | §2.1 typed JSON | `GROUP BY data.response.status` on 1 M events | Single JSON sub-column path read |
| Q2 | §2.1 typed JSON | `GROUP BY data.user.tier, avg(data.response.latency_ms)` | Two paths, one numeric |
| Q3 | §2.1 typed JSON | `WHERE data.user.region = X GROUP BY data.tool` | Filter + group on JSON paths |
| Q4 | §2.2 pandas-compat | DataStore drop-in `groupby + agg` on 3 M rows | chDB-only feature; no DuckDB analogue |
| Q5 | §2.3 AI-agent retrieval | Top-10 cosine-distance neighbors over 100 K × 384-d vectors | Native vector type + `cosineDistance()` |
| Q6 | §2.8 funnel/pattern | Three-step funnel per pickup zone within 1 h | `windowFunnel` vs CTE simulation |
| Q7 | §2.8 funnel/pattern | Count of `low → high → very-high` fare sequences | `sequenceCount('(?1)(?2)(?3)')(...)` vs LAG-CTE |
| Q8 | §2.9 many percentiles | p50 / p95 / p99 of `fare_amount` | `quantilesTDigest(...)` single sketch vs DuckDB's `approx_quantile(x, [...])` list form (also single sketch) |
| Q9 | baseline | `count() / sum() / avg()` over the 6-month glob | Parquet scan throughput |
| Q10 | baseline | `GROUP BY payment_type WHERE fare_amount > 50` | Filter + group |
| Q11 | baseline | Hourly trip volume via `date_trunc` / `toStartOfHour` | SQL dialect translation |
| Q12 | baseline | `approx_count_distinct` / `uniqHLL12` | HLL implementation |
| Q13 | baseline | Register a 3 M-row DataFrame (4 cols), group by, return DataFrame | DataFrame round-trip — narrow |
| Q14 | reference | Top-10 most popular pickup zones | `topK(10)` (chDB, approx) vs `GROUP BY+ORDER+LIMIT` (DuckDB, exact) |
| Q15 | reference | Same DataFrame round-trip but all 19 columns | DataFrame round-trip — wide |
| Q16 | special | `SELECT * FROM <parquet>` → full pandas DataFrame, single file | Parquet → DataFrame export path |

### Results — median of 3 runs each

```
Query                                       DuckDB        chDB    speedup
--------------------------------------------------------------------------
=== use-case section (§2-aligned) ===
Q1  JSON group response.status              35.0 ms      4.2 ms     8.42x  ← chDB
Q2  JSON group tier + avg(latency)          65.5 ms     21.7 ms     3.02x  ← chDB
Q3  JSON filter + group                     36.7 ms      9.8 ms     3.75x  ← chDB
Q4  DataStore drop-in groupby               (chDB-only feature; no DuckDB equivalent)
Q5  vector top-10 cosine                    34.7 ms     63.5 ms     0.55x  ← DuckDB
Q6  funnel (windowFunnel vs CTE)           273.7 ms    105.0 ms     2.61x  ← chDB
Q7  sequence pattern                       229.6 ms     50.6 ms     4.54x  ← chDB
Q8  multi-quantile (p50/p95/p99)            63.8 ms     27.1 ms     2.35x  ← chDB

=== baseline analytical SQL ===
Q9  aggregate                               14.3 ms     19.3 ms     0.74x  ← DuckDB
Q10 group+filter                            14.3 ms     30.0 ms     0.48x  ← DuckDB
Q11 time bucket                             37.3 ms     30.5 ms     1.22x  ← chDB
Q12 approx count distinct                   11.3 ms     23.6 ms     0.48x  ← DuckDB
Q13 DataFrame roundtrip (narrow, 4 cols)     6.3 ms     12.7 ms     0.50x  ← DuckDB

=== reference queries ===
Q14 topK pickup zones                        7.5 ms     11.9 ms     0.63x  ← DuckDB
Q15 DataFrame roundtrip (wide, 19 cols)      7.7 ms     12.8 ms     0.60x  ← DuckDB

=== special ===
Q16 Parquet -> DataFrame export            391.7 ms    243.9 ms     1.61x  ← chDB
```

**Different strengths, different territories.** chDB wins decisively on the agent-workload patterns — typed JSON path access (Q1–Q3, **3.0–8.4×**), funnel and sequence aggregates (Q6, Q7, **2.6–4.5×**), multi-percentile (Q8, **2.35×** against DuckDB's optimal list-form API), time-bucket via `toStartOfHour` (Q11, 1.22×), and Parquet → DataFrame export (Q16, **1.61× cold / 2.99× warm**) — multi-fold speedups in every case it wins. DuckDB wins the conventional Parquet-aggregation kernels (Q9, Q10, Q12) and the DataFrame round-trip path (Q13, Q14, Q15), all within a tight **1.3–2.1× envelope** — its kernel is well-tuned for the classic OLAP-on-Parquet shape. Q5 (raw cosine kernel) is a DuckDB kernel win that the workflow-integration argument reframes — see cookbook [BENCHMARK.md](https://github.com/chdb-io/cookbook/blob/main/migration-from-duckdb/BENCHMARK.md) Case C. Q4 is a chDB-only DataStore feature, not a head-to-head.

Peak RSS during the workload (last query of each phase):

| Engine | Q1–Q5 use case | Q6–Q8 specialty | Q9–Q15 baseline | Q16 export |
|---|---|---|---|---|
| DuckDB | ~1.0 GB | ~2.3 GB | ~3.8 GB | ~4.9 GB |
| chDB | ~1.2 GB | ~1.5 GB | ~2.4 GB | ~4.5 GB |

### Deeper analysis and reproduction

Per-case studies with side-by-side SQL (typed JSON, DataStore, vector retrieval, `windowFunnel`, `sequenceCount`, `quantilesTDigest`, Parquet → DataFrame export), the ingest-path methodology, the DataFrame round-trip operation matrix, and full reproduction instructions live in the cookbook:

→ <https://github.com/chdb-io/cookbook/blob/main/migration-from-duckdb/BENCHMARK.md>

Runnable code (`workload_aligned_duckdb.py`, `workload_aligned_chdb.py`, `run_aligned.py`, `migrate.py`, `gen_data.py`) and the canonical `results_aligned.json` are in the same directory.

---

## 6. Things this guide does not cover

- **UDFs.** DuckDB Python UDFs migrate to chDB's `@chdb_udf` decorator; the surface area is similar but per-call overhead and threading model differ. Worth its own guide.
- **DuckDB extension surface.** Per-extension migration paths (spatial, FUGUE, ML) need a feasibility check before committing to migration.
- **Database file persistence.** A DuckDB `.db` file does not transfer; you re-export to Parquet and re-import to a chDB `Session` directory.
- **Cluster mode.** Both engines are single-process. If you need a server, you are looking at full ClickHouse (or DuckDB's hosted cloud variant on the DuckDB side), not at either embedded engine.

---

## Appendix — `migrate.py` cheat sheet

Source: [chdb-io/cookbook/migration-from-duckdb/migrate.py](https://github.com/chdb-io/cookbook/blob/main/migration-from-duckdb/migrate.py).

```
python migrate.py PROJECT_DIR              # report touch points only
python migrate.py PROJECT_DIR --apply      # rewrite simple cases in place (backs up to *.bak)
python migrate.py PROJECT_DIR --dialect-only  # focus on SQL dialect risks
```

The script handles the mechanical replacements in §4. It flags but does not rewrite the dialect-review items (`PIVOT`, `INSTALL`, `STRUCT`, `CREATE INDEX`) — those need human judgment about how the equivalent should look in your codebase.
