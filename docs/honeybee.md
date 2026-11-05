# HoneyBee retrieval

The adapter builds a workload for HoneyBee-VectorAccess. Each indexed block is one source-grounded fact. A principal represents a patient, scenario and effective set of authorized facts. Paired requests with different access rights use different database principals. A principal with empty scope receives zero blocks. [Upstream code](https://github.com/rjzhb/HoneyBee-VectorAccess/tree/efc1911aa85d1b02593af9c117af7caa01e5c4ed)

```shell
python -m perm2ask.honeybee.bridge export --out outputs/honeybee_export
```

The exporter reads the benchmark requests, subject records and policy. It generates a 384-dimensional query vector from each original question and records file hashes in the export manifest. The same query is reused across turns in this adaptation. The benchmark gold file stays outside the database.

## Build the upstream index

1. Check out HoneyBee-VectorAccess at commit `efc1911aa85d1b02593af9c117af7caa01e5c4ed`. Follow its [installation instructions](https://github.com/rjzhb/HoneyBee-VectorAccess/blob/efc1911aa85d1b02593af9c117af7caa01e5c4ed/README.md) for PostgreSQL, its pgvector fork and the upstream configuration.
2. In the upstream repository, initialize a dedicated empty benchmark database with `python -c "from controller.initialize_main_tables import initialize_database_deduplication as init; init(enable_index=True, vector_dimension=384)"`.
3. From the export directory, run `psql -d DATABASE -f /path/to/PerM2ask/scripts/load_honeybee.psql` to load the six workload tables.
4. In the upstream `controller/dynamic_partition/hnsw` directory, run `python AnonySys_dynamic_partition.py --storage 2.0 --recall 0.95`. Record the index parameters alongside the run.
5. From the PerM²ask repository root, prepare the benchmark principals and retrieve evidence:

```shell
python -m perm2ask.honeybee.prepare_logins --upstream /path/to/HoneyBee-VectorAccess --export outputs/honeybee_export
python -m perm2ask.honeybee.bridge query --upstream /path/to/HoneyBee-VectorAccess --export outputs/honeybee_export --out outputs/honeybee_hits.jsonl
```

Pass those hits to `perm2ask.generation` with `--method honeybee_external --contexts outputs/honeybee_hits.jsonl`. The adapter checks the upstream commit, indexed roles, fact identifiers, source spans and permission assignments. Generation checks the query text and input hashes before accepting a result. This distribution includes the workload exporter and query adapter; the upstream database is a separate deployment.
