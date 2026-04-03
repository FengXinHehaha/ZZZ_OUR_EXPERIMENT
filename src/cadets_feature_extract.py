import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from cadets_event_ingest_common import build_common_parser, connect_target_db, load_ingest_config


DEFAULT_SPLIT_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "day_split" / "split_manifest.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "features"
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S UTC"

PROCESS_VIEW_FILE = "process_view__process_node.tsv"
PROCESS_VIEW_FILE_NODE_FILE = "process_view__file_node.tsv"
PROCESS_VIEW_NETWORK_NODE_FILE = "process_view__network_node.tsv"
FILE_VIEW_FILE = "file_view__file_node.tsv"
FILE_VIEW_PROCESS_NODE_FILE = "file_view__process_node.tsv"
NETWORK_VIEW_FILE = "network_view__network_node.tsv"
NETWORK_VIEW_PROCESS_NODE_FILE = "network_view__process_node.tsv"


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_split_manifest(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def utc_day_bounds(day: str) -> Tuple[int, int]:
    start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return int(start.timestamp() * 1_000_000_000), int(end.timestamp() * 1_000_000_000)


def build_feature_windows(split_manifest: Dict[str, object]) -> List[Dict[str, object]]:
    split = split_manifest["recommended_split"]
    train_days = list(split["train_days"])
    val_days = list(split["val_days"])
    test_days = list(split["test_days"])

    windows: List[Dict[str, object]] = []
    if train_days:
        windows.append({"name": "train", "split": "train", "days": train_days})
    if val_days:
        windows.append({"name": "val", "split": "val", "days": val_days})
    for day in test_days:
        windows.append({"name": f"test_{day}", "split": "test", "days": [day]})
    return windows


def window_bounds(days: List[str]) -> Tuple[int, int]:
    start_ns, _ = utc_day_bounds(days[0])
    _, end_ns = utc_day_bounds(days[-1])
    return start_ns, end_ns


def ns_to_text(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).strftime(TIMESTAMP_FMT)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def log(message: str) -> None:
    print(message, flush=True)


def normalized_text_sql(expr: str) -> str:
    return f"LOWER(BTRIM(COALESCE({expr}, '')))"


def missing_text_sql(expr: str) -> str:
    return f"NULLIF(BTRIM(COALESCE({expr}, '')), '') IS NULL"


def exec_name_condition(expr: str, category: str) -> str:
    normalized = normalized_text_sql(expr)
    if category == "shell":
        return normalized + " ~ '(^|.*/)(sh|bash|dash|zsh|ksh|csh|tcsh)$'"
    if category == "interpreter":
        return normalized + " ~ '(^|.*/)(python([0-9.]*)?|perl|ruby|php|node|java)$'"
    if category == "network_tool":
        return normalized + " ~ '(^|.*/)(curl|wget|nc|ncat|netcat|ssh|scp|sftp|ftp|telnet|socat)$'"
    if category == "package_tool":
        return normalized + " ~ '(^|.*/)(apt|apt-get|yum|dnf|pip|npm|dpkg|rpm)$'"
    if category == "system_tool":
        return normalized + " ~ '(^|.*/)(sudo|su|systemctl|service|mount|chmod|chown|cron)$'"
    raise ValueError(f"Unsupported exec-name category: {category}")


def file_path_condition(expr: str, category: str) -> str:
    normalized = normalized_text_sql(expr)
    if category == "temp":
        return (
            f"{normalized} LIKE '/tmp/%%' OR {normalized} LIKE '/var/tmp/%%' OR {normalized} LIKE '/dev/shm/%%'"
        )
    if category == "config":
        return f"{normalized} LIKE '/etc/%%'"
    if category == "system_bin":
        return (
            f"{normalized} LIKE '/usr/bin/%%' OR {normalized} LIKE '/bin/%%' "
            f"OR {normalized} LIKE '/usr/sbin/%%' OR {normalized} LIKE '/sbin/%%'"
        )
    if category == "system_lib":
        return (
            f"{normalized} LIKE '/usr/lib/%%' OR {normalized} LIKE '/lib/%%' "
            f"OR {normalized} LIKE '/lib64/%%' OR {normalized} LIKE '/usr/lib64/%%'"
        )
    if category == "log":
        return f"{normalized} LIKE '/var/log/%%'"
    if category == "user_home":
        return f"{normalized} LIKE '/home/%%' OR {normalized} LIKE '/root/%%'"
    if category == "hidden":
        return f"REGEXP_REPLACE({normalized}, '^.*/', '') LIKE '.%%'"
    if category == "script":
        return normalized + " ~ '\\.(sh|bash|zsh|py|pl|rb|php|js)$'"
    raise ValueError(f"Unsupported file-path category: {category}")


def tune_feature_session(conn) -> None:
    with conn.cursor() as cursor:
        cursor.execute("SET work_mem TO '1GB'")
        cursor.execute("SET maintenance_work_mem TO '1GB'")
        cursor.execute("SET max_parallel_workers_per_gather TO 4")
        cursor.execute("SET jit TO OFF")
    conn.commit()


def count_tsv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        line_count = sum(1 for _ in handle)
    return max(0, line_count - 1)


def copy_query_to_tsv(
    conn,
    query: str,
    params: Tuple[object, ...],
    output_path: Path,
    log_label: str,
) -> int:
    log(f"[feature-extract] {log_label}: COPY export -> {output_path.name}")
    with conn.cursor() as cursor:
        rendered_query = cursor.mogrify(query, params).decode("utf-8")
        copy_sql = (
            "COPY ("
            + rendered_query
            + ") TO STDOUT WITH (FORMAT CSV, HEADER TRUE, DELIMITER E'\\t')"
        )
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            cursor.copy_expert(copy_sql, handle)
            handle.flush()
    row_count = count_tsv_rows(output_path)
    log(f"[feature-extract] {log_label}: rows={row_count:,}")
    return row_count


PROCESS_VIEW_EXPORT_FROM_TEMP_QUERY = """
WITH event_type_counts AS (
    SELECT node_uuid, COUNT(*)::bigint AS event_type_diversity
    FROM tmp_process_event_types
    GROUP BY node_uuid
),
file_counts AS (
    SELECT node_uuid, COUNT(*)::bigint AS unique_file_count
    FROM tmp_process_file_refs
    GROUP BY node_uuid
),
network_counts AS (
    SELECT node_uuid, COUNT(*)::bigint AS unique_network_count
    FROM tmp_process_network_refs
    GROUP BY node_uuid
),
exec_name_counts AS (
    SELECT node_uuid, COUNT(*)::bigint AS unique_exec_name_count
    FROM tmp_process_exec_names
    GROUP BY node_uuid
)
SELECT
    s.node_uuid,
    'process'::text AS node_type,
    COALESCE(p.host_id, '') AS host_id,
    COALESCE(p.subject_type, '') AS subject_type,
    CASE
        WHEN p.parent_subject_uuid IS NULL OR p.parent_subject_uuid = '' THEN 0
        ELSE 1
    END AS has_parent_flag,
    s.total_events,
    COALESCE(et.event_type_diversity, 0)::bigint AS event_type_diversity,
    COALESCE(fc.unique_file_count, 0)::bigint AS unique_file_count,
    COALESCE(nc.unique_network_count, 0)::bigint AS unique_network_count,
    s.read_count,
    s.write_count,
    s.open_count,
    s.execute_count,
    s.connect_count,
    s.send_count,
    s.recv_count,
    s.accept_count,
    s.create_object_count,
    s.fork_count,
    s.mmap_count,
    s.modify_process_count,
    s.close_count,
    COALESCE(enc.unique_exec_name_count, 0)::bigint AS unique_exec_name_count,
    s.shell_exec_count,
    s.interpreter_exec_count,
    s.network_tool_exec_count,
    s.package_tool_exec_count,
    s.system_tool_exec_count,
    s.missing_exec_name_count,
    ROUND(
        (s.read_count + s.write_count + s.open_count + s.execute_count)::numeric / NULLIF(s.total_events, 0),
        6
    ) AS file_interaction_ratio,
    ROUND(
        (s.connect_count + s.send_count + s.recv_count + s.accept_count)::numeric / NULLIF(s.total_events, 0),
        6
    ) AS network_interaction_ratio,
    ROUND(s.fork_count::numeric / NULLIF(s.total_events, 0), 6) AS fork_ratio
FROM tmp_process_stats AS s
JOIN process_entities AS p
  ON p.uuid = s.node_uuid
LEFT JOIN event_type_counts AS et
  ON et.node_uuid = s.node_uuid
LEFT JOIN file_counts AS fc
  ON fc.node_uuid = s.node_uuid
LEFT JOIN network_counts AS nc
  ON nc.node_uuid = s.node_uuid
LEFT JOIN exec_name_counts AS enc
  ON enc.node_uuid = s.node_uuid
"""


FILE_VIEW_QUERY = f"""
WITH file_hits AS (
    SELECT
        e.subject_uuid AS process_uuid,
        e.object_uuid AS node_uuid,
        e.event_type,
        COALESCE(
            NULLIF(BTRIM(COALESCE(e.object_path, '')), ''),
            NULLIF(BTRIM(COALESCE(e.file_descriptor, '')), ''),
            ''
        ) AS path_text
    FROM events_raw AS e
    JOIN file_entities AS f
      ON f.uuid = e.object_uuid
    WHERE e.timestamp_ns >= %s
      AND e.timestamp_ns < %s
    UNION ALL
    SELECT
        e.subject_uuid AS process_uuid,
        e.object2_uuid AS node_uuid,
        e.event_type,
        COALESCE(
            NULLIF(BTRIM(COALESCE(e.object2_path, '')), ''),
            NULLIF(BTRIM(COALESCE(e.file_descriptor, '')), ''),
            ''
        ) AS path_text
    FROM events_raw AS e
    JOIN file_entities AS f
      ON f.uuid = e.object2_uuid
    WHERE e.timestamp_ns >= %s
      AND e.timestamp_ns < %s
)
SELECT
    h.node_uuid,
    'file'::text AS node_type,
    COALESCE(f.host_id, '') AS host_id,
    COALESCE(f.file_type, '') AS file_type,
    COALESCE(f.size_bytes, 0)::bigint AS size_bytes,
    COUNT(*)::bigint AS total_accesses,
    COUNT(DISTINCT h.process_uuid)::bigint AS unique_process_count,
    COUNT(DISTINCT h.event_type)::bigint AS event_type_diversity,
    COUNT(DISTINCT NULLIF(BTRIM(COALESCE(h.path_text, '')), ''))::bigint AS unique_known_path_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_READ')::bigint AS read_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_WRITE')::bigint AS write_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_OPEN')::bigint AS open_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_EXECUTE')::bigint AS execute_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_CLOSE')::bigint AS close_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_CREATE_OBJECT')::bigint AS create_object_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_UNLINK')::bigint AS unlink_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_RENAME')::bigint AS rename_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_MODIFY_FILE_ATTRIBUTES')::bigint AS modify_file_attr_count,
    COUNT(*) FILTER (WHERE {file_path_condition("h.path_text", "temp")})::bigint AS temp_path_count,
    COUNT(*) FILTER (WHERE {file_path_condition("h.path_text", "config")})::bigint AS config_path_count,
    COUNT(*) FILTER (WHERE {file_path_condition("h.path_text", "system_bin")})::bigint AS system_bin_path_count,
    COUNT(*) FILTER (WHERE {file_path_condition("h.path_text", "system_lib")})::bigint AS system_lib_path_count,
    COUNT(*) FILTER (WHERE {file_path_condition("h.path_text", "log")})::bigint AS log_path_count,
    COUNT(*) FILTER (WHERE {file_path_condition("h.path_text", "user_home")})::bigint AS user_home_path_count,
    COUNT(*) FILTER (WHERE {file_path_condition("h.path_text", "hidden")})::bigint AS hidden_path_count,
    COUNT(*) FILTER (WHERE {file_path_condition("h.path_text", "script")})::bigint AS script_path_count,
    COUNT(*) FILTER (WHERE {missing_text_sql("h.path_text")})::bigint AS missing_path_count,
    ROUND((COUNT(*) FILTER (WHERE h.event_type = 'EVENT_READ'))::numeric / NULLIF(COUNT(*), 0), 6) AS read_ratio,
    ROUND((COUNT(*) FILTER (WHERE h.event_type = 'EVENT_WRITE'))::numeric / NULLIF(COUNT(*), 0), 6) AS write_ratio,
    ROUND((COUNT(*) FILTER (WHERE h.event_type = 'EVENT_EXECUTE'))::numeric / NULLIF(COUNT(*), 0), 6) AS execute_ratio,
    ROUND((COUNT(*) FILTER (WHERE h.event_type = 'EVENT_OPEN'))::numeric / NULLIF(COUNT(*), 0), 6) AS open_ratio
FROM file_hits AS h
JOIN file_entities AS f
  ON f.uuid = h.node_uuid
GROUP BY h.node_uuid, f.host_id, f.file_type, f.size_bytes
"""


NETWORK_VIEW_QUERY = """
WITH network_hits AS (
    SELECT
        e.subject_uuid AS process_uuid,
        e.object_uuid AS node_uuid,
        e.event_type
    FROM events_raw AS e
    JOIN network_entities AS n
      ON n.uuid = e.object_uuid
    WHERE e.timestamp_ns >= %s
      AND e.timestamp_ns < %s
    UNION ALL
    SELECT
        e.subject_uuid AS process_uuid,
        e.object2_uuid AS node_uuid,
        e.event_type
    FROM events_raw AS e
    JOIN network_entities AS n
      ON n.uuid = e.object2_uuid
    WHERE e.timestamp_ns >= %s
      AND e.timestamp_ns < %s
)
SELECT
    h.node_uuid,
    'network'::text AS node_type,
    COALESCE(n.host_id, '') AS host_id,
    COALESCE(n.local_address, '') AS local_address,
    COALESCE(n.remote_address, '') AS remote_address,
    COALESCE(n.local_port, -1) AS local_port,
    COALESCE(n.remote_port, -1) AS remote_port,
    CASE
        WHEN n.local_port IS NULL THEN 'unknown'
        WHEN n.local_port < 0 THEN 'invalid'
        WHEN n.local_port <= 1023 THEN 'well_known'
        WHEN n.local_port <= 49151 THEN 'registered'
        ELSE 'ephemeral'
    END AS local_port_bucket,
    CASE
        WHEN n.remote_port IS NULL THEN 'unknown'
        WHEN n.remote_port < 0 THEN 'invalid'
        WHEN n.remote_port <= 1023 THEN 'well_known'
        WHEN n.remote_port <= 49151 THEN 'registered'
        ELSE 'ephemeral'
    END AS remote_port_bucket,
    CASE
        WHEN NULLIF(BTRIM(COALESCE(n.remote_address, '')), '') IS NULL THEN 'unknown'
        WHEN n.remote_address !~ '^[0-9]{1,3}(\\.[0-9]{1,3}){3}$' THEN 'unknown'
        WHEN inet(n.remote_address) << inet '10.0.0.0/8'
          OR inet(n.remote_address) << inet '172.16.0.0/12'
          OR inet(n.remote_address) << inet '192.168.0.0/16'
          OR inet(n.remote_address) << inet '127.0.0.0/8'
          OR inet(n.remote_address) << inet '169.254.0.0/16'
        THEN 'no'
        ELSE 'yes'
    END AS external_remote_ip_flag,
    COUNT(*)::bigint AS total_net_events,
    COUNT(DISTINCT h.process_uuid)::bigint AS unique_process_count,
    COUNT(DISTINCT h.event_type)::bigint AS event_type_diversity,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_CONNECT')::bigint AS connect_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_ACCEPT')::bigint AS accept_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_BIND')::bigint AS bind_count,
    COUNT(*) FILTER (WHERE h.event_type IN ('EVENT_SENDTO', 'EVENT_SENDMSG'))::bigint AS send_count,
    COUNT(*) FILTER (WHERE h.event_type IN ('EVENT_RECVFROM', 'EVENT_RECVMSG'))::bigint AS recv_count,
    ROUND(
        (COUNT(*) FILTER (WHERE h.event_type IN ('EVENT_SENDTO', 'EVENT_SENDMSG')))::numeric /
        NULLIF(COUNT(*) FILTER (WHERE h.event_type IN ('EVENT_RECVFROM', 'EVENT_RECVMSG')), 0),
        6
    ) AS send_recv_ratio,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_SENDMSG')::bigint AS message_send_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_RECVMSG')::bigint AS message_recv_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_CLOSE')::bigint AS close_count
FROM network_hits AS h
JOIN network_entities AS n
  ON n.uuid = h.node_uuid
GROUP BY
    h.node_uuid,
    n.host_id,
    n.local_address,
    n.remote_address,
    n.local_port,
    n.remote_port
"""


PROCESS_VIEW_FILE_NODE_QUERY = f"""
WITH file_hits AS (
    SELECT
        e.subject_uuid AS process_uuid,
        e.object_uuid AS node_uuid,
        e.event_type,
        COALESCE(
            NULLIF(BTRIM(COALESCE(e.object_path, '')), ''),
            NULLIF(BTRIM(COALESCE(e.file_descriptor, '')), ''),
            ''
        ) AS path_text
    FROM events_raw AS e
    JOIN file_entities AS f
      ON f.uuid = e.object_uuid
    WHERE e.timestamp_ns >= %s
      AND e.timestamp_ns < %s
    UNION ALL
    SELECT
        e.subject_uuid AS process_uuid,
        e.object2_uuid AS node_uuid,
        e.event_type,
        COALESCE(
            NULLIF(BTRIM(COALESCE(e.object2_path, '')), ''),
            NULLIF(BTRIM(COALESCE(e.file_descriptor, '')), ''),
            ''
        ) AS path_text
    FROM events_raw AS e
    JOIN file_entities AS f
      ON f.uuid = e.object2_uuid
    WHERE e.timestamp_ns >= %s
      AND e.timestamp_ns < %s
),
process_network_stats AS (
    SELECT
        e.subject_uuid AS process_uuid,
        (
            COUNT(*) FILTER (WHERE e.event_type = 'EVENT_CONNECT') +
            COUNT(*) FILTER (WHERE e.event_type = 'EVENT_ACCEPT') +
            COUNT(*) FILTER (WHERE e.event_type IN ('EVENT_SENDTO', 'EVENT_SENDMSG')) +
            COUNT(*) FILTER (WHERE e.event_type IN ('EVENT_RECVFROM', 'EVENT_RECVMSG'))
        )::bigint AS total_network_events
    FROM events_raw AS e
    JOIN process_entities AS p
      ON p.uuid = e.subject_uuid
    WHERE e.timestamp_ns >= %s
      AND e.timestamp_ns < %s
    GROUP BY e.subject_uuid
)
SELECT
    h.node_uuid,
    'file'::text AS node_type,
    COALESCE(f.host_id, '') AS host_id,
    COALESCE(f.file_type, '') AS file_type,
    COALESCE(f.size_bytes, 0)::bigint AS size_bytes,
    COUNT(*)::bigint AS total_process_context_events,
    COUNT(DISTINCT h.process_uuid)::bigint AS unique_process_count,
    COUNT(DISTINCT h.event_type)::bigint AS event_type_diversity,
    COUNT(DISTINCT NULLIF(BTRIM(COALESCE(h.path_text, '')), ''))::bigint AS unique_known_path_count,
    COUNT(DISTINCT h.process_uuid) FILTER (WHERE h.event_type = 'EVENT_READ')::bigint AS read_by_process_count,
    COUNT(DISTINCT h.process_uuid) FILTER (WHERE h.event_type = 'EVENT_WRITE')::bigint AS write_by_process_count,
    COUNT(DISTINCT h.process_uuid) FILTER (WHERE h.event_type = 'EVENT_OPEN')::bigint AS open_by_process_count,
    COUNT(DISTINCT h.process_uuid) FILTER (WHERE h.event_type = 'EVENT_EXECUTE')::bigint AS exec_by_process_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_READ')::bigint AS read_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_WRITE')::bigint AS write_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_OPEN')::bigint AS open_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_EXECUTE')::bigint AS execute_count,
    COUNT(*) FILTER (WHERE {file_path_condition("h.path_text", "temp")})::bigint AS temp_path_count,
    COUNT(*) FILTER (WHERE {file_path_condition("h.path_text", "config")})::bigint AS config_path_count,
    COUNT(*) FILTER (WHERE {file_path_condition("h.path_text", "system_bin")})::bigint AS system_bin_path_count,
    COUNT(*) FILTER (WHERE {file_path_condition("h.path_text", "system_lib")})::bigint AS system_lib_path_count,
    COUNT(*) FILTER (WHERE {file_path_condition("h.path_text", "log")})::bigint AS log_path_count,
    COUNT(*) FILTER (WHERE {file_path_condition("h.path_text", "user_home")})::bigint AS user_home_path_count,
    COUNT(*) FILTER (WHERE {file_path_condition("h.path_text", "hidden")})::bigint AS hidden_path_count,
    COUNT(*) FILTER (WHERE {file_path_condition("h.path_text", "script")})::bigint AS script_path_count,
    COUNT(*) FILTER (WHERE {missing_text_sql("h.path_text")})::bigint AS missing_path_count,
    COUNT(DISTINCT h.process_uuid) FILTER (WHERE COALESCE(pns.total_network_events, 0) > 0)::bigint AS network_active_process_count,
    ROUND(AVG(COALESCE(pns.total_network_events, 0)::numeric), 6) AS avg_network_events_of_accessing_processes,
    MAX(COALESCE(pns.total_network_events, 0))::bigint AS max_network_events_of_accessing_processes,
    ROUND(
        (COUNT(DISTINCT h.process_uuid) FILTER (WHERE COALESCE(pns.total_network_events, 0) > 0))::numeric /
        NULLIF(COUNT(DISTINCT h.process_uuid), 0),
        6
    ) AS network_active_process_ratio
FROM file_hits AS h
JOIN file_entities AS f
  ON f.uuid = h.node_uuid
LEFT JOIN process_network_stats AS pns
  ON pns.process_uuid = h.process_uuid
GROUP BY h.node_uuid, f.host_id, f.file_type, f.size_bytes
"""


PROCESS_VIEW_NETWORK_NODE_QUERY = """
WITH network_hits AS (
    SELECT
        e.subject_uuid AS process_uuid,
        e.object_uuid AS node_uuid,
        e.event_type
    FROM events_raw AS e
    JOIN network_entities AS n
      ON n.uuid = e.object_uuid
    WHERE e.timestamp_ns >= %s
      AND e.timestamp_ns < %s
    UNION ALL
    SELECT
        e.subject_uuid AS process_uuid,
        e.object2_uuid AS node_uuid,
        e.event_type
    FROM events_raw AS e
    JOIN network_entities AS n
      ON n.uuid = e.object2_uuid
    WHERE e.timestamp_ns >= %s
      AND e.timestamp_ns < %s
),
process_file_stats AS (
    SELECT
        e.subject_uuid AS process_uuid,
        COUNT(*)::bigint AS total_file_events
    FROM events_raw AS e
    JOIN process_entities AS p
      ON p.uuid = e.subject_uuid
    JOIN file_entities AS f
      ON f.uuid = e.object_uuid OR f.uuid = e.object2_uuid
    WHERE e.timestamp_ns >= %s
      AND e.timestamp_ns < %s
    GROUP BY e.subject_uuid
)
SELECT
    h.node_uuid,
    'network'::text AS node_type,
    COALESCE(n.host_id, '') AS host_id,
    COALESCE(n.local_address, '') AS local_address,
    COALESCE(n.remote_address, '') AS remote_address,
    COALESCE(n.local_port, -1) AS local_port,
    COALESCE(n.remote_port, -1) AS remote_port,
    CASE
        WHEN n.local_port IS NULL THEN 'unknown'
        WHEN n.local_port < 0 THEN 'invalid'
        WHEN n.local_port <= 1023 THEN 'well_known'
        WHEN n.local_port <= 49151 THEN 'registered'
        ELSE 'ephemeral'
    END AS local_port_bucket,
    CASE
        WHEN n.remote_port IS NULL THEN 'unknown'
        WHEN n.remote_port < 0 THEN 'invalid'
        WHEN n.remote_port <= 1023 THEN 'well_known'
        WHEN n.remote_port <= 49151 THEN 'registered'
        ELSE 'ephemeral'
    END AS remote_port_bucket,
    CASE
        WHEN NULLIF(BTRIM(COALESCE(n.remote_address, '')), '') IS NULL THEN 'unknown'
        WHEN n.remote_address !~ '^[0-9]{1,3}(\\.[0-9]{1,3}){3}$' THEN 'unknown'
        WHEN inet(n.remote_address) << inet '10.0.0.0/8'
          OR inet(n.remote_address) << inet '172.16.0.0/12'
          OR inet(n.remote_address) << inet '192.168.0.0/16'
          OR inet(n.remote_address) << inet '127.0.0.0/8'
          OR inet(n.remote_address) << inet '169.254.0.0/16'
        THEN 'no'
        ELSE 'yes'
    END AS external_remote_ip_flag,
    COUNT(*)::bigint AS total_process_context_events,
    COUNT(DISTINCT h.process_uuid)::bigint AS unique_process_count,
    COUNT(DISTINCT h.event_type)::bigint AS event_type_diversity,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_CONNECT')::bigint AS connect_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_ACCEPT')::bigint AS accept_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_BIND')::bigint AS bind_count,
    COUNT(*) FILTER (WHERE h.event_type IN ('EVENT_SENDTO', 'EVENT_SENDMSG'))::bigint AS send_count,
    COUNT(*) FILTER (WHERE h.event_type IN ('EVENT_RECVFROM', 'EVENT_RECVMSG'))::bigint AS recv_count,
    COUNT(DISTINCT h.process_uuid) FILTER (WHERE COALESCE(pfs.total_file_events, 0) > 0)::bigint AS file_active_process_count,
    ROUND(AVG(COALESCE(pfs.total_file_events, 0)::numeric), 6) AS avg_file_events_of_using_processes,
    MAX(COALESCE(pfs.total_file_events, 0))::bigint AS max_file_events_of_using_processes,
    ROUND(
        (COUNT(DISTINCT h.process_uuid) FILTER (WHERE COALESCE(pfs.total_file_events, 0) > 0))::numeric /
        NULLIF(COUNT(DISTINCT h.process_uuid), 0),
        6
    ) AS file_active_process_ratio
FROM network_hits AS h
JOIN network_entities AS n
  ON n.uuid = h.node_uuid
LEFT JOIN process_file_stats AS pfs
  ON pfs.process_uuid = h.process_uuid
GROUP BY
    h.node_uuid,
    n.host_id,
    n.local_address,
    n.remote_address,
    n.local_port,
    n.remote_port
"""


FILE_VIEW_PROCESS_NODE_QUERY = f"""
WITH file_hits AS (
    SELECT
        e.subject_uuid AS node_uuid,
        e.object_uuid AS file_uuid,
        e.event_type,
        e.exec_name
    FROM events_raw AS e
    JOIN file_entities AS f
      ON f.uuid = e.object_uuid
    WHERE e.timestamp_ns >= %s
      AND e.timestamp_ns < %s
    UNION ALL
    SELECT
        e.subject_uuid AS node_uuid,
        e.object2_uuid AS file_uuid,
        e.event_type,
        e.exec_name
    FROM events_raw AS e
    JOIN file_entities AS f
      ON f.uuid = e.object2_uuid
    WHERE e.timestamp_ns >= %s
      AND e.timestamp_ns < %s
)
SELECT
    h.node_uuid,
    'process'::text AS node_type,
    COALESCE(p.host_id, '') AS host_id,
    COALESCE(p.subject_type, '') AS subject_type,
    CASE
        WHEN p.parent_subject_uuid IS NULL OR p.parent_subject_uuid = '' THEN 0
        ELSE 1
    END AS has_parent_flag,
    COUNT(*)::bigint AS total_file_events,
    COUNT(DISTINCT h.file_uuid)::bigint AS unique_file_count,
    COUNT(DISTINCT h.event_type)::bigint AS event_type_diversity,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_READ')::bigint AS read_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_WRITE')::bigint AS write_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_OPEN')::bigint AS open_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_EXECUTE')::bigint AS execute_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_CLOSE')::bigint AS close_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_CREATE_OBJECT')::bigint AS create_object_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_UNLINK')::bigint AS unlink_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_RENAME')::bigint AS rename_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_MODIFY_FILE_ATTRIBUTES')::bigint AS modify_file_attr_count,
    COUNT(DISTINCT NULLIF(BTRIM(COALESCE(h.exec_name, '')), ''))::bigint AS unique_exec_name_count,
    COUNT(*) FILTER (WHERE {exec_name_condition("h.exec_name", "shell")})::bigint AS shell_exec_count,
    COUNT(*) FILTER (WHERE {exec_name_condition("h.exec_name", "interpreter")})::bigint AS interpreter_exec_count,
    COUNT(*) FILTER (WHERE {exec_name_condition("h.exec_name", "network_tool")})::bigint AS network_tool_exec_count,
    COUNT(*) FILTER (WHERE {exec_name_condition("h.exec_name", "package_tool")})::bigint AS package_tool_exec_count,
    COUNT(*) FILTER (WHERE {exec_name_condition("h.exec_name", "system_tool")})::bigint AS system_tool_exec_count,
    COUNT(*) FILTER (WHERE {missing_text_sql("h.exec_name")})::bigint AS missing_exec_name_count,
    COUNT(DISTINCT h.file_uuid) FILTER (WHERE h.event_type = 'EVENT_READ')::bigint AS unique_read_file_count,
    COUNT(DISTINCT h.file_uuid) FILTER (WHERE h.event_type = 'EVENT_WRITE')::bigint AS unique_write_file_count,
    ROUND((COUNT(*) FILTER (WHERE h.event_type = 'EVENT_READ'))::numeric / NULLIF(COUNT(*), 0), 6) AS read_ratio,
    ROUND((COUNT(*) FILTER (WHERE h.event_type = 'EVENT_WRITE'))::numeric / NULLIF(COUNT(*), 0), 6) AS write_ratio,
    ROUND((COUNT(*) FILTER (WHERE h.event_type = 'EVENT_EXECUTE'))::numeric / NULLIF(COUNT(*), 0), 6) AS execute_ratio
FROM file_hits AS h
JOIN process_entities AS p
  ON p.uuid = h.node_uuid
GROUP BY h.node_uuid, p.host_id, p.subject_type, p.parent_subject_uuid
"""


NETWORK_VIEW_PROCESS_NODE_QUERY = f"""
WITH network_hits AS (
    SELECT
        e.subject_uuid AS node_uuid,
        e.object_uuid AS network_uuid,
        e.event_type,
        e.exec_name
    FROM events_raw AS e
    JOIN network_entities AS n
      ON n.uuid = e.object_uuid
    WHERE e.timestamp_ns >= %s
      AND e.timestamp_ns < %s
    UNION ALL
    SELECT
        e.subject_uuid AS node_uuid,
        e.object2_uuid AS network_uuid,
        e.event_type,
        e.exec_name
    FROM events_raw AS e
    JOIN network_entities AS n
      ON n.uuid = e.object2_uuid
    WHERE e.timestamp_ns >= %s
      AND e.timestamp_ns < %s
),
network_hit_enriched AS (
    SELECT
    h.node_uuid,
    h.network_uuid,
    h.event_type,
    h.exec_name,
    n.remote_address,
    n.remote_port,
    CASE
            WHEN NULLIF(BTRIM(COALESCE(n.remote_address, '')), '') IS NULL THEN 'unknown'
            WHEN n.remote_address !~ '^[0-9]{1,3}(\\.[0-9]{1,3}){3}$' THEN 'unknown'
            WHEN inet(n.remote_address) << inet '10.0.0.0/8'
              OR inet(n.remote_address) << inet '172.16.0.0/12'
              OR inet(n.remote_address) << inet '192.168.0.0/16'
              OR inet(n.remote_address) << inet '127.0.0.0/8'
              OR inet(n.remote_address) << inet '169.254.0.0/16'
            THEN 'no'
            ELSE 'yes'
        END AS external_remote_ip_flag
    FROM network_hits AS h
    JOIN network_entities AS n
      ON n.uuid = h.network_uuid
)
SELECT
    h.node_uuid,
    'process'::text AS node_type,
    COALESCE(p.host_id, '') AS host_id,
    COALESCE(p.subject_type, '') AS subject_type,
    CASE
        WHEN p.parent_subject_uuid IS NULL OR p.parent_subject_uuid = '' THEN 0
        ELSE 1
    END AS has_parent_flag,
    COUNT(*)::bigint AS total_network_events,
    COUNT(DISTINCT h.network_uuid)::bigint AS unique_network_count,
    COUNT(DISTINCT h.event_type)::bigint AS event_type_diversity,
    COUNT(DISTINCT NULLIF(BTRIM(COALESCE(h.remote_address, '')), ''))::bigint AS unique_remote_ip_count,
    COUNT(DISTINCT h.remote_port) FILTER (WHERE h.remote_port IS NOT NULL AND h.remote_port >= 0)::bigint AS unique_remote_port_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_CONNECT')::bigint AS connect_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_ACCEPT')::bigint AS accept_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_BIND')::bigint AS bind_count,
    COUNT(*) FILTER (WHERE h.event_type IN ('EVENT_SENDTO', 'EVENT_SENDMSG'))::bigint AS send_count,
    COUNT(*) FILTER (WHERE h.event_type IN ('EVENT_RECVFROM', 'EVENT_RECVMSG'))::bigint AS recv_count,
    COUNT(*) FILTER (WHERE h.event_type = 'EVENT_CLOSE')::bigint AS close_count,
    COUNT(DISTINCT NULLIF(BTRIM(COALESCE(h.exec_name, '')), ''))::bigint AS unique_exec_name_count,
    COUNT(*) FILTER (WHERE {exec_name_condition("h.exec_name", "shell")})::bigint AS shell_exec_count,
    COUNT(*) FILTER (WHERE {exec_name_condition("h.exec_name", "interpreter")})::bigint AS interpreter_exec_count,
    COUNT(*) FILTER (WHERE {exec_name_condition("h.exec_name", "network_tool")})::bigint AS network_tool_exec_count,
    COUNT(*) FILTER (WHERE {exec_name_condition("h.exec_name", "package_tool")})::bigint AS package_tool_exec_count,
    COUNT(*) FILTER (WHERE {exec_name_condition("h.exec_name", "system_tool")})::bigint AS system_tool_exec_count,
    COUNT(*) FILTER (WHERE {missing_text_sql("h.exec_name")})::bigint AS missing_exec_name_count,
    COUNT(DISTINCT h.network_uuid) FILTER (WHERE h.external_remote_ip_flag = 'yes')::bigint AS external_network_count,
    ROUND(
        (COUNT(DISTINCT h.network_uuid) FILTER (WHERE h.external_remote_ip_flag = 'yes'))::numeric /
        NULLIF(COUNT(DISTINCT h.network_uuid), 0),
        6
    ) AS external_network_ratio,
    COUNT(DISTINCT h.network_uuid) FILTER (
        WHERE h.remote_port IN (21, 22, 23, 25, 53, 80, 110, 123, 135, 139, 143, 389, 443, 445, 3389, 4444, 8080)
    )::bigint AS high_risk_port_contact_count
FROM network_hit_enriched AS h
JOIN process_entities AS p
  ON p.uuid = h.node_uuid
GROUP BY h.node_uuid, p.host_id, p.subject_type, p.parent_subject_uuid
"""


def fetch_event_count(conn, start_ns: int, end_ns: int) -> int:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT COUNT(*)::bigint
            FROM events_raw
            WHERE timestamp_ns >= %s
              AND timestamp_ns < %s
            """,
            (
                start_ns,
                end_ns,
            ),
        )
        return int(cursor.fetchone()[0])


def create_process_temp_tables(conn) -> None:
    with conn.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS tmp_process_stats")
        cursor.execute("DROP TABLE IF EXISTS tmp_process_event_types")
        cursor.execute("DROP TABLE IF EXISTS tmp_process_file_refs")
        cursor.execute("DROP TABLE IF EXISTS tmp_process_network_refs")
        cursor.execute("DROP TABLE IF EXISTS tmp_process_exec_names")

        cursor.execute(
            """
            CREATE TEMP TABLE tmp_process_stats (
                node_uuid VARCHAR(255) PRIMARY KEY,
                total_events BIGINT NOT NULL DEFAULT 0,
                read_count BIGINT NOT NULL DEFAULT 0,
                write_count BIGINT NOT NULL DEFAULT 0,
                open_count BIGINT NOT NULL DEFAULT 0,
                execute_count BIGINT NOT NULL DEFAULT 0,
                connect_count BIGINT NOT NULL DEFAULT 0,
                send_count BIGINT NOT NULL DEFAULT 0,
                recv_count BIGINT NOT NULL DEFAULT 0,
                accept_count BIGINT NOT NULL DEFAULT 0,
                create_object_count BIGINT NOT NULL DEFAULT 0,
                fork_count BIGINT NOT NULL DEFAULT 0,
                mmap_count BIGINT NOT NULL DEFAULT 0,
                modify_process_count BIGINT NOT NULL DEFAULT 0,
                close_count BIGINT NOT NULL DEFAULT 0,
                shell_exec_count BIGINT NOT NULL DEFAULT 0,
                interpreter_exec_count BIGINT NOT NULL DEFAULT 0,
                network_tool_exec_count BIGINT NOT NULL DEFAULT 0,
                package_tool_exec_count BIGINT NOT NULL DEFAULT 0,
                system_tool_exec_count BIGINT NOT NULL DEFAULT 0,
                missing_exec_name_count BIGINT NOT NULL DEFAULT 0
            )
            """
        )
        cursor.execute(
            """
            CREATE TEMP TABLE tmp_process_event_types (
                node_uuid VARCHAR(255) NOT NULL,
                event_type VARCHAR(255) NOT NULL,
                PRIMARY KEY (node_uuid, event_type)
            )
            """
        )
        cursor.execute(
            """
            CREATE TEMP TABLE tmp_process_file_refs (
                node_uuid VARCHAR(255) NOT NULL,
                file_uuid VARCHAR(255) NOT NULL,
                PRIMARY KEY (node_uuid, file_uuid)
            )
            """
        )
        cursor.execute(
            """
            CREATE TEMP TABLE tmp_process_network_refs (
                node_uuid VARCHAR(255) NOT NULL,
                network_uuid VARCHAR(255) NOT NULL,
                PRIMARY KEY (node_uuid, network_uuid)
            )
            """
        )
        cursor.execute(
            """
            CREATE TEMP TABLE tmp_process_exec_names (
                node_uuid VARCHAR(255) NOT NULL,
                exec_name VARCHAR(512) NOT NULL,
                PRIMARY KEY (node_uuid, exec_name)
            )
            """
        )
    conn.commit()


def populate_process_temp_tables(conn, start_ns: int, end_ns: int, label: str) -> None:
    steps = [
        (
            "stats",
            f"""
            INSERT INTO tmp_process_stats (
                node_uuid,
                total_events,
                read_count,
                write_count,
                open_count,
                execute_count,
                connect_count,
                send_count,
                recv_count,
                accept_count,
                create_object_count,
                fork_count,
                mmap_count,
                modify_process_count,
                close_count,
                shell_exec_count,
                interpreter_exec_count,
                network_tool_exec_count,
                package_tool_exec_count,
                system_tool_exec_count,
                missing_exec_name_count
            )
            SELECT
                e.subject_uuid AS node_uuid,
                COUNT(*)::bigint AS total_events,
                COUNT(*) FILTER (WHERE e.event_type = 'EVENT_READ')::bigint AS read_count,
                COUNT(*) FILTER (WHERE e.event_type = 'EVENT_WRITE')::bigint AS write_count,
                COUNT(*) FILTER (WHERE e.event_type = 'EVENT_OPEN')::bigint AS open_count,
                COUNT(*) FILTER (WHERE e.event_type = 'EVENT_EXECUTE')::bigint AS execute_count,
                COUNT(*) FILTER (WHERE e.event_type = 'EVENT_CONNECT')::bigint AS connect_count,
                COUNT(*) FILTER (WHERE e.event_type IN ('EVENT_SENDTO', 'EVENT_SENDMSG'))::bigint AS send_count,
                COUNT(*) FILTER (WHERE e.event_type IN ('EVENT_RECVFROM', 'EVENT_RECVMSG'))::bigint AS recv_count,
                COUNT(*) FILTER (WHERE e.event_type = 'EVENT_ACCEPT')::bigint AS accept_count,
                COUNT(*) FILTER (WHERE e.event_type = 'EVENT_CREATE_OBJECT')::bigint AS create_object_count,
                COUNT(*) FILTER (WHERE e.event_type = 'EVENT_FORK')::bigint AS fork_count,
                COUNT(*) FILTER (WHERE e.event_type = 'EVENT_MMAP')::bigint AS mmap_count,
                COUNT(*) FILTER (WHERE e.event_type = 'EVENT_MODIFY_PROCESS')::bigint AS modify_process_count,
                COUNT(*) FILTER (WHERE e.event_type = 'EVENT_CLOSE')::bigint AS close_count,
                COUNT(*) FILTER (WHERE {exec_name_condition("e.exec_name", "shell")})::bigint AS shell_exec_count,
                COUNT(*) FILTER (WHERE {exec_name_condition("e.exec_name", "interpreter")})::bigint AS interpreter_exec_count,
                COUNT(*) FILTER (WHERE {exec_name_condition("e.exec_name", "network_tool")})::bigint AS network_tool_exec_count,
                COUNT(*) FILTER (WHERE {exec_name_condition("e.exec_name", "package_tool")})::bigint AS package_tool_exec_count,
                COUNT(*) FILTER (WHERE {exec_name_condition("e.exec_name", "system_tool")})::bigint AS system_tool_exec_count,
                COUNT(*) FILTER (WHERE {missing_text_sql("e.exec_name")})::bigint AS missing_exec_name_count
            FROM events_raw AS e
            JOIN process_entities AS p
              ON p.uuid = e.subject_uuid
            WHERE e.timestamp_ns >= %s
              AND e.timestamp_ns < %s
            GROUP BY e.subject_uuid
            ON CONFLICT (node_uuid) DO NOTHING
            """,
        ),
        (
            "event types",
            """
            INSERT INTO tmp_process_event_types (node_uuid, event_type)
            SELECT DISTINCT e.subject_uuid, e.event_type
            FROM events_raw AS e
            JOIN process_entities AS p
              ON p.uuid = e.subject_uuid
            WHERE e.timestamp_ns >= %s
              AND e.timestamp_ns < %s
            ON CONFLICT DO NOTHING
            """,
        ),
        (
            "file refs(object)",
            """
            INSERT INTO tmp_process_file_refs (node_uuid, file_uuid)
            SELECT DISTINCT e.subject_uuid, e.object_uuid
            FROM events_raw AS e
            JOIN process_entities AS p
              ON p.uuid = e.subject_uuid
            JOIN file_entities AS f
              ON f.uuid = e.object_uuid
            WHERE e.timestamp_ns >= %s
              AND e.timestamp_ns < %s
            ON CONFLICT DO NOTHING
            """,
        ),
        (
            "file refs(object2)",
            """
            INSERT INTO tmp_process_file_refs (node_uuid, file_uuid)
            SELECT DISTINCT e.subject_uuid, e.object2_uuid
            FROM events_raw AS e
            JOIN process_entities AS p
              ON p.uuid = e.subject_uuid
            JOIN file_entities AS f
              ON f.uuid = e.object2_uuid
            WHERE e.timestamp_ns >= %s
              AND e.timestamp_ns < %s
            ON CONFLICT DO NOTHING
            """,
        ),
        (
            "network refs(object)",
            """
            INSERT INTO tmp_process_network_refs (node_uuid, network_uuid)
            SELECT DISTINCT e.subject_uuid, e.object_uuid
            FROM events_raw AS e
            JOIN process_entities AS p
              ON p.uuid = e.subject_uuid
            JOIN network_entities AS n
              ON n.uuid = e.object_uuid
            WHERE e.timestamp_ns >= %s
              AND e.timestamp_ns < %s
            ON CONFLICT DO NOTHING
            """,
        ),
        (
            "network refs(object2)",
            """
            INSERT INTO tmp_process_network_refs (node_uuid, network_uuid)
            SELECT DISTINCT e.subject_uuid, e.object2_uuid
            FROM events_raw AS e
            JOIN process_entities AS p
              ON p.uuid = e.subject_uuid
            JOIN network_entities AS n
              ON n.uuid = e.object2_uuid
            WHERE e.timestamp_ns >= %s
              AND e.timestamp_ns < %s
            ON CONFLICT DO NOTHING
            """,
        ),
        (
            "exec names",
            """
            INSERT INTO tmp_process_exec_names (node_uuid, exec_name)
            SELECT DISTINCT
                e.subject_uuid,
                LOWER(BTRIM(e.exec_name)) AS exec_name
            FROM events_raw AS e
            JOIN process_entities AS p
              ON p.uuid = e.subject_uuid
            WHERE e.timestamp_ns >= %s
              AND e.timestamp_ns < %s
              AND NULLIF(BTRIM(COALESCE(e.exec_name, '')), '') IS NOT NULL
            ON CONFLICT DO NOTHING
            """,
        ),
    ]

    with tqdm(total=len(steps), desc=f"{label} phases", unit="step", dynamic_ncols=True, leave=False) as phase_bar:
        with conn.cursor() as cursor:
            for step_name, sql in steps:
                phase_bar.set_postfix_str(step_name)
                cursor.execute(sql, (start_ns, end_ns))
                phase_bar.update(1)
        conn.commit()


def export_process_view_incremental(conn, days: List[str], output_path: Path, label: str) -> int:
    create_process_temp_tables(conn)
    start_ns, end_ns = window_bounds(days)
    log(f"[feature-extract] {label}: aggregate full window in one pass")
    populate_process_temp_tables(conn, start_ns, end_ns, label)
    return copy_query_to_tsv(conn, PROCESS_VIEW_EXPORT_FROM_TEMP_QUERY, tuple(), output_path, label)


def export_window_features(conn, window: Dict[str, object], output_dir: Path) -> Dict[str, object]:
    days = list(window["days"])
    start_ns, end_ns = window_bounds(days)
    window_dir = output_dir / str(window["name"])
    ensure_output_dir(window_dir)

    log(f"[feature-extract] {window['name']}: preparing window {days[0]} -> {days[-1]}")
    event_count = fetch_event_count(conn, start_ns, end_ns)
    log(f"[feature-extract] {window['name']}: event_count={event_count:,}")

    process_rows = export_process_view_incremental(
        conn,
        days,
        window_dir / PROCESS_VIEW_FILE,
        label=f"{window['name']} process_view__process_node",
    )

    process_view_file_rows = copy_query_to_tsv(
        conn,
        PROCESS_VIEW_FILE_NODE_QUERY,
        (start_ns, end_ns, start_ns, end_ns, start_ns, end_ns),
        window_dir / PROCESS_VIEW_FILE_NODE_FILE,
        log_label=f"{window['name']} process_view__file_node",
    )

    process_view_network_rows = copy_query_to_tsv(
        conn,
        PROCESS_VIEW_NETWORK_NODE_QUERY,
        (start_ns, end_ns, start_ns, end_ns, start_ns, end_ns),
        window_dir / PROCESS_VIEW_NETWORK_NODE_FILE,
        log_label=f"{window['name']} process_view__network_node",
    )

    file_rows = copy_query_to_tsv(
        conn,
        FILE_VIEW_QUERY,
        (start_ns, end_ns, start_ns, end_ns),
        window_dir / FILE_VIEW_FILE,
        log_label=f"{window['name']} file_view__file_node",
    )

    file_view_process_rows = copy_query_to_tsv(
        conn,
        FILE_VIEW_PROCESS_NODE_QUERY,
        (start_ns, end_ns, start_ns, end_ns),
        window_dir / FILE_VIEW_PROCESS_NODE_FILE,
        log_label=f"{window['name']} file_view__process_node",
    )

    network_rows = copy_query_to_tsv(
        conn,
        NETWORK_VIEW_QUERY,
        (start_ns, end_ns, start_ns, end_ns),
        window_dir / NETWORK_VIEW_FILE,
        log_label=f"{window['name']} network_view__network_node",
    )

    network_view_process_rows = copy_query_to_tsv(
        conn,
        NETWORK_VIEW_PROCESS_NODE_QUERY,
        (start_ns, end_ns, start_ns, end_ns),
        window_dir / NETWORK_VIEW_PROCESS_NODE_FILE,
        log_label=f"{window['name']} network_view__process_node",
    )

    metadata = {
        "window_name": window["name"],
        "split": window["split"],
        "days": days,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "start_time_utc": ns_to_text(start_ns),
        "end_time_utc_exclusive": ns_to_text(end_ns),
        "event_count": event_count,
        "process_node_count": process_rows,
        "file_node_count": file_rows,
        "network_node_count": network_rows,
        "process_view_rows": process_rows,
        "process_view__file_node_rows": process_view_file_rows,
        "process_view__network_node_rows": process_view_network_rows,
        "file_view_rows": file_rows,
        "file_view__process_node_rows": file_view_process_rows,
        "network_view_rows": network_rows,
        "network_view__process_node_rows": network_view_process_rows,
        "feature_files": {
            "process_view__process_node": PROCESS_VIEW_FILE,
            "process_view__file_node": PROCESS_VIEW_FILE_NODE_FILE,
            "process_view__network_node": PROCESS_VIEW_NETWORK_NODE_FILE,
            "file_view__file_node": FILE_VIEW_FILE,
            "file_view__process_node": FILE_VIEW_PROCESS_NODE_FILE,
            "network_view__network_node": NETWORK_VIEW_FILE,
            "network_view__process_node": NETWORK_VIEW_PROCESS_NODE_FILE,
        },
    }
    write_json(window_dir / "metadata.json", metadata)
    log(
        f"[feature-extract] {window['name']}: done "
        f"process_rows={process_rows:,} "
        f"process_view_file_rows={process_view_file_rows:,} "
        f"process_view_network_rows={process_view_network_rows:,} "
        f"file_rows={file_rows:,} "
        f"file_view_process_rows={file_view_process_rows:,} "
        f"network_rows={network_rows:,} "
        f"network_view_process_rows={network_view_process_rows:,}"
    )
    return metadata


def print_summary(window_metadata: Iterable[Dict[str, object]]) -> None:
    print("[feature-extract] window summary")
    for item in window_metadata:
        event_count = item.get("event_count", "n/a")
        process_nodes = item.get("process_node_count", "n/a")
        file_nodes = item.get("file_node_count", "n/a")
        network_nodes = item.get("network_node_count", "n/a")
        print(
            "  "
            f"{item['window_name']}: days={','.join(item['days'])} "
            f"events={event_count} "
            f"process_nodes={process_nodes} "
            f"file_nodes={file_nodes} "
            f"network_nodes={network_nodes}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = build_common_parser("Extract split-aware baseline CADETS node features.")
    parser.add_argument(
        "--split-manifest",
        type=str,
        default=str(DEFAULT_SPLIT_MANIFEST_PATH),
        help=f"Path to split_manifest.json. Default: {DEFAULT_SPLIT_MANIFEST_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for extracted features. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print split/window counts without exporting TSV feature files.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_ingest_config(args)
    split_manifest_path = Path(args.split_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_dir(output_dir)

    split_manifest = load_split_manifest(split_manifest_path)
    windows = build_feature_windows(split_manifest)

    if args.summary_only:
        window_metadata: List[Dict[str, object]] = []
        for window in windows:
            window_metadata.append(
                {
                    "window_name": window["name"],
                    "split": window["split"],
                    "days": list(window["days"]),
                }
            )
        print_summary(window_metadata)
        return

    conn = connect_target_db(config.database)
    try:
        tune_feature_session(conn)
        window_metadata = []
        with tqdm(total=len(windows), desc="feature windows", unit="window", dynamic_ncols=True) as window_bar:
            for window in windows:
                metadata = export_window_features(conn, window, output_dir)
                window_metadata.append(metadata)
                window_bar.update(1)

        print_summary(window_metadata)
        write_json(
            output_dir / "feature_manifest.json",
            {
                "split_manifest_path": str(split_manifest_path),
                "windows": window_metadata,
                "notes": [
                    "This stage now exports seven feature groups: three self-view groups plus four process-centered cross-view groups.",
                    "Unmatched GT UUIDs that look like pipe-style IPC objects are not forced into file/network tables.",
                ],
            },
        )
        print(f"[feature-extract] outputs written to: {output_dir}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
