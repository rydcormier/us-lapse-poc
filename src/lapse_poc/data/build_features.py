from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from lapse_poc.settings import CENSOR_DATE


def _duckdb_read_expr(path: str) -> str:
    p = Path(path)
    # Use forward slashes for DuckDB on Windows too
    s = str(p.resolve()).replace("\\", "/")
    if p.suffix.lower() == ".parquet":
        return f"read_parquet('{s}')"
    return f"read_csv_auto('{s}', header=True)"


def build_policy_quarter_panel(
    raw_path: str,
    out_path: str,
    horizon_qtrs: int = 1,
    include_djia: bool = False,
    censor_date: str = CENSOR_DATE,
    train_end: str = "2006-12-31",
    val_end: str = "2007-12-31",
) -> dict:
    """
    Build a discrete-time hazard / early-warning table.

    Each row is a (policy_id, tenure_qtr) snapshot predicting lapse event within next `horizon_qtrs`.
    For terminated policies, duration is used; for censored, duration is inferred up to censor_date.

    Robustness: if duration values appear scaled (e.g., 0.01), we multiply by 100.

    Args:
        raw_path:       Path to raw uslapseagent data (csv or parquet).
        out_path:       Output path for the panel data (csv or parquet).
        horizon_qtrs:   Prediction horizon in quarters.
        include_djia:   Whether to include DJIA feature.
        censor_date:    Date to censor incomplete policies (YYYY-MM-DD).
        train_end:      End date for training split (YYYY-MM-DD).
        val_end:        End date for validation split (YYYY-MM-DD).
    Returns:
        Manifest dictionary with metadata about the build.
    """
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")
    src = _duckdb_read_expr(raw_path)

    # Load + rename into a clean raw table (snake_case)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE raw_uslapseagent AS
        SELECT
            COALESCE(CAST(policy_id AS BIGINT), row_number() OVER ()) AS policy_id,
            CAST("issue.date" AS DATE) AS issue_date,
            CAST(duration AS DOUBLE) AS duration_raw,
            CAST("acc.death.rider" AS VARCHAR) AS acc_death_rider,
            CAST(gender AS VARCHAR) AS gender,
            CAST("premium.frequency" AS VARCHAR) AS premium_frequency,
            CAST("risk.state" AS VARCHAR) AS risk_state,
            CAST("underwriting.age" AS VARCHAR) AS underwriting_age,
            CAST("living.place" AS VARCHAR) AS living_place,
            CAST("annual.premium" AS DOUBLE) AS annual_premium,
            CAST(DJIA AS DOUBLE) AS djia,
            CAST("termination.cause" AS VARCHAR) AS termination_cause,
            CAST(surrender AS INTEGER) AS surrender,
            CAST(death AS INTEGER) AS death,
            CAST(other AS INTEGER) AS other,
            CAST(allcause AS INTEGER) AS allcause
        FROM {src};
        """
    )

    # Heuristic: detect whether duration is scaled (e.g., 0.01 => 1 quarter)
    result = con.execute(
        "SELECT quantile_cont(duration_raw, 0.99) FROM raw_uslapseagent WHERE duration_raw IS NOT NULL"
    ).fetchone()
    p99 = result[0] if result is not None else None
    scale = 100 if (p99 is not None and p99 < 5) else 1

    # Base table with duration in integer quarters; infer duration for censored policies.
    con.execute(
        f"""
        CREATE OR REPLACE TABLE base AS
        SELECT
            policy_id,
            issue_date,
            CASE
                WHEN duration_raw IS NULL THEN NULL
                ELSE CAST(ROUND(duration_raw * {scale}) AS INTEGER)
            END AS duration_qtr,
            acc_death_rider,
            gender,
            premium_frequency,
            risk_state,
            underwriting_age,
            living_place,
            annual_premium,
            djia,
            termination_cause,
            surrender, death, other, allcause,
            COALESCE(
                CASE
                    WHEN duration_raw IS NULL THEN NULL
                    ELSE CAST(ROUND(duration_raw * {scale}) AS INTEGER)
                END,
                date_diff('quarter', issue_date, DATE '{censor_date}') + 1
            ) AS duration_obs_qtr
        FROM raw_uslapseagent;
        """
    )

    djia_select = ", djia" if include_djia else ""

    # Panel: quarters 1..(duration_obs_qtr - horizon)
    # Labels indicate an event occurs in [q+1, q+horizon].
    con.execute(
        f"""
        CREATE OR REPLACE TABLE policy_quarter_panel AS
        SELECT
            b.policy_id,
            b.issue_date,
            gs.qtr AS tenure_qtr,
            date_add(b.issue_date, INTERVAL (gs.qtr - 1) QUARTER) AS as_of_date,

            -- Features (static repeated + time index)
            b.gender,
            b.premium_frequency,
            b.risk_state,
            b.underwriting_age,
            b.living_place,
            b.acc_death_rider,
            b.annual_premium
            {djia_select},

            year(date_add(b.issue_date, INTERVAL (gs.qtr - 1) QUARTER)) AS as_of_year,
            quarter(date_add(b.issue_date, INTERVAL (gs.qtr - 1) QUARTER)) AS as_of_quarter,

            CASE
                WHEN date_add(b.issue_date, INTERVAL (gs.qtr - 1) QUARTER) <= DATE '{train_end}' THEN 'train'
                WHEN date_add(b.issue_date, INTERVAL (gs.qtr - 1) QUARTER) <= DATE '{val_end}' THEN 'val'
                ELSE 'test'
            END AS split,

            -- Labels
            CASE
                WHEN b.allcause = 1 AND b.surrender = 1
                AND b.duration_qtr BETWEEN (gs.qtr + 1) AND (gs.qtr + {horizon_qtrs})
                THEN 1 ELSE 0
            END AS y_surrender_next,

            CASE
                WHEN b.allcause = 1 AND b.death = 1
                AND b.duration_qtr BETWEEN (gs.qtr + 1) AND (gs.qtr + {horizon_qtrs})
                THEN 1 ELSE 0
            END AS y_death_next,

            CASE
                WHEN b.allcause = 1 AND b.other = 1
                AND b.duration_qtr BETWEEN (gs.qtr + 1) AND (gs.qtr + {horizon_qtrs})
                THEN 1 ELSE 0
            END AS y_other_next,

            CASE
                WHEN b.allcause = 1
                AND b.duration_qtr BETWEEN (gs.qtr + 1) AND (gs.qtr + {horizon_qtrs})
                THEN 1 ELSE 0
            END AS y_any_next

        FROM base b
        CROSS JOIN generate_series(1, GREATEST(0, b.duration_obs_qtr - {horizon_qtrs})) AS gs(qtr)
        WHERE b.duration_obs_qtr - {horizon_qtrs} >= 1;
        """
    )

    # Export
    out_s = str(out_p.resolve()).replace("\\", "/")
    if out_p.suffix.lower() == ".parquet":
        con.execute(f"COPY policy_quarter_panel TO '{out_s}' (FORMAT PARQUET);")
    else:
        con.execute(f"COPY policy_quarter_panel TO '{out_s}' (HEADER, DELIMITER ',');")

    # Basic manifest
    counts = con.execute(
        "SELECT split, COUNT(*) AS n, AVG(y_surrender_next)::DOUBLE AS rate FROM policy_quarter_panel GROUP BY 1"
    ).fetchall()

    manifest = {
        "raw_path": raw_path,
        "out_path": out_path,
        "horizon_qtrs": horizon_qtrs,
        "include_djia": include_djia,
        "censor_date": censor_date,
        "duration_scale": scale,
        "split_counts": [
            {"split": s, "n": int(n), "surrender_rate": float(r)} for (s, n, r) in counts
        ],
    }

    (out_p.parent / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Path to exported uslapseagent csv/parquet")
    ap.add_argument("--out", required=True, help="Output parquet/csv for modeling table")
    ap.add_argument("--horizon", type=int, default=1, help="Prediction horizon in quarters")
    ap.add_argument(
        "--include-djia", action="store_true", help="Include DJIA feature (off by default)"
    )
    args = ap.parse_args()

    manifest = build_policy_quarter_panel(
        raw_path=args.raw,
        out_path=args.out,
        horizon_qtrs=args.horizon,
        include_djia=args.include_djia,
    )
    print("Feature build complete. Manifest:")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
