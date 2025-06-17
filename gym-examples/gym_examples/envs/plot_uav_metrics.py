#!/usr/bin/env python3
"""
Script to load UAV pre- and post-flight metrics from a JSON file,
and plot relationships between pre-flight metrics (x-axis) and post-flight metrics (y-axis).
"""

import json
import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt

def load_metrics(json_path):
    """
    Load the JSON and return:
      - df: a pandas DataFrame indexed by UAV id, with columns for all metrics
      - meta: the top‐level JSON dict (for aggregated fields like total_uavs)
    """
    with open(json_path, 'r') as f:
        meta = json.load(f)

    pre = meta['uav_pre_fligt_info']
    post = meta['uav_post_flight_info']

    # Build DataFrames from the nested dicts
    df_pre  = pd.DataFrame.from_dict(pre, orient='index')
    df_post = pd.DataFrame.from_dict(post, orient='index')

    # Join them into one DataFrame
    df = df_pre.join(df_post)

    return df, meta

def plot_relationships(df, output_dir=None):
    """
    For each numeric pre-flight vs post-flight pair, make a scatter plot.
    Ignores any columns in pre-flight that are all zero or not used,
    and skips post-flight columns that are constant across all UAVs.
    """
    # define which pre-flight metrics to consider
    pre_cols = ['distance', 'time_avg']
    # define which post-flight metrics to consider
    post_cols = ['distance_factor']  # time_factor isn't collected; aggregated metrics are constant

    for x in pre_cols:
        if x not in df.columns:
            continue
        for y in post_cols:
            if y not in df.columns:
                continue

            # skip if y is constant (e.g. aggregated counts)
            if df[y].nunique() <= 1:
                continue

            plt.figure()
            plt.scatter(df[x], df[y])
            plt.xlabel(x.replace('_', ' ').title())
            plt.ylabel(y.replace('_', ' ').title())
            plt.title(f'{y} vs. {x}')
            plt.tight_layout()

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, f'{y}_vs_{x}.png')
                plt.savefig(out_path)
                print(f"Saved plot: {out_path}")
            else:
                plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Plot UAV pre-flight vs post-flight metrics from a JSON file"
    )
    parser.add_argument(
        'json_file',
        help="Path to the JSON metrics file (with uav_pre_fligt_info, uav_post_flight_info, total_uavs)"
    )
    parser.add_argument(
        '--outdir', '-o',
        help="If set, save plots into this directory (will be created if needed). Otherwise, show interactively.",
        default=None
    )
    args = parser.parse_args()

    df, meta = load_metrics(args.json_file)

    print(f"Loaded metrics for {meta.get('total_uavs', len(df))} UAVs from {args.json_file}")
    print("Columns available:\n ", ", ".join(df.columns))

    plot_relationships(df, output_dir=args.outdir)

    # print aggregated counts once
    nmac = df['NMAC_count'].iat[0] if 'NMAC_count' in df.columns else None
    ra   = df['RA_violation_count'].iat[0] if 'RA_violation_count' in df.columns else None
    if nmac is not None or ra is not None:
        print("\nAggregated counts (same for all UAVs):")
        if nmac is not None:
            print(f"  NMAC_count: {nmac}")
        if ra is not None:
            print(f"  RA_violation_count: {ra}")



if __name__ == '__main__':
    main()



