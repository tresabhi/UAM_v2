#!/usr/bin/env python3
"""
Script to plot the average post-flight distance_factor
against the number of UAVs, given one or more JSON metric files.
"""

import os
import json
import argparse

import pandas as pd
import matplotlib.pyplot as plt

def load_one_metrics(json_path):
    """
    Load a single JSON file and return:
      - total_uavs: from top‐level 'total_uavs'
      - avg_distance_factor: mean of post-flight 'distance_factor' across UAVs
    """
    with open(json_path, 'r') as f:
        meta = json.load(f)

    total_uavs = meta.get('total_uavs',
                         len(meta.get('uav_pre_fligt_info', {})))
    post = meta['uav_post_flight_info']
    df_post = pd.DataFrame.from_dict(post, orient='index')

    # compute average distance_factor
    avg_df = df_post['distance_factor'].mean()
    return total_uavs, avg_df

def aggregate_metrics(json_paths):
    """
    Given a list of JSON file paths, returns a DataFrame with columns:
      - total_uavs
      - avg_distance_factor
    """
    records = []
    for path in json_paths:
        try:
            n_uavs, avg_df = load_one_metrics(path)
            records.append({
                'total_uavs': n_uavs,
                'avg_distance_factor': avg_df,
                'source_file': os.path.basename(path)
            })
        except Exception as e:
            print(f"Warning: failed to load {path}: {e}")

    return pd.DataFrame.from_records(records)

def plot_avg_distance_vs_num_uavs(df, output_path=None):
    """
    Create a plot of avg_distance_factor (Y) vs total_uavs (X).
    """
    df = df.sort_values('total_uavs')
    plt.figure()
    plt.plot(df['total_uavs'], df['avg_distance_factor'], marker='o')
    plt.xlabel('Number of UAVs')
    plt.ylabel('Average Distance Factor')
    plt.title('Avg Distance Factor vs. Number of UAVs')
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Plot avg distance_factor vs number of UAVs"
    )
    parser.add_argument(
        'json_files',
        nargs='+',
        help="One or more JSON metric files"
    )
    parser.add_argument(
        '--out', '-o',
        help="Path to save the plot (PNG). If omitted, show interactively.",
        default=None
    )
    args = parser.parse_args()

    df = aggregate_metrics(args.json_files)
    if df.empty:
        print("No valid data to plot.")
        return

    print("Data summary:")
    print(df[['total_uavs', 'avg_distance_factor']])
    plot_avg_distance_vs_num_uavs(df, output_path=args.out)

if __name__ == '__main__':
    main()