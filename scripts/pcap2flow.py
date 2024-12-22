#!/usr/bin/env python3

import sys

from nfstream import NFStreamer


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <label> <pcap_file>")
        sys.exit(1)

    label = int(sys.argv[1])
    pcap_file = sys.argv[2]

    print(f"Processing file: {pcap_file}")

    streamer = NFStreamer(source=pcap_file, statistical_analysis=True)
    flows = streamer.to_pandas()

    flows["label"] = label

    for col in flows.columns:
        if flows[col].nunique() == 1 or flows[col].isnull().any():
            flows.drop(col, inplace=True, axis=1)

    output_csv = f"{pcap_file}.csv"
    flows.to_csv(output_csv, index=False)
    print(f"Output saved to: {output_csv}")


if __name__ == "__main__":
    main()
