import sys

from nfstream import NFStreamer

print(f"Processing file: {sys.argv[2]}")

streamer = NFStreamer(source=sys.argv[2], statistical_analysis=True)
flows = streamer.to_pandas()
flows["label"] = int(sys.argv[1])

for col in flows.columns:
    if flows[col].nunique() == 1 or flows[col].isnull().any():
        flows.drop(col, inplace=True, axis=1)

flows.to_csv(f"{sys.argv[2]}.csv", index=False)

## Usage:
## find ./pcaps/Dataset/Botnet-Capture -name '*.pcap' | xargs -n 1 uv run python ./pcap2flow2.py 1
