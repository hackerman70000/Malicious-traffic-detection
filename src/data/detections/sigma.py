from pathlib import Path
from typing import List, Optional
from nfstream import NFPlugin
import pandas as pd
from sigma.collection import SigmaCollection
from sigma.backends.pd_df.pd_df import PandasDataFramePythonBackend
class SigmaDetections(NFPlugin):
    def __init__(self, sigma_paths: Optional[List[str,Path]]=None, **kwargs):
        if sigma_paths is not None:
            self.collection = SigmaCollection()
            self.collection.load_ruleset(sigma_paths)
            self.sigma = list(map(lambda query: eval(f"lambda df: {query}"), PandasDataFramePythonBackend().convert(self.collection)))
        else:
            self.sigma = []
    def on_init(self, packet, flow):
        flow.udps.enrichments = flow.udps.enrichments if hasattr(flow.udps, "enrichments") else {}
        flow.udps.enrichments["sigma"] = {}
        flow.udps.detections = flow.udps.detections if hasattr(flow.udps, "detections") else {}
        self.on_update(packet, flow)
    def on_expire(self, flow):
        if len(self.sigma) > 0:
            df = pd.DataFrame.from_records(flow)
            for query in self.sigma:
                if query(df).size > 0:
                    flow.udps.detections["sigma"] = flow.udps.detections.get("sigma", 0) + 1
                    break
