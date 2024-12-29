from pathlib import Path
import re
from typing import List, Optional
from nfstream import NFPlugin
import pandas as pd
from sigma.collection import SigmaCollection
from sigma.backends.pd_df.pd_df import PandasDataFramePythonBackend

class SigmaDetections(NFPlugin):
    def __init__(self, sigma_paths: Optional[List[Path]]=None, **kwargs):
        if sigma_paths is not None:
            rules = SigmaCollection.load_ruleset(sigma_paths)
            self.collection = SigmaCollection(rules)
            self.sigma = list(map(lambda query: (query[0], eval(f"lambda df: {query[1]}")), zip(self.collection, PandasDataFramePythonBackend().convert(self.collection))))
        else:
            self.sigma = []
    def on_init(self, packet, flow):
        flow.udps.enrichments = flow.udps.enrichments if hasattr(flow.udps, "enrichments") else {}
        flow.udps.enrichments["sigma"] = {}
        flow.udps.detections = flow.udps.detections if hasattr(flow.udps, "detections") else 0
        self.on_update(packet, flow)
    def on_expire(self, flow):
        if len(self.sigma) > 0:
            df = pd.DataFrame(flow.values(), index=flow.keys()).transpose()
            for query in self.sigma:
                try:
                    if query[1](df).size > 0:
                        flow.udps.detection += 1
                        flow.udps.enrichments["sigma"][query[0].title] = {
                            "detected": True,
                            "query": query[0],
                            "error": None
                        }
                        break
                    else:
                        flow.udps.enrichments["sigma"][query[0].title] = {
                            "detected": False,
                            "query": query[0].to_dict(),
                            "error": None,
                        }
                except Exception as e:
                    flow.udps.enrichments["sigma"][query[0].title] = {
                        "detected": False,
                        "query": query[0].to_dict(),
                        "error": str(e),
                    }