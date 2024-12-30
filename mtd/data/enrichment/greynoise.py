from nfstream import NFPlugin
from greynoise import GreyNoise

class GreyNoiseEnrichment(NFPlugin):
    def __init__(self, greynoise_api_key=None, **kwargs):
        if greynoise_api_key:
            self.greynoise = GreyNoise(api_key=greynoise_api_key)
        else:
            self.greynoise = GreyNoise()
    def on_init(self, packet, flow):
        flow.udps.enrichments = flow.udps.enrichments if hasattr(flow.udps, "enrichments") else {}
        flow.udps.enrichments["greynoise"] = {}
        flow.udps.detections = flow.udps.detections if hasattr(flow.udps, "detections") else 0
        self.on_update(packet, flow)
    def on_expire(self, flow):
        try:
            result = self.greynoise.quick(flow.src_ip)
            flow.udps.enrichments["greynoise"] = result
            if result and isinstance(result, dict) and not result["noise"]:
                flow.udps.detections +=  + 1
        except Exception as _:
            pass
