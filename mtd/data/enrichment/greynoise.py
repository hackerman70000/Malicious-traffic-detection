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
    def on_update(self, packet, flow):
        if packet.src_ip not in flow.udps.enrichments["greynoise"]:
            try:
                result = self.greynoise.quick(packet.src_ip)
                flow.udps.enrichments["greynoise"][packet.src_ip] = result
                if result and isinstance(result, dict) and not result["noise"]:
                    flow.udps.detections +=  + 1
            except Exception as e:
                flow.udps.enrichments["greynoise"][packet.src_ip] = {"error": str(e)}
