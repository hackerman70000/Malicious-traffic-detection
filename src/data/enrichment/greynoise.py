from nfstream import NFPlugin
from greynoise import GreyNoise

class GreyNoiseEnrichment(NFPlugin):
    def __init__(self, greynoise_api_key=None, **kwargs):
        if greynoise_api_key:
            self.greynoise = GreyNoise(api_key=greynoise_api_key)
        else:
            self.greynoise = GreyNoise()
    def on_init(self, packet, flow):
        flow.udps.enrichments = flow.udps.enrichments or {}
        flow.udps.enrichments["greynoise"] = {}
        flow.udps.detections = flow.udps.detections or {}
    def on_update(self, packet, flow):
        if packet.ip_src not in flow.udps.enrichments["greynoise"]:
            result = self.greynoise.quick(packet.ip_src)
            flow.udps.enrichments["greynoise"][packet.ip_src] = result
            if not result["noise"]:
                flow.udps.detections[packet.ip_src] += 1

