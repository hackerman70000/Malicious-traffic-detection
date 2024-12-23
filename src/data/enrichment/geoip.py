from functools import cache
from nfstream import NFPlugin
from spyip import trace_ip

class GeoIpEnrichment(NFPlugin):
    def __init__(self):
        self.cache 
    def on_init(self, packet, flow):
        flow.udps.enrichments = flow.udps.enrichments or {}
        flow.udps.enrichments["geoip"] = {}
    @staticmethod
    @cache
    def _get_geoip(ip):
        return trace_ip(ip)
    def on_update(self, packet, flow):
        if packet.ip_src not in flow.udps.enrichments["geoip"]:
            result = GeoIpEnrichment._get_geoip(packet.ip_src)
            if result.status == "success":
                flow.udps.enrichments["geoip"][packet.ip_src] = result
    def cleanup(self):
        GeoIpEnrichment._get_geoip.cache_clear()