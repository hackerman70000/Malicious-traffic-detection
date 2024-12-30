from functools import cache
from types import SimpleNamespace
from nfstream import NFPlugin
from spyip import trace_ip

class GeoIpEnrichment(NFPlugin):
    def on_init(self, packet, flow):
        flow.udps.enrichments = flow.udps.enrichments if hasattr(flow.udps, "enrichments") else {}
        flow.udps.enrichments["geoip"] = {}
        self.on_update(packet, flow)
    @staticmethod
    @cache
    def _get_geoip(ip):
        try:
            return trace_ip(ip)
        except Exception as e:
            return SimpleNamespace(status="error", error=str(e))
    def on_expire(self, flow):
        result = GeoIpEnrichment._get_geoip(flow.src_ip)
        if result.status == "success":
            flow.udps.enrichments["geoip_src"] = result.json()
        
        result = GeoIpEnrichment._get_geoip(flow.dst_ip)
        if result.status == "success":
            flow.udps.enrichments["geoip_dst"] = result.json()
    def cleanup(self):
        GeoIpEnrichment._get_geoip.cache_clear()