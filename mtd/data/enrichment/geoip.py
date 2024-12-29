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
    def on_update(self, packet, flow):
        if packet.src_ip not in flow.udps.enrichments["geoip"]:
            result = GeoIpEnrichment._get_geoip(packet.src_ip)
            if result.status == "success":
                flow.udps.enrichments["geoip"][str(packet.src_ip)] = result.json()
    def cleanup(self):
        GeoIpEnrichment._get_geoip.cache_clear()