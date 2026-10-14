"""Turning the results of imageautorater-rate into metadata: the XMP packet that both writers use.

No torch, no models: this module only reads iqa_results.json and builds/merges XMP. The two commands that store the
results are imageautorater-write-xmp (sidecar files) and imageautorater-write-exif (inside the image files).
"""
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "iqa": "http://ns.example.com/pyiqa/1.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
}
# namespaces commonly found in Lightroom/darktable metadata, registered so ElementTree keeps their prefixes
EXTRA_NS = {
    "crs": "http://ns.adobe.com/camera-raw-settings/1.0/",
    "lr": "http://ns.adobe.com/lightroom/1.0/",
    "photoshop": "http://ns.adobe.com/photoshop/1.0/",
    "exif": "http://ns.adobe.com/exif/1.0/",
    "tiff": "http://ns.adobe.com/tiff/1.0/",
    "aux": "http://ns.adobe.com/exif/1.0/aux/",
    "xmpMM": "http://ns.adobe.com/xap/1.0/mm/",
    "stEvt": "http://ns.adobe.com/xap/1.0/sType/ResourceEvent#",
    "darktable": "http://darktable.sf.net/",
    "mwg-rs": "http://www.metadataworkinggroup.com/schemas/regions/",
    "Iptc4xmpCore": "http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/",
}
for _p, _u in {**EXTRA_NS, **NS}.items():
    ET.register_namespace(_p, _u)

EMPTY_XMP = (
    '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
    '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    '<rdf:Description rdf:about=""/></rdf:RDF></x:xmpmeta>'
)
XPACKET_HEAD = '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
XPACKET_TAIL = '\n<?xpacket end="w"?>\n'
RDF_LI = "{%s}li" % NS["rdf"]


def q(prefix, name):
    return "{%s}%s" % (NS[prefix], name)


def xml_name(model: str) -> str:
    """Model name -> valid XML name part (e.g. 'clipiqa+' -> 'clipiqaplus')."""
    return re.sub(r"[^A-Za-z0-9_]", "_", model.replace("+", "plus"))


@dataclass
class ImageMeta:
    """What the results say about one image."""
    rating: "int | None" = None                 # rounded mean stars: becomes xmp:Rating / EXIF Rating
    mean: "float | None" = None
    models: dict = field(default_factory=dict)  # model -> (stars, score)
    keywords: "list | None" = None              # people first, then image-type tags; None = not analysed, leave alone


def keywords_from_entry(entry: dict):
    """Keywords for one image: recognised people first, then image-type tags. None when neither was analysed."""
    if "people" not in entry and "tags" not in entry:
        return None
    kws = []
    for pr in entry.get("people", []):
        if pr.get("name") and pr["name"] not in kws:
            kws.append(pr["name"])
    for t in entry.get("tags", {}):
        if t not in kws:
            kws.append(t)
    return kws


def iter_meta(results: dict):
    """(relative path, ImageMeta) for every image in an iqa_results.json."""
    for rel, e in sorted(results.get("images", {}).items()):
        models = {m: (v["rating"], v["score"]) for m, v in e.get("models", {}).items()}
        yield rel, ImageMeta(e.get("rating"), e.get("rating_mean"), models, keywords_from_entry(e))


def parse_xmp(data: "bytes | str | None"):
    """XMP packet (with or without the xpacket wrapper) -> root element. None/empty -> a new empty packet.
    Raises ET.ParseError for unreadable XML."""
    if not data:
        return ET.fromstring(EMPTY_XMP)
    return ET.fromstring(data)


def description(root):
    desc = root.find(".//" + q("rdf", "Description"))
    if desc is None:
        rdf = root.find(".//" + q("rdf", "RDF"))
        desc = ET.SubElement(rdf, q("rdf", "Description"), {q("rdf", "about"): ""})
    return desc


def render_xmp(root) -> bytes:
    ET.indent(root)
    return (XPACKET_HEAD + ET.tostring(root, encoding="unicode") + XPACKET_TAIL).encode("utf-8")


def previous_auto_keywords(root) -> set:
    return {k for k in (description(root).get(q("iqa", "AutoKeywords")) or "").split("|") if k}


def keyword_list(root) -> list:
    """The dc:subject keywords of an XMP packet."""
    subj = description(root).find(q("dc", "subject"))
    return [(li.text or "").strip() for li in subj.iter(RDF_LI)] if subj is not None else []


def merge_keywords(desc, new):
    """Replace keywords this tool added earlier (iqa:AutoKeywords); keep every keyword you added yourself."""
    prev = {k for k in (desc.get(q("iqa", "AutoKeywords")) or "").split("|") if k}
    subj = desc.find(q("dc", "subject"))
    if subj is None:
        if not new and not prev:
            return
        subj = ET.SubElement(desc, q("dc", "subject"))
    bag = next(iter(subj), None)
    if bag is None:
        bag = ET.SubElement(subj, q("rdf", "Bag"))
    existing = []
    for li in list(bag):
        txt = (li.text or "").strip()
        if txt in prev:
            bag.remove(li)
        else:
            existing.append(txt.lower())
    added = []  # only keywords we actually added, so a keyword you already had is never removed later
    for k in new:
        if k.lower() not in existing:
            ET.SubElement(bag, RDF_LI).text = k
            added.append(k)
    if added:
        desc.set(q("iqa", "AutoKeywords"), "|".join(added))
    elif q("iqa", "AutoKeywords") in desc.attrib:
        del desc.attrib[q("iqa", "AutoKeywords")]
    if len(bag) == 0:
        desc.remove(subj)


def apply_meta(root, meta: ImageMeta, rating=True, model_fields=True, keywords=True):
    """Merge one image's results into an XMP root. Everything else in the packet is preserved."""
    desc = description(root)

    def put(prefix, name, value):
        for child in desc.findall(q(prefix, name)):  # element form would shadow the attribute
            desc.remove(child)
        desc.set(q(prefix, name), value)

    if model_fields and meta.models:
        for model, (stars, score) in meta.models.items():
            n = xml_name(model)
            put("iqa", f"Rating_{n}", str(stars))
            put("iqa", f"Score_{n}", f"{score:.5f}")
        mean = meta.mean if meta.mean is not None else sum(s for s, _ in meta.models.values()) / len(meta.models)
        put("iqa", "RatingMean", f"{mean:.2f}")
        put("iqa", "ModelCount", str(len(meta.models)))
    if rating and meta.rating is not None:
        put("xmp", "Rating", str(int(meta.rating)))
    if keywords and meta.keywords is not None:
        merge_keywords(desc, meta.keywords)
    return root
