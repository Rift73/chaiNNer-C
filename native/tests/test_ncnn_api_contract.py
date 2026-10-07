"""Names the backend's ncnn sites use on ncnn objects exist in the PyPI ncnn wheel.

Upstream wrote these sites for the ncnn_vulkan bindings. PyPI ncnn's Extractor has
none of their Vulkan allocator setters, so every NCNN upscale on a Vulkan GPU raised
AttributeError (D-15); a fake Extractor in a test cannot catch that. This reads the
sites' source and looks every attribute of a known receiver up on the wheel's real
class. The receivers are the names the sites give their ncnn objects. Only class
objects are inspected, so nothing here creates a Vulkan instance.
"""

import ast
from pathlib import Path

from ncnn import ncnn

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend" / "src"
AUTO_SPLIT = "nodes/impl/ncnn/auto_split.py"
SESSION = "nodes/impl/ncnn/session.py"
UPSCALE = "packages/chaiNNer_ncnn/ncnn/processing/upscale_image.py"
SETTINGS = "packages/chaiNNer_ncnn/settings.py"

# receiver as written in the sites: the wheel object it names
RECEIVERS = {
    "ncnn": ncnn,
    "net": ncnn.Net,
    "net.opt": ncnn.Option,
    "default_net_opt": ncnn.Option,
    "ex": ncnn.Extractor,
    "vkdev": ncnn.VulkanDevice,
}


def receiver(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return None


def used(source: str) -> set[tuple[str, str]]:
    """(receiver, attribute) for every attribute the source reads or writes on one."""
    return {
        (name, node.attr)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute)
        and (name := receiver(node.value)) in RECEIVERS
    }


def missing(source: str) -> list[tuple[str, str]]:
    return sorted((r, a) for r, a in used(source) if not hasattr(RECEIVERS[r], a))


def test_every_name_the_ncnn_sites_use_exists_on_the_wheel():
    found = {}
    for site in (AUTO_SPLIT, SESSION, UPSCALE, SETTINGS):
        source = (BACKEND / site).read_text(encoding="utf-8")
        found[site] = used(source)
        assert missing(source) == [], site
    assert {
        ("ex", "input"),
        ("ex", "extract"),
        ("net", "create_extractor"),
        ("net.opt", "blob_vkallocator"),
        ("net.opt", "workspace_vkallocator"),
        ("net.opt", "staging_vkallocator"),
    } <= found[AUTO_SPLIT]
    assert {("net", "set_vulkan_device"), ("net.opt", "use_vulkan_compute")} <= found[
        SESSION
    ]
    assert {
        ("vkdev", "acquire_blob_allocator"),
        ("vkdev", "acquire_staging_allocator"),
        ("ncnn", "VkBlobAllocator"),
        ("ncnn", "VkStagingAllocator"),
    } <= found[UPSCALE]
    assert ("ncnn", "get_gpu_count") in found[SETTINGS]


def test_a_name_the_wheel_lacks_is_reported():
    source = (
        "ex = net.create_extractor()\n"
        "ex.set_blob_vkallocator(blob)\n"
        "net.opt.blob_vkallocator = blob\n"
        "net.opt.blob_vk_allocator = blob\n"
    )
    assert missing(source) == [
        ("ex", "set_blob_vkallocator"),
        ("net.opt", "blob_vk_allocator"),
    ]
