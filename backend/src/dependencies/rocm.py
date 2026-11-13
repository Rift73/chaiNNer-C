"""The dependencies the packages share on AMD GPUs: ROCm, ONNX Runtime with the MIGraphX
provider, Triton for ROCm and, on Windows, Triton for Windows.

ROCm 10.0.0 ships as pip wheels: rocm-sdk-core (the HIP runtime), rocm-sdk-libraries
(rocBLAS, hipBLASLt, MIOpen and the other math libraries) and one GPU kernel wheel per
architecture, rocm-sdk-device-gfx<arch>, which unpacks into the libraries' folder. AMD's
builds of PyTorch require AMD's meta package as rocm[libraries], and pip would build that
sdist and choose kernel wheels by the machine it runs on. The packages install the bundled
build of the meta package instead, whose device-all extra requires none
(native/tools/build_rocm_meta_wheel.py), and the kernel wheels of the GPUs found.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from api import KB, Dependency
from gpu import amd
from system import is_windows

ROCM_VERSION = "10.0.0"
# AMD's index: its builds of PyTorch for Linux and Windows, ROCm's wheels, AMD's Triton for
# Linux and its GPU detection helper, none of them on PyPI.
AMD_INDEX = "https://stable.repo.amd.com/rocm/whl-next/"
META_WHEEL = (
    Path(__file__).resolve().parent / "whls/rocm/rocm-10.0.0+chainner-py3-none-any.whl"
)
# AMD's Linux wheels the packages install, from AMD_INDEX, by distribution, with their sizes
# as listed there on 2026-11-07.
LINUX_WHEEL_SIZES = {
    "torch": 202641679,
    "torchvision": 1116376,
    "rocm-bootstrap": 30532,
    "rocm-sdk-core": 414668671,
    "rocm-sdk-libraries": 143349785,
    "amd-torch-device-gfx110x": 243089280,
    "amd-torch-device-gfx115x": 187800922,
    "amd-torch-device-gfx12-0": 157950127,
    "amd-torch-device-gfx90a": 149863462,
    "amd-torch-device-gfx908": 44059530,
    "amd-torch-device-gfx942": 171348226,
    "amd-torch-device-gfx950": 147351134,
    "amd-torch-device-gfx1010": 44538093,
    "amd-torch-device-gfx1011": 44538100,
    "amd-torch-device-gfx1012": 44538538,
    "amd-torch-device-gfx1030": 44666268,
    "amd-torch-device-gfx1031": 44771308,
    "amd-torch-device-gfx1032": 44771406,
    "amd-torch-device-gfx1033": 44771724,
    "amd-torch-device-gfx1034": 44771430,
    "amd-torch-device-gfx1035": 44771567,
    "amd-torch-device-gfx1036": 44771585,
    "amd-torch-device-gfx1100": 50747391,
    "amd-torch-device-gfx1101": 50567343,
    "amd-torch-device-gfx1102": 50584950,
    "amd-torch-device-gfx1103": 50558663,
    "amd-torch-device-gfx1150": 50051142,
    "amd-torch-device-gfx1151": 50080892,
    "amd-torch-device-gfx1152": 50051257,
    "amd-torch-device-gfx1153": 50051194,
    "amd-torch-device-gfx1200": 51503528,
    "amd-torch-device-gfx1201": 51684877,
    "amd-torch-device-gfx1250": 212117844,
    "amd-torchvision-device-gfx90a": 113341,
    "amd-torchvision-device-gfx908": 116935,
    "amd-torchvision-device-gfx942": 111109,
    "amd-torchvision-device-gfx950": 111252,
    "amd-torchvision-device-gfx1010": 117073,
    "amd-torchvision-device-gfx1011": 117066,
    "amd-torchvision-device-gfx1012": 117072,
    "amd-torchvision-device-gfx1030": 118456,
    "amd-torchvision-device-gfx1031": 118451,
    "amd-torchvision-device-gfx1032": 118470,
    "amd-torchvision-device-gfx1033": 118463,
    "amd-torchvision-device-gfx1034": 118482,
    "amd-torchvision-device-gfx1035": 118453,
    "amd-torchvision-device-gfx1036": 118449,
    "amd-torchvision-device-gfx1100": 140634,
    "amd-torchvision-device-gfx1101": 140876,
    "amd-torchvision-device-gfx1102": 140478,
    "amd-torchvision-device-gfx1103": 140625,
    "amd-torchvision-device-gfx1150": 136996,
    "amd-torchvision-device-gfx1151": 137139,
    "amd-torchvision-device-gfx1152": 137004,
    "amd-torchvision-device-gfx1153": 137005,
    "amd-torchvision-device-gfx1200": 132751,
    "amd-torchvision-device-gfx1201": 132749,
    "amd-torchvision-device-gfx1250": 117137,
    "rocm-sdk-device-gfx90a": 471290620,
    "rocm-sdk-device-gfx908": 303798571,
    "rocm-sdk-device-gfx942": 1608140394,
    "rocm-sdk-device-gfx950": 703892423,
    "rocm-sdk-device-gfx1010": 64068105,
    "rocm-sdk-device-gfx1011": 63719296,
    "rocm-sdk-device-gfx1012": 63719897,
    "rocm-sdk-device-gfx1030": 481324000,
    "rocm-sdk-device-gfx1031": 68743195,
    "rocm-sdk-device-gfx1032": 68747989,
    "rocm-sdk-device-gfx1033": 68750897,
    "rocm-sdk-device-gfx1034": 68741053,
    "rocm-sdk-device-gfx1035": 68741817,
    "rocm-sdk-device-gfx1036": 68741942,
    "rocm-sdk-device-gfx1100": 227892228,
    "rocm-sdk-device-gfx1101": 221272364,
    "rocm-sdk-device-gfx1102": 220134635,
    "rocm-sdk-device-gfx1103": 55632931,
    "rocm-sdk-device-gfx1150": 149232466,
    "rocm-sdk-device-gfx1151": 173730308,
    "rocm-sdk-device-gfx1152": 149060993,
    "rocm-sdk-device-gfx1153": 149221700,
    "rocm-sdk-device-gfx1200": 403651626,
    "rocm-sdk-device-gfx1201": 343803456,
    "rocm-sdk-device-gfx1250": 197075246,
}
LINUX_ARCHITECTURES = frozenset(
    name.removeprefix("rocm-sdk-device-")
    for name in LINUX_WHEEL_SIZES
    if name.startswith("rocm-sdk-device-")
)
# AMD's Windows wheels the packages install, by distribution: the file in the distribution's
# folder of AMD_INDEX, its SHA-256 and its size, as downloaded on 2026-11-07 (AMD publishes
# no hashes). The ROCm kernel wheels' architectures (no gfx942, gfx950 or gfx1250 on Windows)
# are ROCm's there.
WINDOWS_WHEELS = {
    "torch": (
        "torch-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "9a01e2a7f89ea89383f6cd5ae884ac4875edefa387c4d8c77f207ba6efd178b0",
        113446960,
    ),
    "torchvision": (
        "torchvision-0.28.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "2508a18eead7fb70ba35da6fcbc45bdc9b953d4ba44bb979a7fafebabebaa8b2",
        903406,
    ),
    "rocm-bootstrap": (
        "rocm_bootstrap-0.3.0-py3-none-any.whl",
        "62d0160ecdd86f7132dbab9aec257cb2c96327df2592b0633aa6bb56399cc095",
        30532,
    ),
    "rocm-sdk-core": (
        "rocm_sdk_core-10.0.0-py3-none-win_amd64.whl",
        "066195cb0f7df02e6009facb17652dd992d13fae44d87652e5ab928d5b4f0851",
        758050981,
    ),
    "rocm-sdk-libraries": (
        "rocm_sdk_libraries-10.0.0-py3-none-win_amd64.whl",
        "bfc7a1bd3b6050fb4d36d8ac2eea80d3afd70c5604a29142463d6cb890fc73bf",
        116831755,
    ),
    "amd-torch-device-gfx110x": (
        "amd_torch_device_gfx110x-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "3cd00796d96348424c7b6c9c9fb28560ef63e27d6cbb8a277ecf9771a0e5068b",
        243089281,
    ),
    "amd-torch-device-gfx115x": (
        "amd_torch_device_gfx115x-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "60ab3a029b1d1971a8b110f03cce468f5f6d81b9e4983438b6fb7947e41e7857",
        187800924,
    ),
    "amd-torch-device-gfx12-0": (
        "amd_torch_device_gfx12_0-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "d441c197769d657e88c8a122ab87e25a8e47aaa65751c6006cce733f3283d8a9",
        157950126,
    ),
    "amd-torch-device-gfx90a": (
        "amd_torch_device_gfx90a-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "906d3a1f6c4a0588a573d0945244b26683cae7b1b8206208b5087551f786c48a",
        149585304,
    ),
    "amd-torch-device-gfx908": (
        "amd_torch_device_gfx908-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "3a51ead4b997ecea2760dcd833b083fa69f0a527c003787415c0f27630762ee6",
        44356888,
    ),
    "amd-torch-device-gfx1010": (
        "amd_torch_device_gfx1010-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "a2a02c256d424fb8ea3913be61480d2e9966e893f9b65fca22b42bc35ea0858a",
        44671163,
    ),
    "amd-torch-device-gfx1011": (
        "amd_torch_device_gfx1011-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "a791f920055f57e44b8f037d33b1364f170b5f5d8978fbbb32fc9c15bdcd01c6",
        44671144,
    ),
    "amd-torch-device-gfx1012": (
        "amd_torch_device_gfx1012-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "e2707fa098ab0f1eebe241ae31095bcc5400fb43ef439c968e393fea81c32ad2",
        44671500,
    ),
    "amd-torch-device-gfx1030": (
        "amd_torch_device_gfx1030-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "7c3c5834d882a36b364b5049c03abf8a67ac943eca64924562b41a0e3e0ab5cc",
        44709997,
    ),
    "amd-torch-device-gfx1031": (
        "amd_torch_device_gfx1031-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "52209ecc74114143e9377b0b4ee0d25f556a8235678981588c6db551f2b05b03",
        44812822,
    ),
    "amd-torch-device-gfx1032": (
        "amd_torch_device_gfx1032-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "7b9a2bf3b4ddea8d528e65e3a1a724dc9a99f6c94363485fe9cb63afe4813ded",
        44812510,
    ),
    "amd-torch-device-gfx1033": (
        "amd_torch_device_gfx1033-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "2154eee712cc6cf8f6fb23b02a7b8d8b935d65c3a689000cb0bdba2455e366b2",
        44812784,
    ),
    "amd-torch-device-gfx1034": (
        "amd_torch_device_gfx1034-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "09a1c6d5bea595cbdaad21ac8b97916722c39f7e7b5f68948817223e231786a3",
        44812532,
    ),
    "amd-torch-device-gfx1035": (
        "amd_torch_device_gfx1035-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "c9dff924062a6b5844d35c1784a78da3584a01742e9d6d66022801f0ef2eff22",
        44812552,
    ),
    "amd-torch-device-gfx1036": (
        "amd_torch_device_gfx1036-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "f08885d3922acaf2aa2828a13b1879397af82097260c9f00c48a66463e46e40d",
        44812716,
    ),
    "amd-torch-device-gfx1100": (
        "amd_torch_device_gfx1100-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "af4b20fe5d6c0f119b0f60452f77ab4e9ac3ddbb7efede359ccbd7bb63a8b035",
        50515826,
    ),
    "amd-torch-device-gfx1101": (
        "amd_torch_device_gfx1101-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "97efdefaaec3f425386a3d9b48516c6c90bfa40aee7c1f9edfbaf1f92fc8c0c9",
        50459736,
    ),
    "amd-torch-device-gfx1102": (
        "amd_torch_device_gfx1102-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "dbb58caf1e506765d256695a271e05ebc7779131c3525ddefef60e69aac41bc2",
        50636986,
    ),
    "amd-torch-device-gfx1103": (
        "amd_torch_device_gfx1103-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "3f0058c257830ca9cf4143caf51f5e90070a5e2d7705b0a1b1f8ae4d7258be69",
        50585236,
    ),
    "amd-torch-device-gfx1150": (
        "amd_torch_device_gfx1150-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "4741d28a727d85f3aadc6bec9cce0949ae9bfd16e888b47b835572e86a81c111",
        50106849,
    ),
    "amd-torch-device-gfx1151": (
        "amd_torch_device_gfx1151-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "fa1f1db6d760bdf04d58cb24aecaba1268e3f8ae3eb402a903f40746d7657680",
        50011243,
    ),
    "amd-torch-device-gfx1152": (
        "amd_torch_device_gfx1152-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "2ba64f71bbe2a0bf6dce399bb90906be1bf2001db22ff7461c1ac1035785a1cc",
        50106669,
    ),
    "amd-torch-device-gfx1153": (
        "amd_torch_device_gfx1153-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "93f36d5a04957b358dddc462ec7b5cff131696c60c6c9597cf28aad90509a737",
        50107242,
    ),
    "amd-torch-device-gfx1200": (
        "amd_torch_device_gfx1200-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "4a42e9937c09862bb7c48a2644f9c634ac090970e30d0c240b3aa9cd51fb9421",
        51106952,
    ),
    "amd-torch-device-gfx1201": (
        "amd_torch_device_gfx1201-2.13.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "22f2de7ce0f50428fa5f7e309418d433cdd4aeb47834ec7bff6840d34dff6fa2",
        51123602,
    ),
    "amd-torchvision-device-gfx90a": (
        "amd_torchvision_device_gfx90a-0.28.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "f9b4bb415c9c5e1a50b647cfe621f5c057e9fcf113216428e400b828bf9ad169",
        112491,
    ),
    "amd-torchvision-device-gfx908": (
        "amd_torchvision_device_gfx908-0.28.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "4c4caefa9e1d1b28f5edd931e0963ef023277946f0e8a7df08ecac622ca05088",
        115425,
    ),
    "amd-torchvision-device-gfx1010": (
        "amd_torchvision_device_gfx1010-0.28.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "b1ed4a93b838cbfd13c9db3a3b05a75c11de35c1548a8340ca3724e9852dd9f2",
        115612,
    ),
    "amd-torchvision-device-gfx1011": (
        "amd_torchvision_device_gfx1011-0.28.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "dcebdd5aa436377680a9370b894d2c6687298465b97269233b6e40ef51f9b383",
        115624,
    ),
    "amd-torchvision-device-gfx1012": (
        "amd_torchvision_device_gfx1012-0.28.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "d0642cdcbbb034014be2134e09ab1dfd389fac4ed519e56f839d5be9e7c36e4e",
        115621,
    ),
    "amd-torchvision-device-gfx1030": (
        "amd_torchvision_device_gfx1030-0.28.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "dfcbd91b59c5ccef77ca6ae0f1f1736ddc5d43d48d4e5cf7b0bffb4b495d5c19",
        117045,
    ),
    "amd-torchvision-device-gfx1031": (
        "amd_torchvision_device_gfx1031-0.28.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "6eecf405b16c93ea3b76ed8d054ca14712ed7b8606b0eae916061c481b6fbfb0",
        117037,
    ),
    "amd-torchvision-device-gfx1032": (
        "amd_torchvision_device_gfx1032-0.28.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "cd32e4c78149e11088e5fabd3bda6383b9da036a38857e304a8da40e4e767ee1",
        117042,
    ),
    "amd-torchvision-device-gfx1033": (
        "amd_torchvision_device_gfx1033-0.28.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "0753e9bf9ec2b5bc967ce12520a77e265c5a40df71927954cb67d9d5bfbb14fb",
        117047,
    ),
    "amd-torchvision-device-gfx1034": (
        "amd_torchvision_device_gfx1034-0.28.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "c90062a0d929b3a9c05bb2d56bb4dcb3ec613ee1a369777d2d7756915468183c",
        117040,
    ),
    "amd-torchvision-device-gfx1035": (
        "amd_torchvision_device_gfx1035-0.28.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "ca3248a67c1aff6687160afde0f613b84f0a9468031d2fc33533abf09b76e49b",
        117041,
    ),
    "amd-torchvision-device-gfx1036": (
        "amd_torchvision_device_gfx1036-0.28.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "a276bb380214b4a851bf8dbd5f4c60e2c42ebb31c67ca1c16efc3edd370b791c",
        117035,
    ),
    "amd-torchvision-device-gfx1100": (
        "amd_torchvision_device_gfx1100-0.28.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "d80fcddb573291a96339fdd5e47d18edee72f801daf059bad826ec3e64eb41d1",
        138392,
    ),
    "amd-torchvision-device-gfx1101": (
        "amd_torchvision_device_gfx1101-0.28.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "6a684614883a1a85018de372bdf5994ddcc417afc4168bdb9761cd84ed341a6a",
        138675,
    ),
    "amd-torchvision-device-gfx1102": (
        "amd_torchvision_device_gfx1102-0.28.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "fc6954a26a7bbec5409b826db880ca9d277d767298b4ce395a3bc76f55981c29",
        138327,
    ),
    "amd-torchvision-device-gfx1103": (
        "amd_torchvision_device_gfx1103-0.28.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "ae07f810158b7d4961f54dc5fc38d02b7f4b4cdccfabd9eb324182217e96df3c",
        138471,
    ),
    "amd-torchvision-device-gfx1150": (
        "amd_torchvision_device_gfx1150-0.28.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "8b6aa943902e8169c69162f32978999b09ad4b40a429b9ccda85b122dce504a2",
        134771,
    ),
    "amd-torchvision-device-gfx1151": (
        "amd_torchvision_device_gfx1151-0.28.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "523e3bf06dcfe477cae97b32b54d03f48c2f28ae12dbe83386112129ca2c2cb4",
        134932,
    ),
    "amd-torchvision-device-gfx1152": (
        "amd_torchvision_device_gfx1152-0.28.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "321670593b64991929971ff30c2991c646cb01d2f371ccff62147f69a34bcc8d",
        134769,
    ),
    "amd-torchvision-device-gfx1153": (
        "amd_torchvision_device_gfx1153-0.28.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "1a3572e9f07438d1309925b746c20a7404ee229e7b1d866ff0c9156853872bf4",
        134764,
    ),
    "amd-torchvision-device-gfx1200": (
        "amd_torchvision_device_gfx1200-0.28.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "f2676ebde675908067fc6709e36098892c34aa687cda38dac3b4ccca6207a2f1",
        131254,
    ),
    "amd-torchvision-device-gfx1201": (
        "amd_torchvision_device_gfx1201-0.28.0%2Brocm10.0.0-cp314-cp314-win_amd64.whl",
        "f74692cef1e1abeaa651644af7ea192c02d86d4aa2c55c01325cd18be712bf53",
        131255,
    ),
    "rocm-sdk-device-gfx90a": (
        "rocm_sdk_device_gfx90a-10.0.0-py3-none-win_amd64.whl",
        "cf2466bba5f476473ca2d7ae3f633b1e3cd858ffb45b1f238440a50839cf2a76",
        271936316,
    ),
    "rocm-sdk-device-gfx908": (
        "rocm_sdk_device_gfx908-10.0.0-py3-none-win_amd64.whl",
        "d27dafc7732b4b71f2c51e797ac7f1669a26dfbc0042d09e5ecc52c98b6a8e27",
        159537671,
    ),
    "rocm-sdk-device-gfx1010": (
        "rocm_sdk_device_gfx1010-10.0.0-py3-none-win_amd64.whl",
        "2b6da925376de9689ab9f22ce7cc55b6d476722d9458fbd1ec52d4aa066d3e6c",
        47525108,
    ),
    "rocm-sdk-device-gfx1011": (
        "rocm_sdk_device_gfx1011-10.0.0-py3-none-win_amd64.whl",
        "1a6de3efe7bc92b1e55367a767eecf0468231b03d9319494d1e4b077571b6757",
        47190349,
    ),
    "rocm-sdk-device-gfx1012": (
        "rocm_sdk_device_gfx1012-10.0.0-py3-none-win_amd64.whl",
        "51f2998d2f33164e4a14594107e6c33db54da1d9867fa6947ea993997c4ae06e",
        47187798,
    ),
    "rocm-sdk-device-gfx1030": (
        "rocm_sdk_device_gfx1030-10.0.0-py3-none-win_amd64.whl",
        "19ada0359b499c19c1ecb1cba21b067fce89b4f0b0de6f0f7b76cbef85dcb85b",
        52153384,
    ),
    "rocm-sdk-device-gfx1031": (
        "rocm_sdk_device_gfx1031-10.0.0-py3-none-win_amd64.whl",
        "ab7cbfb41e1ada00fb7254c974f7386303cdd1139a170fb022e84e6f10d9f5f2",
        52179193,
    ),
    "rocm-sdk-device-gfx1032": (
        "rocm_sdk_device_gfx1032-10.0.0-py3-none-win_amd64.whl",
        "5d0581e077cc1a482e70430539c63e60c4b4d7b9b82be0ac0bab6d62c3feb4de",
        52182885,
    ),
    "rocm-sdk-device-gfx1033": (
        "rocm_sdk_device_gfx1033-10.0.0-py3-none-win_amd64.whl",
        "46ba5b395bf75756f9c5f1154b2083dd576221dfbd15620b5fa597f498aad798",
        52186039,
    ),
    "rocm-sdk-device-gfx1034": (
        "rocm_sdk_device_gfx1034-10.0.0-py3-none-win_amd64.whl",
        "8d527c2f127d42ecb098e0abc45c15ca28689b6317f01b99d455e6d3562a3b52",
        52183796,
    ),
    "rocm-sdk-device-gfx1035": (
        "rocm_sdk_device_gfx1035-10.0.0-py3-none-win_amd64.whl",
        "c06479bad4963f4aecde4e0fe102eb39ae652e80da62546c3bd3fb0b9079a0a4",
        52181539,
    ),
    "rocm-sdk-device-gfx1036": (
        "rocm_sdk_device_gfx1036-10.0.0-py3-none-win_amd64.whl",
        "655f343cd805f21bfd45244902f99afba9b7ada323288a1b9ba3336bd41c1ccc",
        52183283,
    ),
    "rocm-sdk-device-gfx1100": (
        "rocm_sdk_device_gfx1100-10.0.0-py3-none-win_amd64.whl",
        "db39cbc5aae7a67c73d76692899a6a89bbc10eafc92f856f48e9c03247a91f8e",
        189830160,
    ),
    "rocm-sdk-device-gfx1101": (
        "rocm_sdk_device_gfx1101-10.0.0-py3-none-win_amd64.whl",
        "7b8edd82943a11c29c80f84c95822e36ddd794b7024878e9660170d050a29243",
        185030273,
    ),
    "rocm-sdk-device-gfx1102": (
        "rocm_sdk_device_gfx1102-10.0.0-py3-none-win_amd64.whl",
        "b9e0f3fa492f0a05a5c4f479b80915477911f8402956c0e2c593b593d6214531",
        183114327,
    ),
    "rocm-sdk-device-gfx1103": (
        "rocm_sdk_device_gfx1103-10.0.0-py3-none-win_amd64.whl",
        "3de6ba204d78e0cb2a32cf064371c181d7a17d1ca8b25537b1a7f11eaa66fa17",
        55710049,
    ),
    "rocm-sdk-device-gfx1150": (
        "rocm_sdk_device_gfx1150-10.0.0-py3-none-win_amd64.whl",
        "22c19fcee580102cdc2f9379edf056643843c86ac6312dcf34d8ed21964c2cc4",
        134574199,
    ),
    "rocm-sdk-device-gfx1151": (
        "rocm_sdk_device_gfx1151-10.0.0-py3-none-win_amd64.whl",
        "1ab27ebfe61ea3c3fd2c5c1a0e780ae22ed9b662106ce0aefda0a30e3b11577f",
        139346623,
    ),
    "rocm-sdk-device-gfx1152": (
        "rocm_sdk_device_gfx1152-10.0.0-py3-none-win_amd64.whl",
        "88db4e79f31bef68bf8cb321fb83704dda6d11ad6da8dcfbdc8660f077c7195c",
        134480596,
    ),
    "rocm-sdk-device-gfx1153": (
        "rocm_sdk_device_gfx1153-10.0.0-py3-none-win_amd64.whl",
        "1fde64d87edd815403014743700000a016435f040c43114e764b5cc7b919413c",
        134629087,
    ),
    "rocm-sdk-device-gfx1200": (
        "rocm_sdk_device_gfx1200-10.0.0-py3-none-win_amd64.whl",
        "f33226078932ee2200f5595c521895edf5d7abf3170975d8c6f72a2a3949f516",
        377648520,
    ),
    "rocm-sdk-device-gfx1201": (
        "rocm_sdk_device_gfx1201-10.0.0-py3-none-win_amd64.whl",
        "c6e4c5a0da941230bed975765e644597570081c9a41654d5a052c277167f4576",
        312303802,
    ),
}
WINDOWS_ARCHITECTURES = frozenset(
    name.removeprefix("rocm-sdk-device-")
    for name in WINDOWS_WHEELS
    if name.startswith("rocm-sdk-device-")
)

# onnxruntime-migraphx 1.31.1 built with its MIGraphX provider against ROCm 10.0.0
# (manylinux_2_28), MIGraphX's libraries inside it; PyPI's last onnxruntime-migraphx is
# 1.27.1, for glibc 2.34 and a system ROCm 7.2.
ORT_MIGRAPHX_VERSION = "1.31.1+rocm10.0.hipgraph"
ORT_MIGRAPHX_URL = (
    "https://github.com/Rift73/onnxruntime/releases/download/v1.31.1-rocm10.0-hipgraph/"
    "onnxruntime_migraphx-1.31.1%2Brocm10.0.hipgraph-cp314-cp314-manylinux_2_28_x86_64.whl"
)
ORT_MIGRAPHX_SHA256 = "3856b2f06dc1fe30251b4649820555e00e39ea47fc219e0e41c81db8d54d8ece"
ORT_MIGRAPHX_SIZE = 101120585
# Its Windows build (win_amd64), in the same release: the DirectML provider beside
# MIGraphX's, MIGraphX 2.17.0's DLLs and the MSVC runtime inside it, ROCm's from the ROCm
# pip packages.
ORT_MIGRAPHX_WINDOWS_VERSION = "1.31.1+rocm10.0.hipgraph.dml"
ORT_MIGRAPHX_WINDOWS_URL = (
    "https://github.com/Rift73/onnxruntime/releases/download/v1.31.1-rocm10.0-hipgraph/"
    "onnxruntime_migraphx-1.31.1%2Brocm10.0.hipgraph.dml-cp314-cp314-win_amd64.whl"
)
ORT_MIGRAPHX_WINDOWS_SHA256 = (
    "64feb4c6fcf1c030f1a48844d83de0bf729c3b37df7637d0dd9ef0581eba9fa4"
)
ORT_MIGRAPHX_WINDOWS_SIZE = 80406026
# Triton's Windows build (win_amd64): triton_windows.
TRITON_WINDOWS_VERSION = "3.8.0.post29"
TRITON_WINDOWS_URL = (
    "https://files.pythonhosted.org/packages/0c/3c/6"
    "be3de0febab8f0708b32377c59d6f963a492622ce79762ec08ddc7496a8/triton_windows-3.8.0.post29-cp314-cp314-win_amd64.whl"
)
TRITON_WINDOWS_SHA256 = (
    "1d0fd6a6cf36ba5cced997177068e328d015e42cd4300f6995fff5a6f8707aff"
)
TRITON_WINDOWS_SIZE = 54097540
# The other builds of ONNX Runtime, which own its onnxruntime module too: the CPU build
# (Linux), DirectML's (Windows) and NVIDIA's (a computer whose NVIDIA GPU gave way to an
# AMD one). The installer uninstalls an installed one first.
ORT_BUILDS = ("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml")
# The AMD builds that NVIDIA's and the CPU builds replace in turn (a folder that ran on an
# AMD GPU, on a computer whose GPU is now an NVIDIA one or none): ONNX Runtime's MIGraphX
# build; Triton for ROCm, which PyTorch's ROCm build brings on Linux, named triton-rocm up
# to ROCm 7.14 and triton with a "+git" local version since, which triton==3.8.0 would
# accept; PyTorch's ROCm builds, which keep its distributions' names
# (Dependency.replaces' "+rocm"); and ROCm's runtime, which PyTorch's and ONNX Runtime's
# ROCm builds run on and nothing else uses (2 GB and more, and on Windows HIP's GPU listing
# at every start while it is installed): the meta package, the SDK and its GPU kernel
# wheels, AMD's GPU detection helper and PyTorch's and TorchVision's GPU kernel wheels
# (Dependency.replaces' "*"). The PyTorch and ONNX builds each remove the runtime: where
# the ONNX package changes first, PyTorch's ROCm build no longer loads until the PyTorch
# package changes too, as the dependency manager asks (it shows PyTorch missing).
ROCM_RUNTIME_BUILDS = (
    "rocm",
    "rocm-sdk-core",
    "rocm-sdk-libraries",
    "rocm-sdk-device-*",
    "rocm-bootstrap",
    "amd-torch-device-*",
    "amd-torchvision-device-*",
)
ORT_AMD_BUILDS = ("onnxruntime-migraphx", *ROCM_RUNTIME_BUILDS)
TRITON_AMD_BUILDS = ("triton-rocm", "triton+git")
TORCH_AMD_BUILDS = ("torch+rocm", *TRITON_AMD_BUILDS, *ROCM_RUNTIME_BUILDS)
TORCHVISION_AMD_BUILDS = ("torchvision+rocm",)


def gpu_architectures() -> tuple[list[str], list[str]]:
    """The architectures of the AMD GPUs worth running models on (amd.eligible's: a
    discrete GPU or a large APU, not a Ryzen's small graphics), each once: those ROCm has
    GPU kernels for, and the others. A GPU whose driver reports no architecture, or a
    small one beside them, is left out; none at all keeps the CPU builds."""
    kernels = WINDOWS_ARCHITECTURES if is_windows else LINUX_ARCHITECTURES
    found = {device.gfx for device in amd.devices if device.gfx and device.eligible}
    return sorted(found & kernels), sorted(found - kernels)


def rocm_wheel(display_name: str, pypi_name: str, version: str) -> Dependency:
    """One of AMD's wheels: by its pinned URL on Windows, from AMD's index on Linux."""
    if is_windows:
        file, sha256, size = WINDOWS_WHEELS[pypi_name]
        return Dependency(
            display_name=display_name,
            pypi_name=pypi_name,
            version=version,
            size_estimate=size,
            url=f"{AMD_INDEX}{pypi_name}/{file}#sha256={sha256}",
            auto_update=False,
        )
    return Dependency(
        display_name=display_name,
        pypi_name=pypi_name,
        version=version,
        size_estimate=LINUX_WHEEL_SIZES[pypi_name],
        extra_index_url=AMD_INDEX,
        auto_update=False,
    )


def rocm_runtime(architectures: Sequence[str]) -> list[Dependency]:
    """ROCm's runtime and libraries, the bundled meta package, and the GPU kernel wheels
    of the architectures."""
    meta = Dependency(
        display_name="ROCm",
        pypi_name="rocm",
        version=f"{ROCM_VERSION}+chainner",
        size_estimate=14 * KB,
        url=META_WHEEL.as_uri(),
        auto_update=False,
    )
    return [
        meta,
        rocm_wheel("ROCm runtime", "rocm-sdk-core", ROCM_VERSION),
        rocm_wheel("ROCm libraries", "rocm-sdk-libraries", ROCM_VERSION),
        *(
            rocm_wheel(
                f"ROCm GPU kernels ({architecture})",
                f"rocm-sdk-device-{architecture}",
                ROCM_VERSION,
            )
            for architecture in architectures
        ),
    ]


def onnxruntime_migraphx() -> Dependency:
    """ONNX Runtime's MIGraphX build for this system: on Windows the one that also has
    DirectML."""
    if is_windows:
        return Dependency(
            display_name="ONNX Runtime (DirectML, MIGraphX)",
            pypi_name="onnxruntime-migraphx",
            version=ORT_MIGRAPHX_WINDOWS_VERSION,
            size_estimate=ORT_MIGRAPHX_WINDOWS_SIZE,
            import_name="onnxruntime",
            url=f"{ORT_MIGRAPHX_WINDOWS_URL}#sha256={ORT_MIGRAPHX_WINDOWS_SHA256}",
            replaces=ORT_BUILDS,
        )
    return Dependency(
        display_name="ONNX Runtime (MIGraphX)",
        pypi_name="onnxruntime-migraphx",
        version=ORT_MIGRAPHX_VERSION,
        size_estimate=ORT_MIGRAPHX_SIZE,
        import_name="onnxruntime",
        url=f"{ORT_MIGRAPHX_URL}#sha256={ORT_MIGRAPHX_SHA256}",
        replaces=ORT_BUILDS,
    )


def triton_windows() -> Dependency:
    """Triton for Windows on AMD GPUs, which compiles DUAL's kernels for a gfx target with
    no GPU (Build Program, in a child process). 3.8.0.post29 is the floor: 3.5.1.post22,
    which the TensorRT package pinned for NVIDIA GPUs before 3.9.0.post30, fails on its
    temporary files there. It replaces nothing: no other Triton has a Windows wheel, and
    pip upgrades an older copy in a folder that ran on an NVIDIA GPU."""
    return Dependency(
        display_name="Triton",
        pypi_name="triton-windows",
        version=TRITON_WINDOWS_VERSION,
        size_estimate=TRITON_WINDOWS_SIZE,
        import_name="triton",
        url=f"{TRITON_WINDOWS_URL}#sha256={TRITON_WINDOWS_SHA256}",
    )


def triton_rocm() -> Dependency:
    """AMD's Triton (Linux), the build AMD's PyTorch pins. Its distribution is triton, as
    NVIDIA's, so pip replaces NVIDIA's copy; it replaces triton-rocm, the name of PyTorch's
    Triton for ROCm 7.14, which owns the triton module too."""
    return Dependency(
        display_name="Triton",
        pypi_name="triton",
        version="3.8.0+git4cff872c.rocm10.0.0",
        size_estimate=427336698,
        import_name="triton",
        extra_index_url=AMD_INDEX,
        replaces=("triton-rocm",),
    )
