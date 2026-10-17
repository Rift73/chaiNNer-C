import onnx

def convert_to_mixed_precision(
    onnx_path: str,
    low_precision_type: str = ...,
    nodes_to_exclude: list[str] | None = ...,
    op_types_to_exclude: list[str] | None = ...,
    nodes_to_include: list[str] | None = ...,
    op_types_to_include: list[str] | None = ...,
    data_max: float = ...,
    init_max: float = ...,
    keep_io_types: bool = ...,
    calibration_data: str | None = ...,
    providers: list[str] = ...,
    opset: int | None = ...,
) -> onnx.ModelProto: ...
