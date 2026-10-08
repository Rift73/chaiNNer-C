// Fixed C++ translation of the frozen installed optimizer with the CORRECTIONS
// of generate_optimizer_cpp.py. Do not hand-edit.
// Source SHA256: 3b79fe76f4f1b8d5317b838e1471b41a27a281bab21ead47b39427143e4bd4ca
// Frozen installed optimizer.py:13
void pass_fuse_batchnorm_scale(const O &model, const O &types) {
    O batchnorm_output = py::none(), bias = py::none(), i = py::none(), j = py::none(), layer = py::none(), scale = py::none();
    (void)types;
    py::ssize_t cn_2 = 0;
    for (py::handle cn_1 : py::reinterpret_borrow<py::iterable>(attr(model, "layers"))) {
        i = py::int_(cn_2++);
        layer = py::reinterpret_borrow<O>(cn_1);
        if (compare(attr(layer, "op_type"), py::str("BatchNorm"), Py_EQ)) {
            batchnorm_output = item(attr(layer, "outputs"), py::int_(0));
            j = i;
            bool cn_3 = false;
            for (py::handle cn_4 : py::reinterpret_borrow<py::iterable>(builtin("range")(binary(i, py::int_(1), Op::add), py::int_(py::len(attr(model, "layers")))))) {
                j = py::reinterpret_borrow<O>(cn_4);
                if (compare(attr(item(attr(model, "layers"), j), "op_type"), py::str("Scale"), Py_NE)) {
                    continue;
                }
                if (compare(py::int_(py::len(attr(item(attr(model, "layers"), j), "inputs"))), py::int_(1), Py_NE)) {
                    continue;
                }
                if (compare(item(attr(item(attr(model, "layers"), j), "inputs"), py::int_(0)), batchnorm_output, Py_EQ)) {
                    cn_3 = true;
                    break;
                }
            }
            if (!cn_3) {
                j = inplace(j, py::int_(1), Op::add);
            }
            if (compare(j, py::int_(py::len(attr(model, "layers"))), Py_EQ)) {
                continue;
            }
            scale = item(attr(model, "layers"), j);
            bias = attr(item(attr(layer, "weight_data"), py::str("bias")), "weight");
            set_attr(item(attr(layer, "weight_data"), py::str("slope")), "weight", binary(attr(item(attr(layer, "weight_data"), py::str("slope")), "weight"), attr(item(attr(scale, "weight_data"), py::str("scale")), "weight"), Op::mul));
            bias = binary(bias, attr(item(attr(scale, "weight_data"), py::str("scale")), "weight"), Op::mul);
            if (truth(attr(item(attr(scale, "params"), py::int_(1)), "value"))) {
                bias = inplace(bias, attr(item(attr(scale, "weight_data"), py::str("bias")), "weight"), Op::add);
            }
            set_attr(item(attr(layer, "weight_data"), py::str("bias")), "weight", bias);
            set_item(attr(layer, "outputs"), py::int_(0), item(attr(scale, "outputs"), py::int_(0)));
            set_attr(model, "node_count", inplace(attr(model, "node_count"), py::int_(1), Op::sub));
            set_attr(model, "blob_count", inplace(attr(model, "blob_count"), py::int_(1), Op::sub));
            set_attr(scale, "op_type", py::str("ncnnfused"));
        }
    }
}

// Frozen installed optimizer.py:53
void pass_fuse_x_batchnorm(const O &model, const O &types) {
    O a = py::none(), b = py::none(), batchnorm = py::none(), bias_term = py::none(), channels = py::none(), conv_output = py::none(), eps = py::none(), i = py::none(), j = py::none(), layer = py::none(), sqrt_var = py::none(), weight = py::none();
    (void)types;
    py::ssize_t cn_6 = 0;
    for (py::handle cn_5 : py::reinterpret_borrow<py::iterable>(attr(model, "layers"))) {
        i = py::int_(cn_6++);
        layer = py::reinterpret_borrow<O>(cn_5);
        if (contains(make_tuple({py::str("Convolution"), py::str("ConvolutionDepthWise"), py::str("Deconvolution"), py::str("DeconvolutionDepthWise"), py::str("InnerProduct")}), attr(layer, "op_type"))) {
            conv_output = item(attr(layer, "outputs"), py::int_(0));
            j = i;
            bool cn_7 = false;
            for (py::handle cn_8 : py::reinterpret_borrow<py::iterable>(builtin("range")(binary(i, py::int_(1), Op::add), py::int_(py::len(attr(model, "layers")))))) {
                j = py::reinterpret_borrow<O>(cn_8);
                if (compare(attr(item(attr(model, "layers"), j), "op_type"), py::str("BatchNorm"), Py_NE)) {
                    continue;
                }
                if (compare(py::int_(py::len(attr(item(attr(model, "layers"), j), "inputs"))), py::int_(1), Py_NE)) {
                    continue;
                }
                if (compare(item(attr(item(attr(model, "layers"), j), "inputs"), py::int_(0)), conv_output, Py_EQ)) {
                    cn_7 = true;
                    break;
                }
            }
            if (!cn_7) {
                j = inplace(j, py::int_(1), Op::add);
            }
            if (compare(j, py::int_(py::len(attr(model, "layers"))), Py_EQ)) {
                continue;
            }
            batchnorm = item(attr(model, "layers"), j);
            channels = checked(builtin("int"), attr(item(attr(batchnorm, "params"), py::int_(0)), "value"));
            eps = checked(builtin("float"), attr(item(attr(batchnorm, "params"), py::int_(1)), "value"));
            a = array_empty(make_tuple({channels}));
            b = array_empty(make_tuple({channels}));
            sqrt_var = array_sqrt(binary(attr(item(attr(batchnorm, "weight_data"), py::str("variance")), "weight"), eps, Op::add));
            a = binary(attr(item(attr(batchnorm, "weight_data"), py::str("bias")), "weight"), binary(binary(attr(item(attr(batchnorm, "weight_data"), py::str("slope")), "weight"), attr(item(attr(batchnorm, "weight_data"), py::str("mean")), "weight"), Op::mul), sqrt_var, Op::div), Op::sub);
            b = binary(attr(item(attr(batchnorm, "weight_data"), py::str("slope")), "weight"), sqrt_var, Op::div);
            bias_term = (compare(attr(layer, "op_type"), py::str("InnerProduct"), Py_EQ) ? O(py::int_(1)) : O(py::int_(5)));
            if (compare(attr(item(attr(layer, "params"), bias_term), "value"), py::int_(0), Py_EQ)) {
                set_item(attr(layer, "params"), bias_term, py::int_(1));
                attr(layer, "add_weight")(py::str("bias"), array_zeros(channels, np().attr("float32")));
            }
            weight = attr(item(attr(layer, "weight_data"), py::str("weight")), "weight");
            set_attr(item(attr(layer, "weight_data"), py::str("weight")), "weight", binary(weight, array_transpose(array_cast(array_broadcast(b, item(attr(weight, "shape"), py::slice(py::none(), py::none(), negative(py::int_(1))))), attr(weight, "dtype")), make_tuple({py::int_(3), py::int_(2), py::int_(1), py::int_(0)})), Op::mul));
            set_attr(item(attr(layer, "weight_data"), py::str("bias")), "weight", binary(binary(attr(item(attr(layer, "weight_data"), py::str("bias")), "weight"), b, Op::mul), a, Op::add));
            set_item(attr(item(attr(model, "layers"), i), "outputs"), py::int_(0), item(attr(item(attr(model, "layers"), j), "outputs"), py::int_(0)));
            set_attr(model, "node_count", inplace(attr(model, "node_count"), py::int_(1), Op::sub));
            set_attr(model, "blob_count", inplace(attr(model, "blob_count"), py::int_(1), Op::sub));
            set_attr(batchnorm, "op_type", py::str("ncnnfused"));
        }
    }
}

// Frozen installed optimizer.py:127
void pass_fuse_x_mul(const O &model, const O &types) {
    O binaryop = py::none(), channels = py::none(), data = py::none(), i = py::none(), j = py::none(), k = py::none(), layer = py::none(), memorydata = py::none(), output = py::none(), weight = py::none();
    (void)types;
    py::ssize_t cn_10 = 0;
    for (py::handle cn_9 : py::reinterpret_borrow<py::iterable>(attr(model, "layers"))) {
        i = py::int_(cn_10++);
        layer = py::reinterpret_borrow<O>(cn_9);
        if (contains(make_tuple({py::str("Convolution"), py::str("ConvolutionDepthWise"), py::str("Deconvolution")}), attr(layer, "op_type"))) {
            output = item(attr(layer, "outputs"), py::int_(0));
            j = i;
            bool cn_11 = false;
            for (py::handle cn_12 : py::reinterpret_borrow<py::iterable>(builtin("range")(binary(i, py::int_(1), Op::add), py::int_(py::len(attr(model, "layers")))))) {
                j = py::reinterpret_borrow<O>(cn_12);
                if (compare(attr(item(attr(model, "layers"), j), "op_type"), py::str("BinaryOp"), Py_NE)) {
                    continue;
                }
                if (compare(attr(item(attr(model, "layers"), j), "num_inputs"), py::int_(2), Py_NE)) {
                    continue;
                }
                if (compare(item(attr(item(attr(model, "layers"), j), "inputs"), py::int_(0)), output, Py_EQ)) {
                    cn_11 = true;
                    break;
                }
            }
            if (!cn_11) {
                j = inplace(j, py::int_(1), Op::add);
            }
            if (compare(j, py::int_(py::len(attr(model, "layers"))), Py_EQ)) {
                continue;
            }
            binaryop = item(attr(model, "layers"), j);
            if ((compare(attr(item(attr(binaryop, "params"), py::int_(0)), "value"), types.attr("BinaryOpTypes").attr("MUL"), Py_NE) || truth(attr(item(attr(binaryop, "params"), py::int_(1)), "value")))) {
                continue;
            }
            k = py::int_(0);
            bool cn_13 = false;
            for (py::handle cn_14 : py::reinterpret_borrow<py::iterable>(builtin("range")(j))) {
                k = py::reinterpret_borrow<O>(cn_14);
                if (compare(attr(item(attr(model, "layers"), k), "op_type"), py::str("MemoryData"), Py_NE)) {
                    continue;
                }
                if (compare(item(attr(item(attr(model, "layers"), k), "outputs"), py::int_(0)), item(attr(binaryop, "inputs"), py::int_(1)), Py_EQ)) {
                    cn_13 = true;
                    break;
                }
            }
            if (!cn_13) {
                k = inplace(k, py::int_(1), Op::add);
            }
            if (compare(k, j, Py_EQ)) {
                continue;
            }
            memorydata = item(attr(model, "layers"), k);
            channels = checked(builtin("int"), attr(item(attr(layer, "params"), py::int_(0)), "value"));
            if ((compare(attr(item(attr(memorydata, "params"), py::int_(0)), "value"), channels, Py_NE) || compare(attr(item(attr(memorydata, "params"), py::int_(1)), "value"), py::int_(0), Py_NE) || compare(attr(item(attr(memorydata, "params"), py::int_(2)), "value"), py::int_(0), Py_NE))) {
                continue;
            }
            data = attr(item(attr(memorydata, "weight_data"), py::str("data")), "weight");
            weight = attr(item(attr(layer, "weight_data"), py::str("weight")), "weight");
            set_attr(item(attr(layer, "weight_data"), py::str("weight")), "weight", binary(weight, array_transpose(array_cast(array_broadcast(data, item(attr(weight, "shape"), py::slice(py::none(), py::none(), negative(py::int_(1))))), attr(weight, "dtype")), make_tuple({py::int_(3), py::int_(2), py::int_(1), py::int_(0)})), Op::mul));
            try {
                set_attr(item(attr(layer, "weight_data"), py::str("bias")), "weight", binary(attr(item(attr(layer, "weight_data"), py::str("bias")), "weight"), data, Op::mul));
            } catch (py::error_already_set &error) {
                if (!error.matches(PyExc_KeyError)) throw;
            }
            set_item(attr(item(attr(model, "layers"), i), "outputs"), py::int_(0), item(attr(item(attr(model, "layers"), j), "outputs"), py::int_(0)));
            set_attr(model, "node_count", inplace(attr(model, "node_count"), py::int_(1), Op::sub));
            set_attr(model, "blob_count", inplace(attr(model, "blob_count"), py::int_(1), Op::sub));
            set_attr(binaryop, "op_type", py::str("ncnnfused"));
        }
    }
}

// Frozen installed optimizer.py:207
void pass_fuse_x_add(const O &model, const O &types) {
    O bias_data = py::none(), bias_term = py::none(), binaryop = py::none(), channels = py::none(), i = py::none(), j = py::none(), k = py::none(), layer = py::none(), memorydata = py::none(), output = py::none();
    (void)types;
    py::ssize_t cn_16 = 0;
    for (py::handle cn_15 : py::reinterpret_borrow<py::iterable>(attr(model, "layers"))) {
        i = py::int_(cn_16++);
        layer = py::reinterpret_borrow<O>(cn_15);
        if (contains(make_tuple({py::str("Convolution"), py::str("ConvolutionDepthWise"), py::str("Deconvolution"), py::str("InnerProduct")}), attr(layer, "op_type"))) {
            output = item(attr(layer, "outputs"), py::int_(0));
            j = i;
            bool cn_17 = false;
            for (py::handle cn_18 : py::reinterpret_borrow<py::iterable>(builtin("range")(binary(i, py::int_(1), Op::add), py::int_(py::len(attr(model, "layers")))))) {
                j = py::reinterpret_borrow<O>(cn_18);
                if (compare(attr(item(attr(model, "layers"), j), "op_type"), py::str("BinaryOp"), Py_NE)) {
                    continue;
                }
                if (compare(attr(item(attr(model, "layers"), j), "num_inputs"), py::int_(2), Py_NE)) {
                    continue;
                }
                if (compare(item(attr(item(attr(model, "layers"), j), "inputs"), py::int_(0)), output, Py_EQ)) {
                    cn_17 = true;
                    break;
                }
            }
            if (!cn_17) {
                j = inplace(j, py::int_(1), Op::add);
            }
            if (compare(j, py::int_(py::len(attr(model, "layers"))), Py_EQ)) {
                continue;
            }
            binaryop = item(attr(model, "layers"), j);
            if ((compare(attr(item(attr(binaryop, "params"), py::int_(0)), "value"), types.attr("BinaryOpTypes").attr("ADD"), Py_NE) || truth(attr(item(attr(binaryop, "params"), py::int_(1)), "value")))) {
                continue;
            }
            k = py::int_(0);
            bool cn_19 = false;
            for (py::handle cn_20 : py::reinterpret_borrow<py::iterable>(builtin("range")(j))) {
                k = py::reinterpret_borrow<O>(cn_20);
                if (compare(attr(item(attr(model, "layers"), k), "op_type"), py::str("MemoryData"), Py_NE)) {
                    continue;
                }
                if (compare(item(attr(item(attr(model, "layers"), k), "outputs"), py::int_(0)), item(attr(binaryop, "inputs"), py::int_(1)), Py_EQ)) {
                    cn_19 = true;
                    break;
                }
            }
            if (!cn_19) {
                k = inplace(k, py::int_(1), Op::add);
            }
            if (compare(k, j, Py_EQ)) {
                continue;
            }
            memorydata = item(attr(model, "layers"), k);
            channels = checked(builtin("int"), attr(item(attr(layer, "params"), py::int_(0)), "value"));
            if ((!((compare(attr(item(attr(memorydata, "params"), py::int_(0)), "value"), channels, Py_EQ) && compare(attr(item(attr(memorydata, "params"), py::int_(1)), "value"), py::int_(0), Py_EQ) && compare(attr(item(attr(memorydata, "params"), py::int_(2)), "value"), py::int_(0), Py_EQ))) || (compare(attr(item(attr(memorydata, "params"), py::int_(0)), "value"), py::int_(1), Py_EQ) && compare(attr(item(attr(memorydata, "params"), py::int_(1)), "value"), py::int_(1), Py_EQ) && compare(attr(item(attr(memorydata, "params"), py::int_(2)), "value"), channels, Py_EQ)))) {
                continue;
            }
            bias_term = (compare(attr(layer, "op_type"), py::str("InnerProduct"), Py_EQ) ? O(py::int_(1)) : O(py::int_(5)));
            bias_data = array_reshape(attr(item(attr(memorydata, "weight_data"), py::str("data")), "weight"), channels);
            if (compare(attr(item(attr(layer, "params"), bias_term), "value"), py::int_(0), Py_EQ)) {
                set_item(attr(layer, "params"), bias_term, py::int_(1));
                attr(layer, "add_weight")(py::str("bias"), bias_data);
            } else {
                set_attr(item(attr(layer, "weight_data"), py::str("bias")), "weight", binary(attr(item(attr(layer, "weight_data"), py::str("bias")), "weight"), bias_data, Op::add));
            }
            set_item(attr(item(attr(model, "layers"), i), "outputs"), py::int_(0), item(attr(item(attr(model, "layers"), j), "outputs"), py::int_(0)));
            set_attr(model, "node_count", inplace(attr(model, "node_count"), py::int_(1), Op::sub));
            set_attr(model, "blob_count", inplace(attr(model, "blob_count"), py::int_(1), Op::sub));
            set_attr(binaryop, "op_type", py::str("ncnnfused"));
        }
    }
}

// Frozen installed optimizer.py:287
void pass_fuse_innerproduct_dropout(const O &model, const O &types) {
    O dropout = py::none(), i = py::none(), j = py::none(), layer = py::none(), output = py::none(), scale = py::none();
    (void)types;
    py::ssize_t cn_22 = 0;
    for (py::handle cn_21 : py::reinterpret_borrow<py::iterable>(attr(model, "layers"))) {
        i = py::int_(cn_22++);
        layer = py::reinterpret_borrow<O>(cn_21);
        if (compare(attr(layer, "op_type"), py::str("InnerProduct"), Py_EQ)) {
            output = item(attr(layer, "outputs"), py::int_(0));
            j = i;
            bool cn_23 = false;
            for (py::handle cn_24 : py::reinterpret_borrow<py::iterable>(builtin("range")(binary(i, py::int_(1), Op::add), py::int_(py::len(attr(model, "layers")))))) {
                j = py::reinterpret_borrow<O>(cn_24);
                if (compare(attr(item(attr(model, "layers"), j), "op_type"), py::str("Dropout"), Py_NE)) {
                    continue;
                }
                if (compare(attr(item(attr(model, "layers"), j), "num_inputs"), py::int_(1), Py_NE)) {
                    continue;
                }
                if (compare(item(attr(item(attr(model, "layers"), j), "inputs"), py::int_(0)), output, Py_EQ)) {
                    cn_23 = true;
                    break;
                }
            }
            if (!cn_23) {
                j = inplace(j, py::int_(1), Op::add);
            }
            if (compare(j, py::int_(py::len(attr(model, "layers"))), Py_EQ)) {
                continue;
            }
            dropout = item(attr(model, "layers"), j);
            scale = checked(builtin("float"), attr(item(attr(dropout, "params"), py::int_(0)), "value"));
            if (compare(scale, py::int_(1), Py_NE)) {
                set_attr(item(attr(layer, "weight_data"), py::str("weight")), "weight", binary(attr(item(attr(layer, "weight_data"), py::str("weight")), "weight"), scale, Op::mul));
                if (compare(attr(item(attr(layer, "params"), py::int_(1)), "value"), py::int_(1), Py_EQ)) {
                    set_attr(item(attr(layer, "weight_data"), py::str("bias")), "weight", binary(attr(item(attr(layer, "weight_data"), py::str("bias")), "weight"), scale, Op::mul));
                }
            }
            set_item(attr(item(attr(model, "layers"), i), "outputs"), py::int_(0), item(attr(item(attr(model, "layers"), j), "outputs"), py::int_(0)));
            set_attr(model, "node_count", inplace(attr(model, "node_count"), py::int_(1), Op::sub));
            set_attr(model, "blob_count", inplace(attr(model, "blob_count"), py::int_(1), Op::sub));
            set_attr(dropout, "op_type", py::str("ncnnfused"));
        }
    }
}

// Frozen installed optimizer.py:326
void pass_fuse_x_activation(const O &model, const O &types) {
    O act = py::none(), i = py::none(), j = py::none(), layer = py::none(), output = py::none();
    (void)types;
    py::ssize_t cn_26 = 0;
    for (py::handle cn_25 : py::reinterpret_borrow<py::iterable>(attr(model, "layers"))) {
        i = py::int_(cn_26++);
        layer = py::reinterpret_borrow<O>(cn_25);
        if (contains(make_tuple({py::str("Convolution"), py::str("Convolution1D"), py::str("ConvolutionDepthWise"), py::str("Deconvolution"), py::str("DeconvolutionDepthWise"), py::str("InnerProduct")}), attr(layer, "op_type"))) {
            output = item(attr(layer, "outputs"), py::int_(0));
            j = i;
            bool cn_27 = false;
            for (py::handle cn_28 : py::reinterpret_borrow<py::iterable>(builtin("range")(binary(i, py::int_(1), Op::add), py::int_(py::len(attr(model, "layers")))))) {
                j = py::reinterpret_borrow<O>(cn_28);
                if (!contains(make_tuple({py::str("ReLU"), py::str("Clip"), py::str("Sigmoid"), py::str("Mish"), py::str("Hardswish")}), attr(item(attr(model, "layers"), j), "op_type"))) {
                    continue;
                }
                if (((compare(attr(item(attr(model, "layers"), j), "op_type"), py::str("Mish"), Py_EQ) && contains(make_tuple({py::str("Deconvolution"), py::str("DeconvolutionDepthWise")}), attr(layer, "op_type"))) || (compare(attr(item(attr(model, "layers"), j), "op_type"), py::str("HardSwish"), Py_EQ) && contains(make_tuple({py::str("Convolution1D"), py::str("Deconvolution"), py::str("DeconvolutionDepthWise")}), attr(layer, "op_type"))))) {
                    continue;
                }
                if (compare(attr(item(attr(model, "layers"), j), "num_inputs"), py::int_(1), Py_NE)) {
                    continue;
                }
                if (compare(item(attr(item(attr(model, "layers"), j), "inputs"), py::int_(0)), output, Py_EQ)) {
                    cn_27 = true;
                    break;
                }
            }
            if (!cn_27) {
                j = inplace(j, py::int_(1), Op::add);
            }
            if (compare(j, py::int_(py::len(attr(model, "layers"))), Py_EQ)) {
                continue;
            }
            act = item(attr(model, "layers"), j);
            if (compare(attr(act, "op_type"), py::str("ReLU"), Py_EQ)) {
                if (compare(attr(item(attr(act, "params"), py::int_(0)), "value"), py::int_(0), Py_EQ)) {
                    set_item(attr(layer, "params"), py::int_(9), py::int_(1));
                } else {
                    set_item(attr(layer, "params"), py::int_(9), py::int_(2));
                    set_item(attr(layer, "params"), py::int_(10), make_list({py::int_(1), checked(builtin("float"), attr(item(attr(act, "params"), py::int_(0)), "value"))}));
                }
            } else {
                if (compare(attr(act, "op_type"), py::str("Clip"), Py_EQ)) {
                    set_item(attr(layer, "params"), py::int_(9), py::int_(3));
                    set_item(attr(layer, "params"), py::int_(10), make_list({py::int_(2), checked(builtin("float"), attr(item(attr(act, "params"), py::int_(0)), "value")), checked(builtin("float"), attr(item(attr(act, "params"), py::int_(1)), "value"))}));
                } else {
                    if (compare(attr(act, "op_type"), py::str("Sigmoid"), Py_EQ)) {
                        set_item(attr(layer, "params"), py::int_(9), py::int_(4));
                    } else {
                        if (compare(attr(act, "op_type"), py::str("Mish"), Py_EQ)) {
                            set_item(attr(layer, "params"), py::int_(9), py::int_(5));
                        } else {
                            if (compare(attr(act, "op_type"), py::str("HardSwish"), Py_EQ)) {
                                set_item(attr(layer, "params"), py::int_(9), py::int_(6));
                                set_item(attr(layer, "params"), py::int_(10), make_list({py::int_(2), checked(builtin("float"), attr(item(attr(act, "params"), py::int_(0)), "value")), checked(builtin("float"), attr(item(attr(act, "params"), py::int_(1)), "value"))}));
                            }
                        }
                    }
                }
            }
            set_item(attr(item(attr(model, "layers"), i), "outputs"), py::int_(0), item(attr(item(attr(model, "layers"), j), "outputs"), py::int_(0)));
            set_attr(model, "node_count", inplace(attr(model, "node_count"), py::int_(1), Op::sub));
            set_attr(model, "blob_count", inplace(attr(model, "blob_count"), py::int_(1), Op::sub));
            set_attr(act, "op_type", py::str("ncnnfused"));
        }
    }
}

// Frozen installed optimizer.py:409
void pass_fuse_memorydata_binaryop(const O &model, const O &types) {
    O binaryop = py::none(), i = py::none(), j = py::none(), j0 = py::none(), j1 = py::none(), k = py::none(), layer = py::none(), memorydata_index = py::none(), op_type = py::none(), output = py::none(), split = py::none(), split_output_index = py::none();
    (void)types;
    py::ssize_t cn_30 = 0;
    for (py::handle cn_29 : py::reinterpret_borrow<py::iterable>(attr(model, "layers"))) {
        i = py::int_(cn_30++);
        layer = py::reinterpret_borrow<O>(cn_29);
        if (compare(attr(layer, "op_type"), py::str("MemoryData"), Py_EQ)) {
            output = item(attr(layer, "outputs"), py::int_(0));
            j = i;
            bool cn_31 = false;
            for (py::handle cn_32 : py::reinterpret_borrow<py::iterable>(builtin("range")(binary(i, py::int_(1), Op::add), py::int_(py::len(attr(model, "layers")))))) {
                j = py::reinterpret_borrow<O>(cn_32);
                if (compare(attr(item(attr(model, "layers"), j), "op_type"), py::str("BinaryOp"), Py_NE)) {
                    continue;
                }
                if (compare(attr(item(attr(model, "layers"), j), "num_inputs"), py::int_(2), Py_NE)) {
                    continue;
                }
                if (contains(make_tuple({item(attr(item(attr(model, "layers"), j), "inputs"), py::int_(0)), item(attr(item(attr(model, "layers"), j), "inputs"), py::int_(1))}), output)) {
                    cn_31 = true;
                    break;
                }
            }
            if (!cn_31) {
                j = inplace(j, py::int_(1), Op::add);
            }
            if (compare(j, py::int_(py::len(attr(model, "layers"))), Py_EQ)) {
                continue;
            }
            binaryop = item(attr(model, "layers"), j);
            if ((compare(attr(item(attr(layer, "params"), py::int_(0)), "value"), py::int_(1), Py_NE) || compare(attr(item(attr(layer, "params"), py::int_(1)), "value"), py::int_(0), Py_NE) || compare(attr(item(attr(layer, "params"), py::int_(2)), "value"), py::int_(0), Py_NE))) {
                continue;
            }
            memorydata_index = py::int_(1);
            if (compare(item(attr(binaryop, "inputs"), py::int_(0)), output, Py_EQ)) {
                op_type = checked(builtin("int"), attr(item(attr(binaryop, "params"), py::int_(0)), "value"));
                if (compare(op_type, types.attr("BinaryOpTypes").attr("ADD"), Py_EQ)) {
                    memorydata_index = py::int_(0);
                } else {
                    if (compare(op_type, types.attr("BinaryOpTypes").attr("SUB"), Py_EQ)) {
                        set_item(attr(binaryop, "params"), py::int_(0), types.attr("BinaryOpTypes").attr("RSUB"));
                        memorydata_index = py::int_(0);
                    } else {
                        if (compare(op_type, types.attr("BinaryOpTypes").attr("DIV"), Py_EQ)) {
                            set_item(attr(binaryop, "params"), py::int_(0), types.attr("BinaryOpTypes").attr("RDIV"));
                            memorydata_index = py::int_(0);
                        } else {
                            continue;
                        }
                    }
                }
            }
            set_item(attr(binaryop, "params"), py::int_(1), py::int_(1));
            set_item(attr(binaryop, "params"), py::int_(2), item(attr(item(attr(layer, "weight_data"), py::str("data")), "weight"), py::int_(0)));
            attr(attr(binaryop, "inputs"), "pop")(memorydata_index);
            set_attr(model, "node_count", inplace(attr(model, "node_count"), py::int_(1), Op::sub));
            set_attr(model, "blob_count", inplace(attr(model, "blob_count"), py::int_(1), Op::sub));
            set_attr(layer, "op_type", py::str("ncnnfused"));
        }
    }
    i = py::int_(0);
    while (contains(builtin("range")(py::int_(py::len(attr(model, "layers")))), i)) {
        if (compare(attr(item(attr(model, "layers"), i), "op_type"), py::str("MemoryData"), Py_EQ)) {
            output = item(attr(item(attr(model, "layers"), i), "outputs"), py::int_(0));
            j0 = i;
            bool cn_33 = false;
            for (py::handle cn_34 : py::reinterpret_borrow<py::iterable>(builtin("range")(binary(i, py::int_(1), Op::add), py::int_(py::len(attr(model, "layers")))))) {
                j0 = py::reinterpret_borrow<O>(cn_34);
                if (compare(attr(item(attr(model, "layers"), j0), "op_type"), py::str("Split"), Py_NE)) {
                    continue;
                }
                if (compare(attr(item(attr(model, "layers"), j0), "num_inputs"), py::int_(1), Py_NE)) {
                    continue;
                }
                if (compare(item(attr(item(attr(model, "layers"), j0), "inputs"), py::int_(0)), output, Py_EQ)) {
                    cn_33 = true;
                    break;
                }
            }
            if (!cn_33) {
                j0 = inplace(j0, py::int_(1), Op::add);
            }
            if (compare(j0, py::int_(py::len(attr(model, "layers"))), Py_EQ)) {
                i = inplace(i, py::int_(1), Op::add);
                continue;
            }
            split_output_index = negative(py::int_(1));
            j1 = i;
            bool cn_35 = false;
            for (py::handle cn_36 : py::reinterpret_borrow<py::iterable>(builtin("range")(binary(i, py::int_(1), Op::add), py::int_(py::len(attr(model, "layers")))))) {
                j1 = py::reinterpret_borrow<O>(cn_36);
                if (compare(attr(item(attr(model, "layers"), j1), "op_type"), py::str("BinaryOp"), Py_NE)) {
                    continue;
                }
                if (compare(attr(item(attr(model, "layers"), j1), "num_inputs"), py::int_(2), Py_NE)) {
                    continue;
                }
                for (py::handle cn_37 : py::reinterpret_borrow<py::iterable>(builtin("range")(attr(item(attr(model, "layers"), j0), "num_outputs")))) {
                    k = py::reinterpret_borrow<O>(cn_37);
                    if ((compare(item(attr(item(attr(model, "layers"), j1), "inputs"), py::int_(0)), item(attr(item(attr(model, "layers"), j0), "outputs"), k), Py_EQ) || compare(item(attr(item(attr(model, "layers"), j1), "inputs"), py::int_(1)), item(attr(item(attr(model, "layers"), j0), "outputs"), k), Py_EQ))) {
                        split_output_index = k;
                        break;
                    }
                }
                if (compare(split_output_index, negative(py::int_(1)), Py_NE)) {
                    cn_35 = true;
                    break;
                }
            }
            if (!cn_35) {
                j1 = inplace(j1, py::int_(1), Op::add);
            }
            if (compare(j1, py::int_(py::len(attr(model, "layers"))), Py_EQ)) {
                i = inplace(i, py::int_(1), Op::add);
                continue;
            }
            split = item(attr(model, "layers"), j0);
            binaryop = item(attr(model, "layers"), j1);
            if ((compare(attr(item(attr(item(attr(model, "layers"), i), "params"), py::int_(0)), "value"), py::int_(1), Py_NE) || compare(attr(item(attr(item(attr(model, "layers"), i), "params"), py::int_(1)), "value"), py::int_(0), Py_NE) || compare(attr(item(attr(item(attr(model, "layers"), i), "params"), py::int_(2)), "value"), py::int_(0), Py_NE))) {
                i = inplace(i, py::int_(1), Op::add);
                continue;
            }
            memorydata_index = py::int_(1);
            if (compare(item(attr(binaryop, "inputs"), py::int_(0)), item(attr(split, "outputs"), split_output_index), Py_EQ)) {
                op_type = checked(builtin("int"), attr(item(attr(binaryop, "params"), py::int_(0)), "value"));
                if (contains(make_tuple({types.attr("BinaryOpTypes").attr("ADD"), types.attr("BinaryOpTypes").attr("MUL"), types.attr("BinaryOpTypes").attr("MAX"), types.attr("BinaryOpTypes").attr("MIN")}), op_type)) {
                    memorydata_index = py::int_(0);
                } else {
                    if (compare(op_type, types.attr("BinaryOpTypes").attr("SUB"), Py_EQ)) {
                        set_item(attr(binaryop, "params"), py::int_(0), types.attr("BinaryOpTypes").attr("RSUB"));
                        memorydata_index = py::int_(0);
                    } else {
                        if (compare(op_type, types.attr("BinaryOpTypes").attr("DIV"), Py_EQ)) {
                            set_item(attr(binaryop, "params"), py::int_(0), types.attr("BinaryOpTypes").attr("RDIV"));
                            memorydata_index = py::int_(0);
                        } else {
                            i = inplace(i, py::int_(1), Op::add);
                            continue;
                        }
                    }
                }
            }
            set_item(attr(binaryop, "params"), py::int_(1), py::int_(1));
            set_item(attr(binaryop, "params"), py::int_(2), item(attr(item(attr(item(attr(model, "layers"), i), "weight_data"), py::str("data")), "weight"), py::int_(0)));
            attr(attr(binaryop, "inputs"), "pop")(memorydata_index);
            set_attr(binaryop, "num_inputs", inplace(attr(binaryop, "num_inputs"), py::int_(1), Op::sub));
            attr(attr(split, "outputs"), "pop")(split_output_index);
            set_attr(split, "num_outputs", inplace(attr(split, "num_outputs"), py::int_(1), Op::sub));
            if (compare(attr(split, "num_outputs"), py::int_(0), Py_EQ)) {
                set_attr(model, "node_count", inplace(attr(model, "node_count"), py::int_(2), Op::sub));
                set_attr(model, "blob_count", inplace(attr(model, "blob_count"), py::int_(2), Op::sub));
                set_attr(split, "op_type", py::str("ncnnfused"));
                set_attr(item(attr(model, "layers"), i), "op_type", py::str("ncnnfused"));
            }
            i = inplace(i, py::int_(1), Op::sub);
        }
        i = inplace(i, py::int_(1), Op::add);
    }
}

// Frozen installed optimizer.py:558
void pass_fuse_binaryop_eltwise(const O &model, const O &types) {
    O binaryop0 = py::none(), binaryop1 = py::none(), eltwise = py::none(), i = py::none(), input0 = py::none(), input1 = py::none(), j0 = py::none(), j1 = py::none(), layer = py::none();
    (void)types;
    py::ssize_t cn_39 = 0;
    for (py::handle cn_38 : py::reinterpret_borrow<py::iterable>(attr(model, "layers"))) {
        i = py::int_(cn_39++);
        layer = py::reinterpret_borrow<O>(cn_38);
        if (compare(attr(layer, "op_type"), py::str("BinaryOp"), Py_EQ)) {
            if (compare(attr(layer, "num_inputs"), py::int_(2), Py_NE)) {
                continue;
            }
            if ((compare(attr(item(attr(layer, "params"), py::int_(0)), "value"), types.attr("BinaryOpTypes").attr("ADD"), Py_NE) || truth(attr(item(attr(layer, "params"), py::int_(1)), "value")))) {
                continue;
            }
            input0 = item(attr(layer, "inputs"), py::int_(0));
            input1 = item(attr(layer, "inputs"), py::int_(1));
            j0 = py::int_(0);
            bool cn_40 = false;
            for (py::handle cn_41 : py::reinterpret_borrow<py::iterable>(builtin("range")(i))) {
                j0 = py::reinterpret_borrow<O>(cn_41);
                if (compare(attr(item(attr(model, "layers"), j0), "op_type"), py::str("BinaryOp"), Py_NE)) {
                    continue;
                }
                if (compare(attr(item(attr(model, "layers"), j0), "num_inputs"), py::int_(1), Py_NE)) {
                    continue;
                }
                if (compare(attr(item(attr(item(attr(model, "layers"), j0), "params"), py::int_(0)), "value"), types.attr("BinaryOpTypes").attr("MUL"), Py_NE)) {
                    continue;
                }
                if (compare(item(attr(item(attr(model, "layers"), j0), "outputs"), py::int_(0)), input0, Py_EQ)) {
                    cn_40 = true;
                    break;
                }
            }
            if (!cn_40) {
                j0 = inplace(j0, py::int_(1), Op::add);
            }
            j1 = py::int_(0);
            bool cn_42 = false;
            for (py::handle cn_43 : py::reinterpret_borrow<py::iterable>(builtin("range")(i))) {
                j1 = py::reinterpret_borrow<O>(cn_43);
                if (compare(attr(item(attr(model, "layers"), j1), "op_type"), py::str("BinaryOp"), Py_NE)) {
                    continue;
                }
                if (compare(attr(item(attr(model, "layers"), j1), "num_inputs"), py::int_(1), Py_NE)) {
                    continue;
                }
                if (compare(attr(item(attr(item(attr(model, "layers"), j1), "params"), py::int_(0)), "value"), types.attr("BinaryOpTypes").attr("MUL"), Py_NE)) {
                    continue;
                }
                if (compare(item(attr(item(attr(model, "layers"), j1), "outputs"), py::int_(0)), input1, Py_EQ)) {
                    cn_42 = true;
                    break;
                }
            }
            if (!cn_42) {
                j1 = inplace(j1, py::int_(1), Op::add);
            }
            if ((compare(j0, i, Py_EQ) && compare(j1, i, Py_EQ))) {
                continue;
            }
            binaryop0 = item(attr(model, "layers"), j0);
            binaryop1 = item(attr(model, "layers"), j1);
            eltwise = types.attr("NcnnLayer")(py::str("Eltwise"), attr(layer, "name"), attr(layer, "num_inputs"), attr(layer, "num_outputs"), attr(layer, "inputs"), attr(layer, "outputs"));
            attr(eltwise, "add_param")(py::int_(0), types.attr("EltwiseOpTypes").attr("SUM"));
            if (!contains(make_tuple({j0, j1}), i)) {
                attr(eltwise, "add_param")(py::int_(1), make_list({py::int_(2), checked(builtin("float"), attr(item(attr(binaryop0, "params"), py::int_(2)), "value")), checked(builtin("float"), attr(item(attr(binaryop1, "params"), py::int_(2)), "value"))}));
                set_item(attr(eltwise, "inputs"), py::int_(0), item(attr(binaryop0, "inputs"), py::int_(0)));
                set_item(attr(eltwise, "inputs"), py::int_(1), item(attr(binaryop1, "inputs"), py::int_(0)));
                set_attr(model, "node_count", inplace(attr(model, "node_count"), py::int_(2), Op::sub));
                set_attr(model, "blob_count", inplace(attr(model, "blob_count"), py::int_(2), Op::sub));
                set_attr(binaryop0, "op_type", py::str("ncnnfused"));
                set_attr(binaryop1, "op_type", py::str("ncnnfused"));
            } else {
                if ((compare(j0, i, Py_NE) && compare(j1, i, Py_EQ))) {
                    attr(eltwise, "add_param")(py::int_(1), make_list({py::int_(2), checked(builtin("float"), attr(item(attr(binaryop0, "params"), py::int_(2)), "value")), py::float_(1.0)}));
                    set_item(attr(eltwise, "inputs"), py::int_(0), item(attr(binaryop0, "inputs"), py::int_(0)));
                    set_attr(model, "node_count", inplace(attr(model, "node_count"), py::int_(1), Op::sub));
                    set_attr(model, "blob_count", inplace(attr(model, "blob_count"), py::int_(1), Op::sub));
                    set_attr(binaryop0, "op_type", py::str("ncnnfused"));
                } else {
                    attr(eltwise, "add_param")(py::int_(1), make_list({py::int_(2), py::float_(1.0), checked(builtin("float"), attr(item(attr(binaryop1, "params"), py::int_(2)), "value"))}));
                    set_item(attr(eltwise, "inputs"), py::int_(1), item(attr(binaryop1, "inputs"), py::int_(0)));
                    set_attr(model, "node_count", inplace(attr(model, "node_count"), py::int_(1), Op::sub));
                    set_attr(model, "blob_count", inplace(attr(model, "blob_count"), py::int_(1), Op::sub));
                    set_attr(binaryop1, "op_type", py::str("ncnnfused"));
                }
            }
            set_item(attr(model, "layers"), i, eltwise);
        }
    }
}

// Frozen installed optimizer.py:648
void pass_eliminate_dropout(const O &model, const O &types) {
    O dropout_input = py::none(), i = py::none(), j = py::none(), layer = py::none();
    (void)types;
    py::ssize_t cn_45 = 0;
    for (py::handle cn_44 : py::reinterpret_borrow<py::iterable>(attr(model, "layers"))) {
        i = py::int_(cn_45++);
        layer = py::reinterpret_borrow<O>(cn_44);
        if (compare(attr(layer, "op_type"), py::str("Dropout"), Py_EQ)) {
            if (compare(attr(item(attr(layer, "params"), py::int_(0)), "value"), py::int_(1), Py_NE)) {
                continue;
            }
            dropout_input = item(attr(layer, "inputs"), py::int_(0));
            j = binary(i, py::int_(1), Op::sub);
            bool cn_46 = false;
            for (py::handle cn_47 : py::reinterpret_borrow<py::iterable>(builtin("range")(binary(i, py::int_(1), Op::sub), negative(py::int_(1)), negative(py::int_(1))))) {
                j = py::reinterpret_borrow<O>(cn_47);
                if (compare(attr(item(attr(model, "layers"), j), "op_type"), py::str("ncnnfused"), Py_EQ)) {
                    continue;
                }
                if (compare(attr(item(attr(model, "layers"), j), "num_outputs"), py::int_(1), Py_NE)) {
                    continue;
                }
                if (compare(item(attr(item(attr(model, "layers"), j), "outputs"), py::int_(0)), dropout_input, Py_EQ)) {
                    cn_46 = true;
                    break;
                }
            }
            if (!cn_46) {
                j = inplace(j, py::int_(1), Op::sub);
            }
            if (compare(j, negative(py::int_(1)), Py_EQ)) {
                continue;
            }
            set_item(attr(item(attr(model, "layers"), j), "outputs"), py::int_(0), item(attr(layer, "outputs"), py::int_(0)));
            set_attr(model, "node_count", inplace(attr(model, "node_count"), py::int_(1), Op::sub));
            set_attr(model, "blob_count", inplace(attr(model, "blob_count"), py::int_(1), Op::sub));
            set_attr(layer, "op_type", py::str("ncnnfused"));
        }
    }
}

// Frozen installed optimizer.py:676
void pass_eliminate_pooling1x1(const O &model, const O &types) {
    O i = py::none(), j = py::none(), k = py::none(), layer = py::none(), pooling_input = py::none(), top_i = py::none();
    (void)types;
    py::ssize_t cn_49 = 0;
    for (py::handle cn_48 : py::reinterpret_borrow<py::iterable>(attr(model, "layers"))) {
        i = py::int_(cn_49++);
        layer = py::reinterpret_borrow<O>(cn_48);
        if (compare(attr(layer, "op_type"), py::str("Pooling"), Py_EQ)) {
            if ((compare(attr(item(attr(layer, "params"), py::int_(3)), "value"), py::int_(0), Py_NE) || compare(attr(item(attr(layer, "params"), py::int_(13)), "value"), py::int_(0), Py_NE) || compare(attr(item(attr(layer, "params"), py::int_(14)), "value"), py::int_(0), Py_NE) || compare(attr(item(attr(layer, "params"), py::int_(15)), "value"), py::int_(0), Py_NE))) {
                continue;
            }
            if ((compare(attr(item(attr(layer, "params"), py::int_(1)), "value"), py::int_(1), Py_NE) || compare(attr(item(attr(layer, "params"), py::int_(11)), "value"), py::int_(1), Py_NE) || compare(attr(item(attr(layer, "params"), py::int_(2)), "value"), py::int_(1), Py_NE) || compare(attr(item(attr(layer, "params"), py::int_(12)), "value"), py::int_(1), Py_NE))) {
                continue;
            }
            if (compare(attr(item(attr(layer, "params"), py::int_(4)), "value"), py::int_(0), Py_NE)) {
                continue;
            }
            pooling_input = item(attr(layer, "inputs"), py::int_(0));
            top_i = negative(py::int_(1));
            j = binary(i, py::int_(1), Op::sub);
            bool cn_50 = false;
            for (py::handle cn_51 : py::reinterpret_borrow<py::iterable>(builtin("range")(binary(i, py::int_(1), Op::sub), negative(py::int_(1)), negative(py::int_(1))))) {
                j = py::reinterpret_borrow<O>(cn_51);
                if (compare(attr(item(attr(model, "layers"), j), "op_type"), py::str("ncnnfused"), Py_EQ)) {
                    continue;
                }
                for (py::handle cn_52 : py::reinterpret_borrow<py::iterable>(builtin("range")(attr(item(attr(model, "layers"), j), "num_outputs")))) {
                    k = py::reinterpret_borrow<O>(cn_52);
                    if (compare(item(attr(item(attr(model, "layers"), j), "outputs"), k), pooling_input, Py_EQ)) {
                        top_i = k;
                        break;
                    }
                }
                if (compare(top_i, negative(py::int_(1)), Py_NE)) {
                    cn_50 = true;
                    break;
                }
            }
            if (!cn_50) {
                j = inplace(j, py::int_(1), Op::sub);
            }
            if (compare(j, negative(py::int_(1)), Py_EQ)) {
                continue;
            }
            set_item(attr(item(attr(model, "layers"), j), "outputs"), top_i, item(attr(layer, "outputs"), py::int_(0)));
            set_attr(model, "node_count", inplace(attr(model, "node_count"), py::int_(1), Op::sub));
            set_attr(model, "blob_count", inplace(attr(model, "blob_count"), py::int_(1), Op::sub));
            set_attr(layer, "op_type", py::str("ncnnfused"));
        }
    }
}

// Frozen installed optimizer.py:723
void pass_eliminate_noop(const O &model, const O &types) {
    O any_k = py::none(), i = py::none(), j = py::none(), k = py::none(), layer = py::none(), link_noop = py::none(), noop_input = py::none();
    (void)types;
    py::ssize_t cn_54 = 0;
    for (py::handle cn_53 : py::reinterpret_borrow<py::iterable>(attr(model, "layers"))) {
        i = py::int_(cn_54++);
        layer = py::reinterpret_borrow<O>(cn_53);
        if (compare(attr(layer, "op_type"), py::str("Noop"), Py_EQ)) {
            if (compare(attr(layer, "num_inputs"), py::int_(0), Py_EQ)) {
                set_attr(layer, "op_type", py::str("ncnnfused"));
                continue;
            }
            noop_input = item(attr(layer, "inputs"), py::int_(0));
            j = binary(i, py::int_(1), Op::sub);
            any_k = negative(py::int_(1));
            bool cn_55 = false;
            for (py::handle cn_56 : py::reinterpret_borrow<py::iterable>(builtin("range")(binary(i, py::int_(1), Op::sub), negative(py::int_(1)), negative(py::int_(1))))) {
                j = py::reinterpret_borrow<O>(cn_56);
                if (compare(attr(item(attr(model, "layers"), j), "op_type"), py::str("ncnnfused"), Py_EQ)) {
                    continue;
                }
                link_noop = py::bool_(false);
                for (py::handle cn_57 : py::reinterpret_borrow<py::iterable>(builtin("range")(attr(item(attr(model, "layers"), j), "num_outputs")))) {
                    k = py::reinterpret_borrow<O>(cn_57);
                    if (compare(item(attr(item(attr(model, "layers"), j), "outputs"), k), noop_input, Py_EQ)) {
                        link_noop = py::bool_(true);
                        any_k = k;
                        break;
                    }
                }
                if (truth(link_noop)) {
                    cn_55 = true;
                    break;
                }
            }
            if (!cn_55) {
                j = inplace(j, py::int_(1), Op::sub);
            }
            if ((compare(j, negative(py::int_(1)), Py_EQ) || compare(any_k, negative(py::int_(1)), Py_EQ))) {
                continue;
            }
            set_item(attr(item(attr(model, "layers"), j), "outputs"), any_k, item(attr(layer, "outputs"), py::int_(0)));
            set_attr(model, "node_count", inplace(attr(model, "node_count"), py::int_(1), Op::sub));
            set_attr(model, "blob_count", inplace(attr(model, "blob_count"), py::int_(1), Op::sub));
            set_attr(layer, "op_type", py::str("ncnnfused"));
        }
    }
}

// Frozen installed optimizer.py:760
void pass_eliminate_split(const O &model, const O &types) {
    O _ = py::none(), blob_input_references = py::none(), i = py::none(), input_name = py::none(), j = py::none(), k = py::none(), layer = py::none(), real_split_output_count = py::none(), real_split_output_index = py::none(), split_input = py::none(), top_i = py::none();
    (void)types;
    blob_input_references = make_list({});
    py::ssize_t cn_59 = 0;
    for (py::handle cn_58 : py::reinterpret_borrow<py::iterable>(attr(model, "layers"))) {
        _ = py::int_(cn_59++);
        layer = py::reinterpret_borrow<O>(cn_58);
        for (py::handle cn_60 : py::reinterpret_borrow<py::iterable>(attr(layer, "inputs"))) {
            input_name = py::reinterpret_borrow<O>(cn_60);
            attr(blob_input_references, "append")(input_name);
        }
    }
    py::ssize_t cn_62 = 0;
    for (py::handle cn_61 : py::reinterpret_borrow<py::iterable>(attr(model, "layers"))) {
        i = py::int_(cn_62++);
        layer = py::reinterpret_borrow<O>(cn_61);
        if (compare(attr(layer, "op_type"), py::str("Split"), Py_EQ)) {
            real_split_output_count = py::int_(0);
            real_split_output_index = negative(py::int_(1));
            for (py::handle cn_63 : py::reinterpret_borrow<py::iterable>(builtin("range")(attr(layer, "num_outputs")))) {
                j = py::reinterpret_borrow<O>(cn_63);
                if (contains(blob_input_references, item(attr(layer, "outputs"), j))) {
                    real_split_output_count = inplace(real_split_output_count, py::int_(1), Op::add);
                    real_split_output_index = j;
                }
            }
            if (compare(real_split_output_count, py::int_(1), Py_GT)) {
                continue;
            }
            split_input = item(attr(layer, "inputs"), py::int_(0));
            top_i = negative(py::int_(1));
            j = binary(i, py::int_(1), Op::sub);
            bool cn_64 = false;
            for (py::handle cn_65 : py::reinterpret_borrow<py::iterable>(builtin("range")(binary(i, py::int_(1), Op::sub), negative(py::int_(1)), negative(py::int_(1))))) {
                j = py::reinterpret_borrow<O>(cn_65);
                if (compare(attr(item(attr(model, "layers"), j), "op_type"), py::str("ncnnfused"), Py_EQ)) {
                    continue;
                }
                for (py::handle cn_66 : py::reinterpret_borrow<py::iterable>(builtin("range")(attr(item(attr(model, "layers"), j), "num_outputs")))) {
                    k = py::reinterpret_borrow<O>(cn_66);
                    if (compare(item(attr(item(attr(model, "layers"), j), "outputs"), k), split_input, Py_EQ)) {
                        top_i = k;
                        break;
                    }
                }
                if (compare(top_i, negative(py::int_(1)), Py_NE)) {
                    cn_64 = true;
                    break;
                }
            }
            if (!cn_64) {
                j = inplace(j, py::int_(1), Op::sub);
            }
            if (compare(j, negative(py::int_(1)), Py_EQ)) {
                continue;
            }
            set_item(attr(item(attr(model, "layers"), j), "outputs"), top_i, item(attr(layer, "outputs"), real_split_output_index));
            set_attr(model, "node_count", inplace(attr(model, "node_count"), py::int_(1), Op::sub));
            set_attr(model, "blob_count", inplace(attr(model, "blob_count"), py::int_(1), Op::sub));
            set_attr(layer, "op_type", py::str("ncnnfused"));
        }
    }
}

// Frozen installed optimizer.py:807
void pass_eliminate_orphaned_memorydata(const O &model, const O &types) {
    O i = py::none(), j = py::none(), k = py::none(), layer = py::none(), memdata_output = py::none(), orphaned = py::none();
    (void)types;
    py::ssize_t cn_68 = 0;
    for (py::handle cn_67 : py::reinterpret_borrow<py::iterable>(attr(model, "layers"))) {
        i = py::int_(cn_68++);
        layer = py::reinterpret_borrow<O>(cn_67);
        if (compare(attr(layer, "op_type"), py::str("MemoryData"), Py_EQ)) {
            memdata_output = item(attr(layer, "outputs"), py::int_(0));
            j = i;
            for (py::handle cn_69 : py::reinterpret_borrow<py::iterable>(builtin("range")(binary(i, py::int_(1), Op::add), py::int_(py::len(attr(model, "layers")))))) {
                j = py::reinterpret_borrow<O>(cn_69);
                if (compare(attr(item(attr(model, "layers"), j), "op_type"), py::str("ncnnfused"), Py_EQ)) {
                    continue;
                }
                orphaned = py::bool_(true);
                for (py::handle cn_70 : py::reinterpret_borrow<py::iterable>(builtin("range")(attr(item(attr(model, "layers"), j), "num_inputs")))) {
                    k = py::reinterpret_borrow<O>(cn_70);
                    if (compare(item(attr(item(attr(model, "layers"), j), "inputs"), k), memdata_output, Py_EQ)) {
                        orphaned = py::bool_(false);
                        break;
                    }
                }
                if (!(truth(orphaned))) {
                    break;
                }
            }
            if (compare(j, py::int_(py::len(attr(model, "layers"))), Py_LT)) {
                continue;
            }
            set_attr(model, "node_count", inplace(attr(model, "node_count"), py::int_(1), Op::sub));
            set_attr(model, "blob_count", inplace(attr(model, "blob_count"), py::int_(1), Op::sub));
            set_attr(layer, "op_type", py::str("ncnnfused"));
        }
    }
}

// Frozen installed optimizer.py:834
void pass_eliminate_reshape_after_global_pooling(const O &model, const O &types) {
    O i = py::none(), j = py::none(), layer = py::none(), pooling_output = py::none(), reshape = py::none();
    (void)types;
    py::ssize_t cn_72 = 0;
    for (py::handle cn_71 : py::reinterpret_borrow<py::iterable>(attr(model, "layers"))) {
        i = py::int_(cn_72++);
        layer = py::reinterpret_borrow<O>(cn_71);
        if (compare(attr(layer, "op_type"), py::str("Pooling"), Py_EQ)) {
            if (compare(attr(item(attr(layer, "params"), py::int_(4)), "value"), py::int_(0), Py_EQ)) {
                continue;
            }
            pooling_output = item(attr(layer, "outputs"), py::int_(0));
            j = i;
            bool cn_73 = false;
            for (py::handle cn_74 : py::reinterpret_borrow<py::iterable>(builtin("range")(binary(i, py::int_(1), Op::add), py::int_(py::len(attr(model, "layers")))))) {
                j = py::reinterpret_borrow<O>(cn_74);
                if (compare(attr(item(attr(model, "layers"), j), "op_type"), py::str("Reshape"), Py_NE)) {
                    continue;
                }
                if (compare(attr(item(attr(model, "layers"), j), "num_inputs"), py::int_(1), Py_NE)) {
                    continue;
                }
                if (compare(item(attr(item(attr(model, "layers"), j), "inputs"), py::int_(0)), pooling_output, Py_EQ)) {
                    cn_73 = true;
                    break;
                }
            }
            if (!cn_73) {
                j = inplace(j, py::int_(1), Op::add);
            }
            if (compare(j, py::int_(py::len(attr(model, "layers"))), Py_EQ)) {
                continue;
            }
            reshape = item(attr(model, "layers"), j);
            if ((compare(attr(item(attr(reshape, "params"), py::int_(1)), "value"), negative(py::int_(233)), Py_NE) || compare(attr(item(attr(reshape, "params"), py::int_(2)), "value"), negative(py::int_(233)), Py_NE) || compare(attr(item(attr(reshape, "params"), py::int_(3)), "value"), py::int_(0), Py_NE))) {
                continue;
            }
            set_item(attr(layer, "outputs"), py::int_(0), item(attr(reshape, "outputs"), py::int_(0)));
            set_attr(model, "node_count", inplace(attr(model, "node_count"), py::int_(1), Op::sub));
            set_attr(model, "blob_count", inplace(attr(model, "blob_count"), py::int_(1), Op::sub));
            set_attr(reshape, "op_type", py::str("ncnnfused"));
        }
    }
}

// Frozen installed optimizer.py:871
void pass_eliminate_flatten_after_global_pooling(const O &model, const O &types) {
    O flatten = py::none(), i = py::none(), j = py::none(), layer = py::none(), pooling_output = py::none();
    (void)types;
    py::ssize_t cn_76 = 0;
    for (py::handle cn_75 : py::reinterpret_borrow<py::iterable>(attr(model, "layers"))) {
        i = py::int_(cn_76++);
        layer = py::reinterpret_borrow<O>(cn_75);
        if (compare(attr(layer, "op_type"), py::str("Pooling"), Py_EQ)) {
            if (compare(attr(item(attr(layer, "params"), py::int_(4)), "value"), py::int_(0), Py_EQ)) {
                continue;
            }
            pooling_output = item(attr(layer, "outputs"), py::int_(0));
            j = i;
            bool cn_77 = false;
            for (py::handle cn_78 : py::reinterpret_borrow<py::iterable>(builtin("range")(binary(i, py::int_(1), Op::add), py::int_(py::len(attr(model, "layers")))))) {
                j = py::reinterpret_borrow<O>(cn_78);
                if (compare(attr(item(attr(model, "layers"), j), "op_type"), py::str("Flatten"), Py_NE)) {
                    continue;
                }
                if (compare(attr(item(attr(model, "layers"), j), "num_inputs"), py::int_(1), Py_NE)) {
                    continue;
                }
                if (compare(item(attr(item(attr(model, "layers"), j), "inputs"), py::int_(0)), pooling_output, Py_EQ)) {
                    cn_77 = true;
                    break;
                }
            }
            if (!cn_77) {
                j = inplace(j, py::int_(1), Op::add);
            }
            if (compare(j, py::int_(py::len(attr(model, "layers"))), Py_EQ)) {
                continue;
            }
            flatten = item(attr(model, "layers"), j);
            set_item(attr(layer, "outputs"), py::int_(0), item(attr(flatten, "outputs"), py::int_(0)));
            set_attr(model, "node_count", inplace(attr(model, "node_count"), py::int_(1), Op::sub));
            set_attr(model, "blob_count", inplace(attr(model, "blob_count"), py::int_(1), Op::sub));
            set_attr(flatten, "op_type", py::str("ncnnfused"));
        }
    }
}

// Frozen installed optimizer.py:901
void pass_eliminate_flatten_after_innerproduct(const O &model, const O &types) {
    O flatten = py::none(), i = py::none(), inprod_output = py::none(), j = py::none(), layer = py::none();
    (void)types;
    py::ssize_t cn_80 = 0;
    for (py::handle cn_79 : py::reinterpret_borrow<py::iterable>(attr(model, "layers"))) {
        i = py::int_(cn_80++);
        layer = py::reinterpret_borrow<O>(cn_79);
        if (compare(attr(layer, "op_type"), py::str("InnerProduct"), Py_EQ)) {
            inprod_output = item(attr(layer, "outputs"), py::int_(0));
            j = i;
            bool cn_81 = false;
            for (py::handle cn_82 : py::reinterpret_borrow<py::iterable>(builtin("range")(binary(i, py::int_(1), Op::add), py::int_(py::len(attr(model, "layers")))))) {
                j = py::reinterpret_borrow<O>(cn_82);
                if (compare(attr(item(attr(model, "layers"), j), "op_type"), py::str("Flatten"), Py_NE)) {
                    continue;
                }
                if (compare(attr(item(attr(model, "layers"), j), "num_inputs"), py::int_(1), Py_NE)) {
                    continue;
                }
                if (compare(item(attr(item(attr(model, "layers"), j), "inputs"), py::int_(0)), inprod_output, Py_EQ)) {
                    cn_81 = true;
                    break;
                }
            }
            if (!cn_81) {
                j = inplace(j, py::int_(1), Op::add);
            }
            if (compare(j, py::int_(py::len(attr(model, "layers"))), Py_EQ)) {
                continue;
            }
            flatten = item(attr(model, "layers"), j);
            set_item(attr(layer, "outputs"), py::int_(0), item(attr(flatten, "outputs"), py::int_(0)));
            set_attr(model, "node_count", inplace(attr(model, "node_count"), py::int_(1), Op::sub));
            set_attr(model, "blob_count", inplace(attr(model, "blob_count"), py::int_(1), Op::sub));
            set_attr(flatten, "op_type", py::str("ncnnfused"));
        }
    }
}

// Frozen installed optimizer.py:928
void pass_eliminate_reshape_before_binaryop(const O &model, const O &types) {
    O binaryop = py::none(), i = py::none(), input_blob_final = py::none(), j = py::none(), layer = py::none(), reshape_output = py::none();
    (void)types;
    py::ssize_t cn_84 = 0;
    for (py::handle cn_83 : py::reinterpret_borrow<py::iterable>(attr(model, "layers"))) {
        i = py::int_(cn_84++);
        layer = py::reinterpret_borrow<O>(cn_83);
        if (compare(attr(layer, "op_type"), py::str("Reshape"), Py_EQ)) {
            if ((compare(attr(item(attr(layer, "params"), py::int_(0)), "value"), py::int_(1), Py_NE) || compare(attr(item(attr(layer, "params"), py::int_(1)), "value"), py::int_(1), Py_NE) || compare(attr(item(attr(layer, "params"), py::int_(3)), "value"), py::int_(1), Py_NE))) {
                continue;
            }
            reshape_output = item(attr(layer, "outputs"), py::int_(0));
            j = i;
            bool cn_85 = false;
            for (py::handle cn_86 : py::reinterpret_borrow<py::iterable>(builtin("range")(binary(i, py::int_(1), Op::add), py::int_(py::len(attr(model, "layers")))))) {
                j = py::reinterpret_borrow<O>(cn_86);
                if (compare(attr(item(attr(model, "layers"), j), "op_type"), py::str("BinaryOp"), Py_NE)) {
                    continue;
                }
                if (compare(attr(item(attr(model, "layers"), j), "num_inputs"), py::int_(2), Py_NE)) {
                    continue;
                }
                if (contains(make_tuple({item(attr(item(attr(model, "layers"), j), "inputs"), py::int_(0)), item(attr(item(attr(model, "layers"), j), "inputs"), py::int_(1))}), reshape_output)) {
                    cn_85 = true;
                    break;
                }
            }
            if (!cn_85) {
                j = inplace(j, py::int_(1), Op::add);
            }
            if (compare(j, py::int_(py::len(attr(model, "layers"))), Py_EQ)) {
                continue;
            }
            binaryop = item(attr(model, "layers"), j);
            input_blob_final = item(attr(layer, "inputs"), py::int_(0));
            if (compare(item(attr(binaryop, "inputs"), py::int_(0)), reshape_output, Py_EQ)) {
                set_item(attr(binaryop, "inputs"), py::int_(0), input_blob_final);
            }
            if (compare(item(attr(binaryop, "inputs"), py::int_(1)), reshape_output, Py_EQ)) {
                set_item(attr(binaryop, "inputs"), py::int_(1), input_blob_final);
            }
            set_attr(model, "node_count", inplace(attr(model, "node_count"), py::int_(1), Op::sub));
            set_attr(model, "blob_count", inplace(attr(model, "blob_count"), py::int_(1), Op::sub));
            set_attr(layer, "op_type", py::str("ncnnfused"));
        }
    }
}

// Frozen installed optimizer.py:969
void pass_replace_reduction_with_global_pooling(const O &model, const O &types) {
    O axes = py::none(), axes2 = py::none(), i = py::none(), j = py::none(), layer = py::none(), pooling = py::none(), reduction1_output = py::none(), reduction2 = py::none();
    (void)types;
    py::ssize_t cn_88 = 0;
    for (py::handle cn_87 : py::reinterpret_borrow<py::iterable>(attr(model, "layers"))) {
        i = py::int_(cn_88++);
        layer = py::reinterpret_borrow<O>(cn_87);
        if (compare(attr(layer, "op_type"), py::str("Reduction"), Py_EQ)) {
            if ((compare(attr(item(attr(layer, "params"), py::int_(0)), "value"), py::int_(3), Py_NE) || compare(attr(item(attr(layer, "params"), py::int_(1)), "value"), py::int_(0), Py_NE) || compare(attr(item(attr(layer, "params"), py::int_(2)), "value"), py::int_(1), Py_NE))) {
                continue;
            }
            axes = checked(builtin("list"), attr(item(attr(layer, "params"), py::int_(3)), "value"));
            if (compare(py::int_(py::len(axes)), py::int_(1), Py_NE)) {
                continue;
            }
            if ((compare(item(axes, py::int_(0)), py::int_(2), Py_NE) && compare(item(axes, py::int_(0)), py::int_(3), Py_NE))) {
                continue;
            }
            reduction1_output = item(attr(layer, "outputs"), py::int_(0));
            j = i;
            bool cn_89 = false;
            for (py::handle cn_90 : py::reinterpret_borrow<py::iterable>(builtin("range")(binary(i, py::int_(1), Op::add), py::int_(py::len(attr(model, "layers")))))) {
                j = py::reinterpret_borrow<O>(cn_90);
                if (compare(attr(item(attr(model, "layers"), j), "op_type"), py::str("Reduction"), Py_NE)) {
                    continue;
                }
                if (compare(attr(item(attr(model, "layers"), j), "num_inputs"), py::int_(1), Py_NE)) {
                    continue;
                }
                if (compare(item(attr(item(attr(model, "layers"), j), "inputs"), py::int_(0)), reduction1_output, Py_EQ)) {
                    cn_89 = true;
                    break;
                }
            }
            if (!cn_89) {
                j = inplace(j, py::int_(1), Op::add);
            }
            if (compare(j, py::int_(py::len(attr(model, "layers"))), Py_EQ)) {
                continue;
            }
            reduction2 = item(attr(model, "layers"), j);
            if ((compare(attr(item(attr(reduction2, "params"), py::int_(0)), "value"), py::int_(3), Py_NE) || compare(attr(item(attr(reduction2, "params"), py::int_(1)), "value"), py::int_(0), Py_NE) || compare(attr(item(attr(reduction2, "params"), py::int_(2)), "value"), py::int_(1), Py_NE))) {
                continue;
            }
            axes2 = checked(builtin("list"), attr(item(attr(layer, "params"), py::int_(3)), "value"));
            if (compare(py::int_(py::len(axes2)), py::int_(1), Py_NE)) {
                continue;
            }
            if (compare(item(axes2, py::int_(0)), py::int_(2), Py_NE)) {
                continue;
            }
            pooling = types.attr("NcnnLayer")(py::str("Pooling"), attr(reduction2, "name"), attr(reduction2, "num_inputs"), attr(reduction2, "num_outputs"), attr(reduction2, "inputs"), attr(reduction2, "outputs"));
            attr(pooling, "add_param")(py::int_(0), py::int_(1));
            attr(pooling, "add_param")(py::int_(4), py::int_(1));
            set_item(attr(model, "layers"), j, pooling);
            set_item(attr(pooling, "inputs"), py::int_(0), item(attr(layer, "inputs"), py::int_(0)));
            set_attr(model, "node_count", inplace(attr(model, "node_count"), py::int_(1), Op::sub));
            set_attr(model, "blob_count", inplace(attr(model, "blob_count"), py::int_(1), Op::sub));
            set_attr(layer, "op_type", py::str("ncnnfused"));
        }
    }
}

// Frozen installed optimizer.py:1035
void pass_replace_prelu_with_leaky_relu(const O &model, const O &types) {
    O i = py::none(), layer = py::none(), relu_layer = py::none();
    (void)types;
    py::ssize_t cn_92 = 0;
    for (py::handle cn_91 : py::reinterpret_borrow<py::iterable>(attr(model, "layers"))) {
        i = py::int_(cn_92++);
        layer = py::reinterpret_borrow<O>(cn_91);
        if (compare(attr(layer, "op_type"), py::str("PReLU"), Py_EQ)) {
            if (compare(attr(item(attr(layer, "params"), py::int_(0)), "value"), py::int_(1), Py_NE)) {
                continue;
            }
            relu_layer = types.attr("NcnnLayer")(py::str("ReLU"), attr(layer, "name"), attr(layer, "num_inputs"), attr(layer, "num_outputs"), attr(layer, "inputs"), attr(layer, "outputs"));
            attr(relu_layer, "add_param")(py::int_(0), checked(builtin("float"), item(attr(item(attr(layer, "weight_data"), py::str("slope")), "weight"), py::int_(0))));
            set_item(attr(model, "layers"), i, relu_layer);
        }
    }
}

// Frozen installed optimizer.py:1055
void pass_replace_convolution_with_innerproduct_after_global_pooling(const O &model, const O &types) {
    O convolution = py::none(), i = py::none(), innerproduct = py::none(), j = py::none(), layer = py::none(), pooling_output = py::none();
    (void)types;
    py::ssize_t cn_94 = 0;
    for (py::handle cn_93 : py::reinterpret_borrow<py::iterable>(attr(model, "layers"))) {
        i = py::int_(cn_94++);
        layer = py::reinterpret_borrow<O>(cn_93);
        if (compare(attr(layer, "op_type"), py::str("Pooling"), Py_EQ)) {
            if (compare(attr(item(attr(layer, "params"), py::int_(4)), "value"), py::int_(0), Py_EQ)) {
                continue;
            }
            pooling_output = item(attr(layer, "outputs"), py::int_(0));
            j = i;
            bool cn_95 = false;
            for (py::handle cn_96 : py::reinterpret_borrow<py::iterable>(builtin("range")(binary(i, py::int_(1), Op::add), py::int_(py::len(attr(model, "layers")))))) {
                j = py::reinterpret_borrow<O>(cn_96);
                if (compare(attr(item(attr(model, "layers"), j), "op_type"), py::str("Convolution"), Py_NE)) {
                    continue;
                }
                if (compare(attr(item(attr(model, "layers"), j), "num_inputs"), py::int_(1), Py_NE)) {
                    continue;
                }
                if (compare(item(attr(item(attr(model, "layers"), j), "inputs"), py::int_(0)), pooling_output, Py_EQ)) {
                    cn_95 = true;
                    break;
                }
            }
            if (!cn_95) {
                j = inplace(j, py::int_(1), Op::add);
            }
            if (compare(j, py::int_(py::len(attr(model, "layers"))), Py_EQ)) {
                continue;
            }
            convolution = item(attr(model, "layers"), j);
            innerproduct = types.attr("NcnnLayer")(py::str("InnerProduct"), attr(convolution, "name"), attr(convolution, "num_inputs"), attr(convolution, "num_outputs"), attr(convolution, "inputs"), attr(convolution, "outputs"));
            attr(innerproduct, "add_param")(py::int_(0), checked(builtin("int"), attr(item(attr(convolution, "params"), py::int_(0)), "value")));
            attr(innerproduct, "add_param")(py::int_(1), checked(builtin("int"), attr(item(attr(convolution, "params"), py::int_(5)), "value")));
            attr(innerproduct, "add_param")(py::int_(2), checked(builtin("int"), attr(item(attr(convolution, "params"), py::int_(6)), "value")));
            attr(innerproduct, "add_param")(py::int_(8), checked(builtin("int"), attr(item(attr(convolution, "params"), py::int_(8)), "value")));
            attr(innerproduct, "add_param")(py::int_(9), checked(builtin("int"), attr(item(attr(convolution, "params"), py::int_(9)), "value")));
            attr(innerproduct, "add_param")(py::int_(10), checked(builtin("list"), attr(item(attr(convolution, "params"), py::int_(10)), "value")));
            attr(innerproduct, "add_weight")(py::str("weight"), attr(item(attr(convolution, "weight_data"), py::str("weight")), "weight"), attr(item(attr(convolution, "weight_data"), py::str("weight")), "quantize_tag"));
            attr(innerproduct, "add_weight")(py::str("bias"), attr(item(attr(convolution, "weight_data"), py::str("bias")), "weight"));
            set_item(attr(model, "layers"), j, innerproduct);
        }
    }
}

// Frozen installed optimizer.py:1116
void pass_replace_convolution_with_innerproduct_after_innerproduct(const O &model, const O &types) {
    O convolution = py::none(), i = py::none(), innerproduct2 = py::none(), inprod_output = py::none(), j = py::none(), layer = py::none(), replaced = py::none();
    (void)types;
    while (truth(py::bool_(true))) {
        replaced = py::bool_(false);
        py::ssize_t cn_98 = 0;
        for (py::handle cn_97 : py::reinterpret_borrow<py::iterable>(attr(model, "layers"))) {
            i = py::int_(cn_98++);
            layer = py::reinterpret_borrow<O>(cn_97);
            if (compare(attr(layer, "op_type"), py::str("InnerProduct"), Py_EQ)) {
                inprod_output = item(attr(layer, "outputs"), py::int_(0));
                j = i;
                bool cn_99 = false;
                for (py::handle cn_100 : py::reinterpret_borrow<py::iterable>(builtin("range")(binary(i, py::int_(1), Op::add), py::int_(py::len(attr(model, "layers")))))) {
                    j = py::reinterpret_borrow<O>(cn_100);
                    if (compare(attr(item(attr(model, "layers"), j), "op_type"), py::str("Convolution"), Py_NE)) {
                        continue;
                    }
                    if (compare(attr(item(attr(model, "layers"), j), "num_inputs"), py::int_(1), Py_NE)) {
                        continue;
                    }
                    if (compare(item(attr(item(attr(model, "layers"), j), "inputs"), py::int_(0)), inprod_output, Py_EQ)) {
                        cn_99 = true;
                        break;
                    }
                }
                if (!cn_99) {
                    j = inplace(j, py::int_(1), Op::add);
                }
                if (compare(j, py::int_(py::len(attr(model, "layers"))), Py_EQ)) {
                    continue;
                }
                convolution = item(attr(model, "layers"), j);
                innerproduct2 = types.attr("NcnnLayer")(py::str("InnerProduct"), attr(convolution, "name"), attr(convolution, "num_inputs"), attr(convolution, "num_outputs"), attr(convolution, "inputs"), attr(convolution, "outputs"));
                attr(innerproduct2, "add_param")(py::int_(0), checked(builtin("int"), attr(item(attr(convolution, "params"), py::int_(0)), "value")));
                attr(innerproduct2, "add_param")(py::int_(1), checked(builtin("int"), attr(item(attr(convolution, "params"), py::int_(5)), "value")));
                attr(innerproduct2, "add_param")(py::int_(2), checked(builtin("int"), attr(item(attr(convolution, "params"), py::int_(6)), "value")));
                attr(innerproduct2, "add_param")(py::int_(8), checked(builtin("int"), attr(item(attr(convolution, "params"), py::int_(8)), "value")));
                attr(innerproduct2, "add_param")(py::int_(9), checked(builtin("int"), attr(item(attr(convolution, "params"), py::int_(9)), "value")));
                attr(innerproduct2, "add_param")(py::int_(10), checked(builtin("list"), attr(item(attr(convolution, "params"), py::int_(10)), "value")));
                attr(innerproduct2, "add_weight")(py::str("weight"), attr(item(attr(convolution, "weight_data"), py::str("weight")), "weight"), attr(item(attr(convolution, "weight_data"), py::str("weight")), "quantize_tag"));
                attr(innerproduct2, "add_weight")(py::str("bias"), attr(item(attr(convolution, "weight_data"), py::str("bias")), "weight"));
                set_item(attr(model, "layers"), j, innerproduct2);
                replaced = py::bool_(true);
            }
        }
        if (!(truth(replaced))) {
            break;
        }
    }
}

// Frozen installed optimizer.py:1182
void pass_optimize(const O &model, const O &types) {
    (void)types;
    pass_fuse_batchnorm_scale(model, types);
    pass_fuse_x_batchnorm(model, types);
    pass_fuse_x_mul(model, types);
    pass_fuse_x_add(model, types);
    pass_fuse_innerproduct_dropout(model, types);
    pass_replace_reduction_with_global_pooling(model, types);
    pass_replace_prelu_with_leaky_relu(model, types);
    pass_fuse_x_activation(model, types);
    pass_fuse_memorydata_binaryop(model, types);
    pass_fuse_binaryop_eltwise(model, types);
    pass_eliminate_dropout(model, types);
    pass_eliminate_pooling1x1(model, types);
    pass_eliminate_noop(model, types);
    pass_eliminate_split(model, types);
    pass_eliminate_flatten_after_global_pooling(model, types);
    pass_eliminate_reshape_after_global_pooling(model, types);
    pass_eliminate_reshape_before_binaryop(model, types);
    pass_replace_convolution_with_innerproduct_after_global_pooling(model, types);
    pass_replace_convolution_with_innerproduct_after_innerproduct(model, types);
    pass_eliminate_flatten_after_innerproduct(model, types);
    pass_eliminate_orphaned_memorydata(model, types);
}
