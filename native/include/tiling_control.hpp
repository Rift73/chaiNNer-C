// Fixed native tile state machines. See generate_tiling_cpp.py and reference_tiling/sources.json.
O tile_installed_auto_split_auto_split(const O &globals, O v_img, O v_upscale, O v_tiler, O v_overlap, O v_progress);
O tile_installed_auto_split_exact_split(const O &globals, O v_img, O v_upscale, O v_starting_tile_size, O v_split_tile_size, O v_overlap, O v_progress);
O tile_installed_auto_split_max_split(const O &globals, O v_img, O v_upscale, O v_starting_tile_size, O v_split_tile_size, O v_overlap, O v_progress);
O tile_installed_exact_split_pad_image(const O &globals, O v_img, O v_min_size);
O tile_installed_exact_split__Segment_length(const O &globals, const O &self);
O tile_installed_exact_split__Segment_padded_length(const O &globals, const O &self);
O tile_installed_exact_split_exact_split_into_segments(const O &globals, O v_length, O v_exact, O v_overlap);
O tile_installed_exact_split_exact_split_into_regions(const O &globals, O v_w, O v_h, O v_exact_w, O v_exact_h, O v_overlap);
O tile_installed_exact_split_exact_split_without_padding(const O &globals, O v_img, O v_exact_size, O v_upscale, O v_overlap, O v_progress);
O tile_installed_exact_split_exact_split(const O &globals, O v_img, O v_exact_size, O v_upscale, O v_overlap, O v_progress);
O tile_installed_tiler_Tiler_allow_smaller_tile_size(const O &globals, const O &self);
O tile_installed_tiler_Tiler_starting_tile_size(const O &globals, const O &self, O v_width, O v_height, O v_channels);
O tile_installed_tiler_Tiler_split(const O &globals, const O &self, O v_tile_size);
O tile_installed_tiler_NoTiling_allow_smaller_tile_size(const O &globals, const O &self);
O tile_installed_tiler_NoTiling_starting_tile_size(const O &globals, const O &self, O v_width, O v_height, O v_channels);
O tile_installed_tiler_NoTiling_split(const O &globals, const O &self, O v_tile_size);
O tile_installed_tiler_MaxTileSize_init(const O &globals, const O &self, O v_tile_size);
O tile_installed_tiler_MaxTileSize_allow_smaller_tile_size(const O &globals, const O &self);
O tile_installed_tiler_MaxTileSize_starting_tile_size(const O &globals, const O &self, O v_width, O v_height, O v_channels);
O tile_installed_tiler_ExactTileSize_init(const O &globals, const O &self, O v_exact_size);
O tile_installed_tiler_ExactTileSize_allow_smaller_tile_size(const O &globals, const O &self);
O tile_installed_tiler_ExactTileSize_starting_tile_size(const O &globals, const O &self, O v_width, O v_height, O v_channels);
O tile_installed_tiler_ExactTileSize_split(const O &globals, const O &self, O v_tile_size);
O tile_installed_auto_split_auto_split(const O &globals, O v_img, O v_upscale, O v_tiler, O v_overlap, O v_progress) {
    O v_c;
    O v_h;
    O v_split;
    O v_w;
    (void)globals;
    (void)v_img;
    (void)v_upscale;
    (void)v_tiler;
    (void)v_overlap;
    (void)v_progress;
    O cn_3 = unpack(([&]() -> O { O cn_1 = item(globals, py::str("get_h_w_c")); O cn_2 = local(v_img, "img"); return cn_1(cn_2); }()), 3);
    v_h = item(cn_3, py::int_(0));
    v_w = item(cn_3, py::int_(1));
    v_c = item(cn_3, py::int_(2));
    v_split = (truth(([&]() -> O { O cn_4 = attr(local(v_tiler, "tiler"), "allow_smaller_tile_size"); return cn_4(); }())) ? O(item(globals, py::str("_max_split"))) : O(item(globals, py::str("_exact_split"))));
    return ([&]() -> O { O cn_9 = local(v_split, "split"); O cn_10 = local(v_img, "img"); O cn_11 = local(v_upscale, "upscale"); O cn_12 = ([&]() -> O { O cn_5 = attr(local(v_tiler, "tiler"), "starting_tile_size"); O cn_6 = local(v_w, "w"); O cn_7 = local(v_h, "h"); O cn_8 = local(v_c, "c"); return cn_5(cn_6, cn_7, cn_8); }()); O cn_13 = attr(local(v_tiler, "tiler"), "split"); O cn_14 = local(v_overlap, "overlap"); O cn_15 = local(v_progress, "progress"); return cn_9(cn_10, py::arg("upscale") = cn_11, py::arg("starting_tile_size") = cn_12, py::arg("split_tile_size") = cn_13, py::arg("overlap") = cn_14, py::arg("progress") = cn_15); }());
}
O tile_installed_auto_split_exact_split(const O &globals, O v_img, O v_upscale, O v_starting_tile_size, O v_split_tile_size, O v_overlap, O v_progress) {
    O v_MAX_ITER;
    O v__;
    O v_c;
    O v_h;
    O v_max_overlap;
    O v_no_split_upscale;
    O v_result;
    O v_w;
    (void)globals;
    (void)v_img;
    (void)v_upscale;
    (void)v_starting_tile_size;
    (void)v_split_tile_size;
    (void)v_overlap;
    (void)v_progress;
    O cn_18 = unpack(([&]() -> O { O cn_16 = item(globals, py::str("get_h_w_c")); O cn_17 = local(v_img, "img"); return cn_16(cn_17); }()), 3);
    v_h = item(cn_18, py::int_(0));
    v_w = item(cn_18, py::int_(1));
    v_c = item(cn_18, py::int_(2));
    ([&]() -> O { O cn_23 = attr(item(globals, py::str("logger")), "debug"); O cn_24 = concat_text({py::str("Exact size split image ("), py::str(local(v_w, "w")), py::str("x"), py::str(local(v_h, "h")), py::str("px @ "), py::str(local(v_c, "c")), py::str(") with exact tile size "), py::str(([&]() -> O { O cn_19 = local(v_starting_tile_size, "starting_tile_size"); O cn_20 = py::int_(0); return item(cn_19, cn_20); }())), py::str("x"), py::str(([&]() -> O { O cn_21 = local(v_starting_tile_size, "starting_tile_size"); O cn_22 = py::int_(1); return item(cn_21, cn_22); }())), py::str("px.")}); return cn_23(cn_24); }());
    v_no_split_upscale = py::cpp_function([=](O v_i, O v_r) mutable -> O {
        O v_result;
        v_result = ([&]() -> O { O cn_25 = local(v_upscale, "upscale"); O cn_26 = local(v_i, "i"); O cn_27 = local(v_r, "r"); return cn_25(cn_26, cn_27); }());
        if (truth(([&]() -> O { O cn_28 = builtin("isinstance"); O cn_29 = local(v_result, "result"); O cn_30 = item(globals, py::str("Split")); return cn_28(cn_29, cn_30); }()))) {
            raise_class(item(globals, py::str("_SplitEx")));
        }
        return local(v_result, "result");
    });
    v_MAX_ITER = py::int_(20);
    for (py::handle cn_31 : py::reinterpret_borrow<py::iterable>(([&]() -> O { O cn_32 = builtin("range"); O cn_33 = local(v_MAX_ITER, "MAX_ITER"); return cn_32(cn_33); }()))) {
        v__ = py::reinterpret_borrow<O>(cn_31);
        try {
            v_max_overlap = ([&]() -> O { O cn_36 = ([&]() -> O { O cn_34 = builtin("min"); py::list cn_35; cn_35.attr("extend")(local(v_starting_tile_size, "starting_tile_size")); return cn_34(*py::tuple(cn_35)); }()); O cn_37 = py::int_(4); return binary(cn_36, cn_37, Op::floordiv); }());
            return ([&]() -> O { O cn_41 = item(globals, py::str("exact_split")); O cn_42 = local(v_img, "img"); O cn_43 = local(v_starting_tile_size, "starting_tile_size"); O cn_44 = local(v_no_split_upscale, "no_split_upscale"); O cn_45 = ([&]() -> O { O cn_38 = builtin("min"); O cn_39 = local(v_max_overlap, "max_overlap"); O cn_40 = local(v_overlap, "overlap"); return cn_38(cn_39, cn_40); }()); O cn_46 = local(v_progress, "progress"); return cn_41(py::arg("img") = cn_42, py::arg("exact_size") = cn_43, py::arg("upscale") = cn_44, py::arg("overlap") = cn_45, py::arg("progress") = cn_46); }());
        } catch (const py::error_already_set &error) {
            if (!error.matches(item(globals, py::str("_SplitEx")).ptr())) throw;
            GraphHandledException handled(error);
            v_starting_tile_size = ([&]() -> O { O cn_47 = local(v_split_tile_size, "split_tile_size"); O cn_48 = local(v_starting_tile_size, "starting_tile_size"); return cn_47(cn_48); }());
        }
    }
    raise(PyExc_ValueError, concat_text({py::str("Aborting after "), py::str(local(v_MAX_ITER, "MAX_ITER")), py::str(" splits. Unable to upscale image.")}));
}
O tile_installed_auto_split_max_split(const O &globals, O v_img, O v_upscale, O v_starting_tile_size, O v_split_tile_size, O v_overlap, O v_progress) {
    O v_c;
    O v_current_scale;
    O v_h;
    O v_img_region;
    O v_max_tile_size;
    O v_new_tile_count_y;
    O v_new_tile_size_y;
    O v_out_channels;
    O v_pad;
    O v_padded_tile;
    O v_prev_row_result;
    O v_restart;
    O v_result;
    O v_row_overlap;
    O v_row_result;
    O v_scale;
    O v_start_y;
    O v_tile;
    O v_tile_count_x;
    O v_tile_count_y;
    O v_tile_size_x;
    O v_tile_size_y;
    O v_tiles_processed;
    O v_total_tiles;
    O v_up_c;
    O v_up_h;
    O v_up_w;
    O v_upscale_result;
    O v_w;
    O v_x;
    O v_y;
    (void)globals;
    (void)v_img;
    (void)v_upscale;
    (void)v_starting_tile_size;
    (void)v_split_tile_size;
    (void)v_overlap;
    (void)v_progress;
    O cn_51 = unpack(([&]() -> O { O cn_49 = item(globals, py::str("get_h_w_c")); O cn_50 = local(v_img, "img"); return cn_49(cn_50); }()), 3);
    v_h = item(cn_51, py::int_(0));
    v_w = item(cn_51, py::int_(1));
    v_c = item(cn_51, py::int_(2));
    v_img_region = ([&]() -> O { O cn_52 = item(globals, py::str("Region")); O cn_53 = py::int_(0); O cn_54 = py::int_(0); O cn_55 = local(v_w, "w"); O cn_56 = local(v_h, "h"); return cn_52(cn_53, cn_54, cn_55, cn_56); }());
    v_max_tile_size = local(v_starting_tile_size, "starting_tile_size");
    ([&]() -> O { O cn_57 = attr(item(globals, py::str("logger")), "debug"); O cn_58 = concat_text({py::str("Auto split image ("), py::str(local(v_w, "w")), py::str("x"), py::str(local(v_h, "h")), py::str("px @ "), py::str(local(v_c, "c")), py::str(") with initial tile size "), py::str(local(v_max_tile_size, "max_tile_size")), py::str(".")}); return cn_57(cn_58); }());
    if ((truth(([&]() -> O { O cn_63 = local(v_w, "w"); O cn_64 = ([&]() -> O { O cn_61 = local(v_max_tile_size, "max_tile_size"); O cn_62 = py::int_(0); return item(cn_61, cn_62); }()); return rich_compare(cn_63, cn_64, Py_LE); }())) && truth(([&]() -> O { O cn_69 = local(v_h, "h"); O cn_70 = ([&]() -> O { O cn_67 = local(v_max_tile_size, "max_tile_size"); O cn_68 = py::int_(1); return item(cn_67, cn_68); }()); return rich_compare(cn_69, cn_70, Py_LE); }())))) {
        v_upscale_result = ([&]() -> O { O cn_71 = local(v_upscale, "upscale"); O cn_72 = local(v_img, "img"); O cn_73 = local(v_img_region, "img_region"); return cn_71(cn_72, cn_73); }());
        if (!(truth(([&]() -> O { O cn_74 = builtin("isinstance"); O cn_75 = local(v_upscale_result, "upscale_result"); O cn_76 = item(globals, py::str("Split")); return cn_74(cn_75, cn_76); }())))) {
            if (truth(([&]() -> O { O cn_77 = local(v_progress, "progress"); O cn_78 = py::none(); return py::bool_(cn_77.ptr() != cn_78.ptr()); }()))) {
                ([&]() -> O { O cn_79 = attr(local(v_progress, "progress"), "set_progress"); O cn_80 = py::float_(1.0); return cn_79(cn_80); }());
            }
            return local(v_upscale_result, "upscale_result");
        }
        v_max_tile_size = ([&]() -> O { O cn_81 = local(v_split_tile_size, "split_tile_size"); O cn_82 = local(v_max_tile_size, "max_tile_size"); return cn_81(cn_82); }());
        ([&]() -> O { O cn_83 = attr(item(globals, py::str("logger")), "warn"); O cn_84 = concat_text({py::str("Unable to upscale the whole image at once. Reduced tile size to "), py::str(local(v_max_tile_size, "max_tile_size")), py::str(".")}); return cn_83(cn_84); }());
    }
    v_start_y = py::int_(0);
    v_result = py::none();
    v_scale = py::int_(0);
    v_out_channels = py::int_(0);
    v_restart = py::bool_(true);
    while (truth(local(v_restart, "restart"))) {
        v_restart = py::bool_(false);
        v_tile_count_x = ([&]() -> O { O cn_89 = attr(item(globals, py::str("math")), "ceil"); O cn_90 = ([&]() -> O { O cn_87 = local(v_w, "w"); O cn_88 = ([&]() -> O { O cn_85 = local(v_max_tile_size, "max_tile_size"); O cn_86 = py::int_(0); return item(cn_85, cn_86); }()); return binary(cn_87, cn_88, Op::div); }()); return cn_89(cn_90); }());
        v_tile_count_y = ([&]() -> O { O cn_95 = attr(item(globals, py::str("math")), "ceil"); O cn_96 = ([&]() -> O { O cn_93 = local(v_h, "h"); O cn_94 = ([&]() -> O { O cn_91 = local(v_max_tile_size, "max_tile_size"); O cn_92 = py::int_(1); return item(cn_91, cn_92); }()); return binary(cn_93, cn_94, Op::div); }()); return cn_95(cn_96); }());
        v_tile_size_x = ([&]() -> O { O cn_99 = attr(item(globals, py::str("math")), "ceil"); O cn_100 = ([&]() -> O { O cn_97 = local(v_w, "w"); O cn_98 = local(v_tile_count_x, "tile_count_x"); return binary(cn_97, cn_98, Op::div); }()); return cn_99(cn_100); }());
        v_tile_size_y = ([&]() -> O { O cn_103 = attr(item(globals, py::str("math")), "ceil"); O cn_104 = ([&]() -> O { O cn_101 = local(v_h, "h"); O cn_102 = local(v_tile_count_y, "tile_count_y"); return binary(cn_101, cn_102, Op::div); }()); return cn_103(cn_104); }());
        v_total_tiles = ([&]() -> O { O cn_105 = local(v_tile_count_x, "tile_count_x"); O cn_106 = local(v_tile_count_y, "tile_count_y"); return binary(cn_105, cn_106, Op::mul); }());
        ([&]() -> O { O cn_107 = attr(item(globals, py::str("logger")), "debug"); O cn_108 = concat_text({py::str("Currently "), py::str(local(v_tile_count_x, "tile_count_x")), py::str("x"), py::str(local(v_tile_count_y, "tile_count_y")), py::str(" tiles each "), py::str(local(v_tile_size_x, "tile_size_x")), py::str("x"), py::str(local(v_tile_size_y, "tile_size_y")), py::str("px.")}); return cn_107(cn_108); }());
        v_prev_row_result = py::none();
        v_tiles_processed = py::int_(0);
        for (py::handle cn_109 : py::reinterpret_borrow<py::iterable>(([&]() -> O { O cn_110 = builtin("range"); O cn_111 = local(v_tile_count_y, "tile_count_y"); return cn_110(cn_111); }()))) {
            v_y = py::reinterpret_borrow<O>(cn_109);
            if (truth(([&]() -> O { O cn_112 = local(v_y, "y"); O cn_113 = local(v_start_y, "start_y"); return rich_compare(cn_112, cn_113, Py_LT); }()))) {
                continue;
            }
            v_row_result = py::none();
            v_row_overlap = py::none();
            for (py::handle cn_114 : py::reinterpret_borrow<py::iterable>(([&]() -> O { O cn_115 = builtin("range"); O cn_116 = local(v_tile_count_x, "tile_count_x"); return cn_115(cn_116); }()))) {
                v_x = py::reinterpret_borrow<O>(cn_114);
                v_tile = ([&]() -> O { O cn_126 = attr(([&]() -> O { O cn_121 = item(globals, py::str("Region")); O cn_122 = ([&]() -> O { O cn_117 = local(v_x, "x"); O cn_118 = local(v_tile_size_x, "tile_size_x"); return binary(cn_117, cn_118, Op::mul); }()); O cn_123 = ([&]() -> O { O cn_119 = local(v_y, "y"); O cn_120 = local(v_tile_size_y, "tile_size_y"); return binary(cn_119, cn_120, Op::mul); }()); O cn_124 = local(v_tile_size_x, "tile_size_x"); O cn_125 = local(v_tile_size_y, "tile_size_y"); return cn_121(cn_122, cn_123, cn_124, cn_125); }()), "intersect"); O cn_127 = local(v_img_region, "img_region"); return cn_126(cn_127); }());
                v_pad = ([&]() -> O { O cn_130 = attr(([&]() -> O { O cn_128 = attr(local(v_img_region, "img_region"), "child_padding"); O cn_129 = local(v_tile, "tile"); return cn_128(cn_129); }()), "min"); O cn_131 = local(v_overlap, "overlap"); return cn_130(cn_131); }());
                v_padded_tile = ([&]() -> O { O cn_132 = attr(local(v_tile, "tile"), "add_padding"); O cn_133 = local(v_pad, "pad"); return cn_132(cn_133); }());
                v_upscale_result = ([&]() -> O { O cn_136 = local(v_upscale, "upscale"); O cn_137 = ([&]() -> O { O cn_134 = attr(local(v_padded_tile, "padded_tile"), "read_from"); O cn_135 = local(v_img, "img"); return cn_134(cn_135); }()); O cn_138 = local(v_padded_tile, "padded_tile"); return cn_136(cn_137, cn_138); }());
                if (truth(([&]() -> O { O cn_139 = builtin("isinstance"); O cn_140 = local(v_upscale_result, "upscale_result"); O cn_141 = item(globals, py::str("Split")); return cn_139(cn_140, cn_141); }()))) {
                    v_max_tile_size = ([&]() -> O { O cn_142 = local(v_split_tile_size, "split_tile_size"); O cn_143 = local(v_max_tile_size, "max_tile_size"); return cn_142(cn_143); }());
                    v_new_tile_count_y = ([&]() -> O { O cn_148 = attr(item(globals, py::str("math")), "ceil"); O cn_149 = ([&]() -> O { O cn_146 = local(v_h, "h"); O cn_147 = ([&]() -> O { O cn_144 = local(v_max_tile_size, "max_tile_size"); O cn_145 = py::int_(1); return item(cn_144, cn_145); }()); return binary(cn_146, cn_147, Op::div); }()); return cn_148(cn_149); }());
                    v_new_tile_size_y = ([&]() -> O { O cn_152 = attr(item(globals, py::str("math")), "ceil"); O cn_153 = ([&]() -> O { O cn_150 = local(v_h, "h"); O cn_151 = local(v_new_tile_count_y, "new_tile_count_y"); return binary(cn_150, cn_151, Op::div); }()); return cn_152(cn_153); }());
                    v_start_y = ([&]() -> O { O cn_156 = ([&]() -> O { O cn_154 = local(v_y, "y"); O cn_155 = local(v_tile_size_y, "tile_size_y"); return binary(cn_154, cn_155, Op::mul); }()); O cn_157 = local(v_new_tile_size_y, "new_tile_size_y"); return binary(cn_156, cn_157, Op::floordiv); }());
                    ([&]() -> O { O cn_158 = attr(item(globals, py::str("logger")), "debug"); O cn_159 = concat_text({py::str("Split occurred. New tile size is "), py::str(local(v_max_tile_size, "max_tile_size")), py::str(". Starting at row "), py::str(local(v_start_y, "start_y")), py::str(".")}); return cn_158(cn_159); }());
                    if (truth(([&]() -> O { O cn_160 = local(v_result, "result"); O cn_161 = py::none(); return py::bool_(cn_160.ptr() != cn_161.ptr()); }()))) {
                        O cn_166 = ([&]() -> O { O cn_164 = ([&]() -> O { O cn_162 = local(v_start_y, "start_y"); O cn_163 = local(v_new_tile_size_y, "new_tile_size_y"); return binary(cn_162, cn_163, Op::mul); }()); O cn_165 = local(v_scale, "scale"); return binary(cn_164, cn_165, Op::mul); }());
                        set_attr(local(v_result, "result"), "offset", cn_166);
                    }
                    v_restart = py::bool_(true);
                    break;
                }
                O cn_169 = unpack(([&]() -> O { O cn_167 = item(globals, py::str("get_h_w_c")); O cn_168 = local(v_upscale_result, "upscale_result"); return cn_167(cn_168); }()), 3);
                v_up_h = item(cn_169, py::int_(0));
                v_up_w = item(cn_169, py::int_(1));
                v_up_c = item(cn_169, py::int_(2));
                v_current_scale = ([&]() -> O { O cn_170 = local(v_up_h, "up_h"); O cn_171 = attr(local(v_padded_tile, "padded_tile"), "height"); return binary(cn_170, cn_171, Op::floordiv); }());
                if (!(truth(([&]() -> O { O cn_172 = local(v_current_scale, "current_scale"); O cn_173 = py::int_(0); return rich_compare(cn_172, cn_173, Py_GT); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
                if (!(truth(([&]() -> O { O cn_178 = ([&]() -> O { O cn_176 = attr(local(v_padded_tile, "padded_tile"), "height"); O cn_177 = local(v_current_scale, "current_scale"); return binary(cn_176, cn_177, Op::mul); }()); O cn_179 = local(v_up_h, "up_h"); return rich_compare(cn_178, cn_179, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
                if (!(truth(([&]() -> O { O cn_184 = ([&]() -> O { O cn_182 = attr(local(v_padded_tile, "padded_tile"), "width"); O cn_183 = local(v_current_scale, "current_scale"); return binary(cn_182, cn_183, Op::mul); }()); O cn_185 = local(v_up_w, "up_w"); return rich_compare(cn_184, cn_185, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
                if (truth(([&]() -> O { O cn_186 = local(v_row_result, "row_result"); O cn_187 = py::none(); return py::bool_(cn_186.ptr() == cn_187.ptr()); }()))) {
                    v_scale = local(v_current_scale, "current_scale");
                    v_out_channels = local(v_up_c, "up_c");
                    v_row_result = ([&]() -> O { O cn_192 = item(globals, py::str("TileBlender")); O cn_193 = ([&]() -> O { O cn_188 = local(v_w, "w"); O cn_189 = local(v_scale, "scale"); return binary(cn_188, cn_189, Op::mul); }()); O cn_194 = ([&]() -> O { O cn_190 = attr(local(v_padded_tile, "padded_tile"), "height"); O cn_191 = local(v_scale, "scale"); return binary(cn_190, cn_191, Op::mul); }()); O cn_195 = local(v_out_channels, "out_channels"); O cn_196 = attr(item(globals, py::str("BlendDirection")), "X"); O cn_197 = item(globals, py::str("half_sin_blend_fn")); O cn_198 = local(v_prev_row_result, "prev_row_result"); return cn_192(py::arg("width") = cn_193, py::arg("height") = cn_194, py::arg("channels") = cn_195, py::arg("direction") = cn_196, py::arg("blend_fn") = cn_197, py::arg("_prev") = cn_198); }());
                    v_prev_row_result = local(v_row_result, "row_result");
                    v_row_overlap = ([&]() -> O { O cn_203 = item(globals, py::str("TileOverlap")); O cn_204 = ([&]() -> O { O cn_199 = attr(local(v_pad, "pad"), "top"); O cn_200 = local(v_scale, "scale"); return binary(cn_199, cn_200, Op::mul); }()); O cn_205 = ([&]() -> O { O cn_201 = attr(local(v_pad, "pad"), "bottom"); O cn_202 = local(v_scale, "scale"); return binary(cn_201, cn_202, Op::mul); }()); return cn_203(cn_204, cn_205); }());
                }
                if (!(truth(([&]() -> O { O cn_206 = local(v_current_scale, "current_scale"); O cn_207 = local(v_scale, "scale"); return rich_compare(cn_206, cn_207, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
                ([&]() -> O { O cn_215 = attr(local(v_row_result, "row_result"), "add_tile"); O cn_216 = local(v_upscale_result, "upscale_result"); O cn_217 = ([&]() -> O { O cn_212 = item(globals, py::str("TileOverlap")); O cn_213 = ([&]() -> O { O cn_208 = attr(local(v_pad, "pad"), "left"); O cn_209 = local(v_scale, "scale"); return binary(cn_208, cn_209, Op::mul); }()); O cn_214 = ([&]() -> O { O cn_210 = attr(local(v_pad, "pad"), "right"); O cn_211 = local(v_scale, "scale"); return binary(cn_210, cn_211, Op::mul); }()); return cn_212(cn_213, cn_214); }()); return cn_215(cn_216, cn_217); }());
                O cn_218 = local(v_tiles_processed, "tiles_processed");
                O cn_219 = py::int_(1);
                O cn_220 = inplace(cn_218, cn_219, Op::add);
                v_tiles_processed = cn_220;
                if (truth(([&]() -> O { O cn_221 = local(v_progress, "progress"); O cn_222 = py::none(); return py::bool_(cn_221.ptr() != cn_222.ptr()); }()))) {
                    ([&]() -> O { O cn_225 = attr(local(v_progress, "progress"), "set_progress"); O cn_226 = ([&]() -> O { O cn_223 = local(v_tiles_processed, "tiles_processed"); O cn_224 = local(v_total_tiles, "total_tiles"); return binary(cn_223, cn_224, Op::div); }()); return cn_225(cn_226); }());
                }
            }
            if (truth(local(v_restart, "restart"))) {
                break;
            }
            if (!(truth(([&]() -> O { O cn_227 = local(v_row_result, "row_result"); O cn_228 = py::none(); return py::bool_(cn_227.ptr() != cn_228.ptr()); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            if (!(truth(([&]() -> O { O cn_229 = local(v_row_overlap, "row_overlap"); O cn_230 = py::none(); return py::bool_(cn_229.ptr() != cn_230.ptr()); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            if (truth(([&]() -> O { O cn_231 = local(v_result, "result"); O cn_232 = py::none(); return py::bool_(cn_231.ptr() == cn_232.ptr()); }()))) {
                v_result = ([&]() -> O { O cn_237 = item(globals, py::str("TileBlender")); O cn_238 = ([&]() -> O { O cn_233 = local(v_w, "w"); O cn_234 = local(v_scale, "scale"); return binary(cn_233, cn_234, Op::mul); }()); O cn_239 = ([&]() -> O { O cn_235 = local(v_h, "h"); O cn_236 = local(v_scale, "scale"); return binary(cn_235, cn_236, Op::mul); }()); O cn_240 = local(v_out_channels, "out_channels"); O cn_241 = attr(item(globals, py::str("BlendDirection")), "Y"); O cn_242 = item(globals, py::str("half_sin_blend_fn")); return cn_237(py::arg("width") = cn_238, py::arg("height") = cn_239, py::arg("channels") = cn_240, py::arg("direction") = cn_241, py::arg("blend_fn") = cn_242); }());
            }
            ([&]() -> O { O cn_244 = attr(local(v_result, "result"), "add_tile"); O cn_245 = ([&]() -> O { O cn_243 = attr(local(v_row_result, "row_result"), "get_result"); return cn_243(); }()); O cn_246 = local(v_row_overlap, "row_overlap"); return cn_244(cn_245, cn_246); }());
        }
    }
    if (!(truth(([&]() -> O { O cn_247 = local(v_result, "result"); O cn_248 = py::none(); return py::bool_(cn_247.ptr() != cn_248.ptr()); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
    return ([&]() -> O { O cn_249 = attr(local(v_result, "result"), "get_result"); return cn_249(); }());
}
O tile_installed_exact_split_pad_image(const O &globals, O v_img, O v_min_size) {
    O v__;
    O v_h;
    O v_min_h;
    O v_min_w;
    O v_padding;
    O v_w;
    O v_x;
    O v_y;
    (void)globals;
    (void)v_img;
    (void)v_min_size;
    O cn_252 = unpack(([&]() -> O { O cn_250 = item(globals, py::str("get_h_w_c")); O cn_251 = local(v_img, "img"); return cn_250(cn_251); }()), 3);
    v_h = item(cn_252, py::int_(0));
    v_w = item(cn_252, py::int_(1));
    v__ = item(cn_252, py::int_(2));
    O cn_253 = unpack(local(v_min_size, "min_size"), 2);
    v_min_w = item(cn_253, py::int_(0));
    v_min_h = item(cn_253, py::int_(1));
    v_x = ([&]() -> O { O cn_259 = ([&]() -> O { O cn_256 = builtin("max"); O cn_257 = py::int_(0); O cn_258 = ([&]() -> O { O cn_254 = local(v_min_w, "min_w"); O cn_255 = local(v_w, "w"); return binary(cn_254, cn_255, Op::sub); }()); return cn_256(cn_257, cn_258); }()); O cn_260 = py::int_(2); return binary(cn_259, cn_260, Op::div); }());
    v_y = ([&]() -> O { O cn_266 = ([&]() -> O { O cn_263 = builtin("max"); O cn_264 = py::int_(0); O cn_265 = ([&]() -> O { O cn_261 = local(v_min_h, "min_h"); O cn_262 = local(v_h, "h"); return binary(cn_261, cn_262, Op::sub); }()); return cn_263(cn_264, cn_265); }()); O cn_267 = py::int_(2); return binary(cn_266, cn_267, Op::div); }());
    v_padding = ([&]() -> O { O cn_276 = item(globals, py::str("Padding")); O cn_277 = ([&]() -> O { O cn_268 = attr(item(globals, py::str("math")), "floor"); O cn_269 = local(v_y, "y"); return cn_268(cn_269); }()); O cn_278 = ([&]() -> O { O cn_270 = attr(item(globals, py::str("math")), "floor"); O cn_271 = local(v_x, "x"); return cn_270(cn_271); }()); O cn_279 = ([&]() -> O { O cn_272 = attr(item(globals, py::str("math")), "ceil"); O cn_273 = local(v_y, "y"); return cn_272(cn_273); }()); O cn_280 = ([&]() -> O { O cn_274 = attr(item(globals, py::str("math")), "ceil"); O cn_275 = local(v_x, "x"); return cn_274(cn_275); }()); return cn_276(cn_277, cn_278, cn_279, cn_280); }());
    return make_tuple({([&]() -> O { O cn_281 = item(globals, py::str("create_border")); O cn_282 = local(v_img, "img"); O cn_283 = attr(item(globals, py::str("BorderType")), "REFLECT_MIRROR"); O cn_284 = local(v_padding, "padding"); return cn_281(cn_282, cn_283, cn_284); }()), local(v_padding, "padding")});
}
O tile_installed_exact_split__Segment_length(const O &globals, const O &self) {
    (void)globals;
    (void)self;
    return ([&]() -> O { O cn_285 = attr(self, "end"); O cn_286 = attr(self, "start"); return binary(cn_285, cn_286, Op::sub); }());
}
O tile_installed_exact_split__Segment_padded_length(const O &globals, const O &self) {
    (void)globals;
    (void)self;
    return ([&]() -> O { O cn_291 = ([&]() -> O { O cn_287 = attr(self, "end"); O cn_288 = attr(self, "end_padding"); return binary(cn_287, cn_288, Op::add); }()); O cn_292 = ([&]() -> O { O cn_289 = attr(self, "start"); O cn_290 = attr(self, "start_padding"); return binary(cn_289, cn_290, Op::sub); }()); return binary(cn_291, cn_292, Op::sub); }());
}
O tile_installed_exact_split_exact_split_into_segments(const O &globals, O v_length, O v_exact, O v_overlap) {
    O v_add;
    O v_end;
    O v_end_padding;
    O v_result;
    O v_start;
    O v_start_padding;
    (void)globals;
    (void)v_length;
    (void)v_exact;
    (void)v_overlap;
    if (truth(([&]() -> O { O cn_293 = local(v_length, "length"); O cn_294 = local(v_exact, "exact"); return rich_compare(cn_293, cn_294, Py_EQ); }()))) {
        return make_list({([&]() -> O { O cn_295 = item(globals, py::str("_Segment")); O cn_296 = py::int_(0); O cn_297 = local(v_exact, "exact"); O cn_298 = py::int_(0); O cn_299 = py::int_(0); return cn_295(cn_296, cn_297, cn_298, cn_299); }())});
    }
    if (!(truth(([&]() -> O { O cn_300 = local(v_length, "length"); O cn_301 = local(v_exact, "exact"); return rich_compare(cn_300, cn_301, Py_GT); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
    if (!(truth(([&]() -> O { O cn_306 = local(v_exact, "exact"); O cn_307 = ([&]() -> O { O cn_304 = local(v_overlap, "overlap"); O cn_305 = py::int_(2); return binary(cn_304, cn_305, Op::mul); }()); return rich_compare(cn_306, cn_307, Py_GT); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
    v_result = make_list({});
    v_add = py::cpp_function([=](O v_s) mutable -> O {
        if (!(truth(([&]() -> O { O cn_308 = attr(local(v_s, "s"), "padded_length"); O cn_309 = local(v_exact, "exact"); return rich_compare(cn_308, cn_309, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
        ([&]() -> O { O cn_310 = attr(local(v_result, "result"), "append"); O cn_311 = local(v_s, "s"); return cn_310(cn_311); }());
        return py::none();
    });
    ([&]() -> O { O cn_319 = local(v_add, "add"); O cn_320 = ([&]() -> O { O cn_314 = item(globals, py::str("_Segment")); O cn_315 = py::int_(0); O cn_316 = ([&]() -> O { O cn_312 = local(v_exact, "exact"); O cn_313 = local(v_overlap, "overlap"); return binary(cn_312, cn_313, Op::sub); }()); O cn_317 = py::int_(0); O cn_318 = local(v_overlap, "overlap"); return cn_314(cn_315, cn_316, cn_317, cn_318); }()); return cn_319(cn_320); }());
    while (truth(([&]() -> O { O cn_325 = attr(([&]() -> O { O cn_323 = local(v_result, "result"); O cn_324 = negative(py::int_(1)); return item(cn_323, cn_324); }()), "end"); O cn_326 = local(v_length, "length"); return rich_compare(cn_325, cn_326, Py_LT); }()))) {
        v_start_padding = local(v_overlap, "overlap");
        v_start = attr(([&]() -> O { O cn_327 = local(v_result, "result"); O cn_328 = negative(py::int_(1)); return item(cn_327, cn_328); }()), "end");
        v_end = ([&]() -> O { O cn_333 = ([&]() -> O { O cn_329 = local(v_start, "start"); O cn_330 = local(v_exact, "exact"); return binary(cn_329, cn_330, Op::add); }()); O cn_334 = ([&]() -> O { O cn_331 = local(v_overlap, "overlap"); O cn_332 = py::int_(2); return binary(cn_331, cn_332, Op::mul); }()); return binary(cn_333, cn_334, Op::sub); }());
        v_end_padding = local(v_overlap, "overlap");
        if (truth(([&]() -> O { O cn_339 = ([&]() -> O { O cn_337 = local(v_end, "end"); O cn_338 = local(v_end_padding, "end_padding"); return binary(cn_337, cn_338, Op::add); }()); O cn_340 = local(v_length, "length"); return rich_compare(cn_339, cn_340, Py_GE); }()))) {
            v_end_padding = py::int_(0);
            v_end = local(v_length, "length");
            v_start_padding = ([&]() -> O { O cn_343 = local(v_exact, "exact"); O cn_344 = ([&]() -> O { O cn_341 = local(v_end, "end"); O cn_342 = local(v_start, "start"); return binary(cn_341, cn_342, Op::sub); }()); return binary(cn_343, cn_344, Op::sub); }());
        }
        ([&]() -> O { O cn_350 = local(v_add, "add"); O cn_351 = ([&]() -> O { O cn_345 = item(globals, py::str("_Segment")); O cn_346 = local(v_start, "start"); O cn_347 = local(v_end, "end"); O cn_348 = local(v_start_padding, "start_padding"); O cn_349 = local(v_end_padding, "end_padding"); return cn_345(cn_346, cn_347, cn_348, cn_349); }()); return cn_350(cn_351); }());
    }
    return local(v_result, "result");
}
O tile_installed_exact_split_exact_split_into_regions(const O &globals, O v_w, O v_h, O v_exact_w, O v_exact_h, O v_overlap) {
    O v_result;
    O v_row;
    O v_x;
    O v_x_segments;
    O v_y;
    O v_y_segments;
    (void)globals;
    (void)v_w;
    (void)v_h;
    (void)v_exact_w;
    (void)v_exact_h;
    (void)v_overlap;
    v_x_segments = ([&]() -> O { O cn_352 = item(globals, py::str("_exact_split_into_segments")); O cn_353 = local(v_w, "w"); O cn_354 = local(v_exact_w, "exact_w"); O cn_355 = local(v_overlap, "overlap"); return cn_352(cn_353, cn_354, cn_355); }());
    v_y_segments = ([&]() -> O { O cn_356 = item(globals, py::str("_exact_split_into_segments")); O cn_357 = local(v_h, "h"); O cn_358 = local(v_exact_h, "exact_h"); O cn_359 = local(v_overlap, "overlap"); return cn_356(cn_357, cn_358, cn_359); }());
    ([&]() -> O { O cn_360 = attr(item(globals, py::str("logger")), "info"); O cn_361 = concat_text({py::str("Image is split into "), py::str(py::int_(py::len(local(v_x_segments, "x_segments")))), py::str("x"), py::str(py::int_(py::len(local(v_y_segments, "y_segments")))), py::str(" tiles each exactly "), py::str(local(v_exact_w, "exact_w")), py::str("x"), py::str(local(v_exact_h, "exact_h")), py::str("px.")}); return cn_360(cn_361); }());
    v_result = make_list({});
    for (py::handle cn_362 : py::reinterpret_borrow<py::iterable>(local(v_y_segments, "y_segments"))) {
        v_y = py::reinterpret_borrow<O>(cn_362);
        v_row = make_list({});
        for (py::handle cn_363 : py::reinterpret_borrow<py::iterable>(local(v_x_segments, "x_segments"))) {
            v_x = py::reinterpret_borrow<O>(cn_363);
            ([&]() -> O { O cn_374 = attr(local(v_row, "row"), "append"); O cn_375 = make_tuple({([&]() -> O { O cn_364 = item(globals, py::str("Region")); O cn_365 = attr(local(v_x, "x"), "start"); O cn_366 = attr(local(v_y, "y"), "start"); O cn_367 = attr(local(v_x, "x"), "length"); O cn_368 = attr(local(v_y, "y"), "length"); return cn_364(cn_365, cn_366, cn_367, cn_368); }()), ([&]() -> O { O cn_369 = item(globals, py::str("Padding")); O cn_370 = attr(local(v_y, "y"), "start_padding"); O cn_371 = attr(local(v_x, "x"), "end_padding"); O cn_372 = attr(local(v_y, "y"), "end_padding"); O cn_373 = attr(local(v_x, "x"), "start_padding"); return cn_369(cn_370, cn_371, cn_372, cn_373); }())}); return cn_374(cn_375); }());
        }
        ([&]() -> O { O cn_376 = attr(local(v_result, "result"), "append"); O cn_377 = local(v_row, "row"); return cn_376(cn_377); }());
    }
    return local(v_result, "result");
}
O tile_installed_exact_split_exact_split_without_padding(const O &globals, O v_img, O v_exact_size, O v_upscale, O v_overlap, O v_progress) {
    O v__;
    O v_current_scale;
    O v_exact_h;
    O v_exact_w;
    O v_h;
    O v_out_channels;
    O v_pad;
    O v_padded_tile;
    O v_regions;
    O v_result;
    O v_row;
    O v_row_overlap;
    O v_row_result;
    O v_scale;
    O v_tile;
    O v_tiles_processed;
    O v_total_tiles;
    O v_up_c;
    O v_up_h;
    O v_up_w;
    O v_upscale_result;
    O v_w;
    (void)globals;
    (void)v_img;
    (void)v_exact_size;
    (void)v_upscale;
    (void)v_overlap;
    (void)v_progress;
    O cn_380 = unpack(([&]() -> O { O cn_378 = item(globals, py::str("get_h_w_c")); O cn_379 = local(v_img, "img"); return cn_378(cn_379); }()), 3);
    v_h = item(cn_380, py::int_(0));
    v_w = item(cn_380, py::int_(1));
    v__ = item(cn_380, py::int_(2));
    O cn_381 = unpack(local(v_exact_size, "exact_size"), 2);
    v_exact_w = item(cn_381, py::int_(0));
    v_exact_h = item(cn_381, py::int_(1));
    if (!((truth(([&]() -> O { O cn_382 = local(v_w, "w"); O cn_383 = local(v_exact_w, "exact_w"); return rich_compare(cn_382, cn_383, Py_GE); }())) && truth(([&]() -> O { O cn_384 = local(v_h, "h"); O cn_385 = local(v_exact_h, "exact_h"); return rich_compare(cn_384, cn_385, Py_GE); }()))))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
    if (truth(([&]() -> O { O cn_386 = make_tuple({local(v_w, "w"), local(v_h, "h")}); O cn_387 = local(v_exact_size, "exact_size"); return rich_compare(cn_386, cn_387, Py_EQ); }()))) {
        if (truth(([&]() -> O { O cn_388 = local(v_progress, "progress"); O cn_389 = py::none(); return py::bool_(cn_388.ptr() != cn_389.ptr()); }()))) {
            ([&]() -> O { O cn_390 = attr(local(v_progress, "progress"), "set_progress"); O cn_391 = py::float_(1.0); return cn_390(cn_391); }());
        }
        return ([&]() -> O { O cn_397 = local(v_upscale, "upscale"); O cn_398 = local(v_img, "img"); O cn_399 = ([&]() -> O { O cn_392 = item(globals, py::str("Region")); O cn_393 = py::int_(0); O cn_394 = py::int_(0); O cn_395 = local(v_w, "w"); O cn_396 = local(v_h, "h"); return cn_392(cn_393, cn_394, cn_395, cn_396); }()); return cn_397(cn_398, cn_399); }());
    }
    v_result = py::none();
    v_scale = py::int_(0);
    v_out_channels = py::int_(0);
    v_regions = ([&]() -> O { O cn_400 = item(globals, py::str("_exact_split_into_regions")); O cn_401 = local(v_w, "w"); O cn_402 = local(v_h, "h"); O cn_403 = local(v_exact_w, "exact_w"); O cn_404 = local(v_exact_h, "exact_h"); O cn_405 = local(v_overlap, "overlap"); return cn_400(cn_401, cn_402, cn_403, cn_404, cn_405); }());
    v_total_tiles = ([&]() -> O { O cn_406 = py::int_(0); for(py::handle cn_407 : py::reinterpret_borrow<py::iterable>(local(v_regions, "regions"))) { O v_row = py::reinterpret_borrow<O>(cn_407); cn_406 = binary(cn_406, py::int_(py::len(local(v_row, "row"))), Op::add); } return cn_406; }());
    v_tiles_processed = py::int_(0);
    for (py::handle cn_408 : py::reinterpret_borrow<py::iterable>(local(v_regions, "regions"))) {
        v_row = py::reinterpret_borrow<O>(cn_408);
        v_row_result = py::none();
        v_row_overlap = py::none();
        for (py::handle cn_409 : py::reinterpret_borrow<py::iterable>(local(v_row, "row"))) {
            O cn_410 = unpack(py::reinterpret_borrow<O>(cn_409), 2);
            v_tile = item(cn_410, py::int_(0));
            v_pad = item(cn_410, py::int_(1));
            v_padded_tile = ([&]() -> O { O cn_411 = attr(local(v_tile, "tile"), "add_padding"); O cn_412 = local(v_pad, "pad"); return cn_411(cn_412); }());
            if (!(truth(([&]() -> O { O cn_413 = attr(local(v_padded_tile, "padded_tile"), "size"); O cn_414 = local(v_exact_size, "exact_size"); return rich_compare(cn_413, cn_414, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            v_upscale_result = ([&]() -> O { O cn_417 = local(v_upscale, "upscale"); O cn_418 = ([&]() -> O { O cn_415 = attr(local(v_padded_tile, "padded_tile"), "read_from"); O cn_416 = local(v_img, "img"); return cn_415(cn_416); }()); O cn_419 = local(v_padded_tile, "padded_tile"); return cn_417(cn_418, cn_419); }());
            O cn_422 = unpack(([&]() -> O { O cn_420 = item(globals, py::str("get_h_w_c")); O cn_421 = local(v_upscale_result, "upscale_result"); return cn_420(cn_421); }()), 3);
            v_up_h = item(cn_422, py::int_(0));
            v_up_w = item(cn_422, py::int_(1));
            v_up_c = item(cn_422, py::int_(2));
            v_current_scale = ([&]() -> O { O cn_423 = local(v_up_h, "up_h"); O cn_424 = local(v_exact_h, "exact_h"); return binary(cn_423, cn_424, Op::floordiv); }());
            if (!(truth(([&]() -> O { O cn_425 = local(v_current_scale, "current_scale"); O cn_426 = py::int_(0); return rich_compare(cn_425, cn_426, Py_GT); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            if (!(truth(([&]() -> O { O cn_431 = ([&]() -> O { O cn_429 = local(v_exact_h, "exact_h"); O cn_430 = local(v_current_scale, "current_scale"); return binary(cn_429, cn_430, Op::mul); }()); O cn_432 = local(v_up_h, "up_h"); return rich_compare(cn_431, cn_432, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            if (!(truth(([&]() -> O { O cn_437 = ([&]() -> O { O cn_435 = local(v_exact_w, "exact_w"); O cn_436 = local(v_current_scale, "current_scale"); return binary(cn_435, cn_436, Op::mul); }()); O cn_438 = local(v_up_w, "up_w"); return rich_compare(cn_437, cn_438, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            if (truth(([&]() -> O { O cn_439 = local(v_row_result, "row_result"); O cn_440 = py::none(); return py::bool_(cn_439.ptr() == cn_440.ptr()); }()))) {
                v_scale = local(v_current_scale, "current_scale");
                v_out_channels = local(v_up_c, "up_c");
                v_row_result = ([&]() -> O { O cn_445 = item(globals, py::str("TileBlender")); O cn_446 = ([&]() -> O { O cn_441 = local(v_w, "w"); O cn_442 = local(v_scale, "scale"); return binary(cn_441, cn_442, Op::mul); }()); O cn_447 = ([&]() -> O { O cn_443 = local(v_exact_h, "exact_h"); O cn_444 = local(v_scale, "scale"); return binary(cn_443, cn_444, Op::mul); }()); O cn_448 = local(v_out_channels, "out_channels"); O cn_449 = attr(item(globals, py::str("BlendDirection")), "X"); O cn_450 = item(globals, py::str("half_sin_blend_fn")); return cn_445(py::arg("width") = cn_446, py::arg("height") = cn_447, py::arg("channels") = cn_448, py::arg("direction") = cn_449, py::arg("blend_fn") = cn_450); }());
                v_row_overlap = ([&]() -> O { O cn_455 = item(globals, py::str("TileOverlap")); O cn_456 = ([&]() -> O { O cn_451 = attr(local(v_pad, "pad"), "top"); O cn_452 = local(v_scale, "scale"); return binary(cn_451, cn_452, Op::mul); }()); O cn_457 = ([&]() -> O { O cn_453 = attr(local(v_pad, "pad"), "bottom"); O cn_454 = local(v_scale, "scale"); return binary(cn_453, cn_454, Op::mul); }()); return cn_455(cn_456, cn_457); }());
            }
            if (!(truth(([&]() -> O { O cn_458 = local(v_current_scale, "current_scale"); O cn_459 = local(v_scale, "scale"); return rich_compare(cn_458, cn_459, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            ([&]() -> O { O cn_467 = attr(local(v_row_result, "row_result"), "add_tile"); O cn_468 = local(v_upscale_result, "upscale_result"); O cn_469 = ([&]() -> O { O cn_464 = item(globals, py::str("TileOverlap")); O cn_465 = ([&]() -> O { O cn_460 = attr(local(v_pad, "pad"), "left"); O cn_461 = local(v_scale, "scale"); return binary(cn_460, cn_461, Op::mul); }()); O cn_466 = ([&]() -> O { O cn_462 = attr(local(v_pad, "pad"), "right"); O cn_463 = local(v_scale, "scale"); return binary(cn_462, cn_463, Op::mul); }()); return cn_464(cn_465, cn_466); }()); return cn_467(cn_468, cn_469); }());
            O cn_470 = local(v_tiles_processed, "tiles_processed");
            O cn_471 = py::int_(1);
            O cn_472 = inplace(cn_470, cn_471, Op::add);
            v_tiles_processed = cn_472;
            if (truth(([&]() -> O { O cn_473 = local(v_progress, "progress"); O cn_474 = py::none(); return py::bool_(cn_473.ptr() != cn_474.ptr()); }()))) {
                ([&]() -> O { O cn_477 = attr(local(v_progress, "progress"), "set_progress"); O cn_478 = ([&]() -> O { O cn_475 = local(v_tiles_processed, "tiles_processed"); O cn_476 = local(v_total_tiles, "total_tiles"); return binary(cn_475, cn_476, Op::div); }()); return cn_477(cn_478); }());
            }
        }
        if (!(truth(([&]() -> O { O cn_479 = local(v_row_result, "row_result"); O cn_480 = py::none(); return py::bool_(cn_479.ptr() != cn_480.ptr()); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
        if (!(truth(([&]() -> O { O cn_481 = local(v_row_overlap, "row_overlap"); O cn_482 = py::none(); return py::bool_(cn_481.ptr() != cn_482.ptr()); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
        if (truth(([&]() -> O { O cn_483 = local(v_result, "result"); O cn_484 = py::none(); return py::bool_(cn_483.ptr() == cn_484.ptr()); }()))) {
            v_result = ([&]() -> O { O cn_489 = item(globals, py::str("TileBlender")); O cn_490 = ([&]() -> O { O cn_485 = local(v_w, "w"); O cn_486 = local(v_scale, "scale"); return binary(cn_485, cn_486, Op::mul); }()); O cn_491 = ([&]() -> O { O cn_487 = local(v_h, "h"); O cn_488 = local(v_scale, "scale"); return binary(cn_487, cn_488, Op::mul); }()); O cn_492 = local(v_out_channels, "out_channels"); O cn_493 = attr(item(globals, py::str("BlendDirection")), "Y"); O cn_494 = item(globals, py::str("half_sin_blend_fn")); return cn_489(py::arg("width") = cn_490, py::arg("height") = cn_491, py::arg("channels") = cn_492, py::arg("direction") = cn_493, py::arg("blend_fn") = cn_494); }());
        }
        ([&]() -> O { O cn_496 = attr(local(v_result, "result"), "add_tile"); O cn_497 = ([&]() -> O { O cn_495 = attr(local(v_row_result, "row_result"), "get_result"); return cn_495(); }()); O cn_498 = local(v_row_overlap, "row_overlap"); return cn_496(cn_497, cn_498); }());
    }
    if (!(truth(([&]() -> O { O cn_499 = local(v_result, "result"); O cn_500 = py::none(); return py::bool_(cn_499.ptr() != cn_500.ptr()); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
    return ([&]() -> O { O cn_501 = attr(local(v_result, "result"), "get_result"); return cn_501(); }());
}
O tile_installed_exact_split_exact_split(const O &globals, O v_img, O v_exact_size, O v_upscale, O v_overlap, O v_progress) {
    O v__;
    O v_base_padding;
    O v_h;
    O v_result;
    O v_scale;
    O v_w;
    (void)globals;
    (void)v_img;
    (void)v_exact_size;
    (void)v_upscale;
    (void)v_overlap;
    (void)v_progress;
    O cn_505 = unpack(([&]() -> O { O cn_502 = item(globals, py::str("_pad_image")); O cn_503 = local(v_img, "img"); O cn_504 = local(v_exact_size, "exact_size"); return cn_502(cn_503, cn_504); }()), 2);
    v_img = item(cn_505, py::int_(0));
    v_base_padding = item(cn_505, py::int_(1));
    O cn_508 = unpack(([&]() -> O { O cn_506 = item(globals, py::str("get_h_w_c")); O cn_507 = local(v_img, "img"); return cn_506(cn_507); }()), 3);
    v_h = item(cn_508, py::int_(0));
    v_w = item(cn_508, py::int_(1));
    v__ = item(cn_508, py::int_(2));
    v_result = ([&]() -> O { O cn_509 = item(globals, py::str("_exact_split_without_padding")); O cn_510 = local(v_img, "img"); O cn_511 = local(v_exact_size, "exact_size"); O cn_512 = local(v_upscale, "upscale"); O cn_513 = local(v_overlap, "overlap"); O cn_514 = local(v_progress, "progress"); return cn_509(cn_510, cn_511, cn_512, cn_513, cn_514); }());
    v_scale = ([&]() -> O { O cn_519 = ([&]() -> O { O cn_517 = ([&]() -> O { O cn_515 = item(globals, py::str("get_h_w_c")); O cn_516 = local(v_result, "result"); return cn_515(cn_516); }()); O cn_518 = py::int_(0); return item(cn_517, cn_518); }()); O cn_520 = local(v_h, "h"); return binary(cn_519, cn_520, Op::floordiv); }());
    if (truth(attr(local(v_base_padding, "base_padding"), "empty"))) {
        return local(v_result, "result");
    }
    return ([&]() -> O { O cn_530 = attr(([&]() -> O { O cn_528 = attr(([&]() -> O { O cn_526 = attr(([&]() -> O { O cn_521 = item(globals, py::str("Region")); O cn_522 = py::int_(0); O cn_523 = py::int_(0); O cn_524 = local(v_w, "w"); O cn_525 = local(v_h, "h"); return cn_521(cn_522, cn_523, cn_524, cn_525); }()), "remove_padding"); O cn_527 = local(v_base_padding, "base_padding"); return cn_526(cn_527); }()), "scale"); O cn_529 = local(v_scale, "scale"); return cn_528(cn_529); }()), "read_from"); O cn_531 = local(v_result, "result"); return cn_530(cn_531); }());
}
O tile_installed_tiler_Tiler_allow_smaller_tile_size(const O &globals, const O &self) {
    (void)globals;
    (void)self;
    return py::none();
}
O tile_installed_tiler_Tiler_starting_tile_size(const O &globals, const O &self, O v_width, O v_height, O v_channels) {
    (void)globals;
    (void)self;
    (void)v_width;
    (void)v_height;
    (void)v_channels;
    return py::none();
}
O tile_installed_tiler_Tiler_split(const O &globals, const O &self, O v_tile_size) {
    O v_h;
    O v_w;
    (void)globals;
    (void)self;
    (void)v_tile_size;
    O cn_532 = unpack(local(v_tile_size, "tile_size"), 2);
    v_w = item(cn_532, py::int_(0));
    v_h = item(cn_532, py::int_(1));
    if (!((truth(([&]() -> O { O cn_533 = local(v_w, "w"); O cn_534 = py::int_(16); return rich_compare(cn_533, cn_534, Py_GE); }())) && truth(([&]() -> O { O cn_535 = local(v_h, "h"); O cn_536 = py::int_(16); return rich_compare(cn_535, cn_536, Py_GE); }()))))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
    return make_tuple({([&]() -> O { O cn_539 = builtin("max"); O cn_540 = py::int_(16); O cn_541 = ([&]() -> O { O cn_537 = local(v_w, "w"); O cn_538 = py::int_(2); return binary(cn_537, cn_538, Op::floordiv); }()); return cn_539(cn_540, cn_541); }()), ([&]() -> O { O cn_544 = builtin("max"); O cn_545 = py::int_(16); O cn_546 = ([&]() -> O { O cn_542 = local(v_h, "h"); O cn_543 = py::int_(2); return binary(cn_542, cn_543, Op::floordiv); }()); return cn_544(cn_545, cn_546); }())});
}
O tile_installed_tiler_NoTiling_allow_smaller_tile_size(const O &globals, const O &self) {
    (void)globals;
    (void)self;
    return py::bool_(true);
}
O tile_installed_tiler_NoTiling_starting_tile_size(const O &globals, const O &self, O v_width, O v_height, O v_channels) {
    O v_size;
    (void)globals;
    (void)self;
    (void)v_width;
    (void)v_height;
    (void)v_channels;
    v_size = ([&]() -> O { O cn_547 = builtin("max"); O cn_548 = local(v_width, "width"); O cn_549 = local(v_height, "height"); return cn_547(cn_548, cn_549); }());
    return make_tuple({local(v_size, "size"), local(v_size, "size")});
}
O tile_installed_tiler_NoTiling_split(const O &globals, const O &self, O v_tile_size) {
    (void)globals;
    (void)self;
    (void)v_tile_size;
    raise(PyExc_ValueError, py::str("Image cannot be upscale with No Tiling mode."));
}
O tile_installed_tiler_MaxTileSize_init(const O &globals, const O &self, O v_tile_size) {
    (void)globals;
    (void)self;
    (void)v_tile_size;
    O cn_550 = local(v_tile_size, "tile_size");
    set_attr(self, "tile_size", cn_550);
    return py::none();
}
O tile_installed_tiler_MaxTileSize_allow_smaller_tile_size(const O &globals, const O &self) {
    (void)globals;
    (void)self;
    return py::bool_(true);
}
O tile_installed_tiler_MaxTileSize_starting_tile_size(const O &globals, const O &self, O v_width, O v_height, O v_channels) {
    O v_max_tile_size;
    O v_size;
    (void)globals;
    (void)self;
    (void)v_width;
    (void)v_height;
    (void)v_channels;
    v_max_tile_size = ([&]() -> O { O cn_555 = builtin("max"); O cn_556 = ([&]() -> O { O cn_551 = local(v_width, "width"); O cn_552 = py::int_(10); return binary(cn_551, cn_552, Op::add); }()); O cn_557 = ([&]() -> O { O cn_553 = local(v_height, "height"); O cn_554 = py::int_(10); return binary(cn_553, cn_554, Op::add); }()); return cn_555(cn_556, cn_557); }());
    v_size = ([&]() -> O { O cn_558 = builtin("min"); O cn_559 = attr(self, "tile_size"); O cn_560 = local(v_max_tile_size, "max_tile_size"); return cn_558(cn_559, cn_560); }());
    return make_tuple({local(v_size, "size"), local(v_size, "size")});
}
O tile_installed_tiler_ExactTileSize_init(const O &globals, const O &self, O v_exact_size) {
    (void)globals;
    (void)self;
    (void)v_exact_size;
    O cn_561 = local(v_exact_size, "exact_size");
    set_attr(self, "exact_size", cn_561);
    return py::none();
}
O tile_installed_tiler_ExactTileSize_allow_smaller_tile_size(const O &globals, const O &self) {
    (void)globals;
    (void)self;
    return py::bool_(false);
}
O tile_installed_tiler_ExactTileSize_starting_tile_size(const O &globals, const O &self, O v_width, O v_height, O v_channels) {
    (void)globals;
    (void)self;
    (void)v_width;
    (void)v_height;
    (void)v_channels;
    return attr(self, "exact_size");
}
O tile_installed_tiler_ExactTileSize_split(const O &globals, const O &self, O v_tile_size) {
    (void)globals;
    (void)self;
    (void)v_tile_size;
    raise(PyExc_ValueError, concat_text({py::str("Splits are not supported for exact size ("), py::str(([&]() -> O { O cn_562 = attr(self, "exact_size"); O cn_563 = py::int_(0); return item(cn_562, cn_563); }())), py::str("x"), py::str(([&]() -> O { O cn_564 = attr(self, "exact_size"); O cn_565 = py::int_(1); return item(cn_564, cn_565); }())), py::str("px) splitting. This typically means that your machine does not have enough VRAM to run the current model.")}));
}
void bind_tiling(py::module_ &module) {
    module.def("tiling_installed_auto_split_auto_split", &tile_installed_auto_split_auto_split);
    module.def("tiling_installed_auto_split_exact_split", &tile_installed_auto_split_exact_split);
    module.def("tiling_installed_auto_split_max_split", &tile_installed_auto_split_max_split);
    module.def("tiling_installed_exact_split_pad_image", &tile_installed_exact_split_pad_image);
    module.def("tiling_installed_exact_split__Segment_length", &tile_installed_exact_split__Segment_length);
    module.def("tiling_installed_exact_split__Segment_padded_length", &tile_installed_exact_split__Segment_padded_length);
    module.def("tiling_installed_exact_split_exact_split_into_segments", &tile_installed_exact_split_exact_split_into_segments);
    module.def("tiling_installed_exact_split_exact_split_into_regions", &tile_installed_exact_split_exact_split_into_regions);
    module.def("tiling_installed_exact_split_exact_split_without_padding", &tile_installed_exact_split_exact_split_without_padding);
    module.def("tiling_installed_exact_split_exact_split", &tile_installed_exact_split_exact_split);
    module.def("tiling_installed_tiler_Tiler_allow_smaller_tile_size", &tile_installed_tiler_Tiler_allow_smaller_tile_size);
    module.def("tiling_installed_tiler_Tiler_starting_tile_size", &tile_installed_tiler_Tiler_starting_tile_size);
    module.def("tiling_installed_tiler_Tiler_split", &tile_installed_tiler_Tiler_split);
    module.def("tiling_installed_tiler_NoTiling_allow_smaller_tile_size", &tile_installed_tiler_NoTiling_allow_smaller_tile_size);
    module.def("tiling_installed_tiler_NoTiling_starting_tile_size", &tile_installed_tiler_NoTiling_starting_tile_size);
    module.def("tiling_installed_tiler_NoTiling_split", &tile_installed_tiler_NoTiling_split);
    module.def("tiling_installed_tiler_MaxTileSize_init", &tile_installed_tiler_MaxTileSize_init);
    module.def("tiling_installed_tiler_MaxTileSize_allow_smaller_tile_size", &tile_installed_tiler_MaxTileSize_allow_smaller_tile_size);
    module.def("tiling_installed_tiler_MaxTileSize_starting_tile_size", &tile_installed_tiler_MaxTileSize_starting_tile_size);
    module.def("tiling_installed_tiler_ExactTileSize_init", &tile_installed_tiler_ExactTileSize_init);
    module.def("tiling_installed_tiler_ExactTileSize_allow_smaller_tile_size", &tile_installed_tiler_ExactTileSize_allow_smaller_tile_size);
    module.def("tiling_installed_tiler_ExactTileSize_starting_tile_size", &tile_installed_tiler_ExactTileSize_starting_tile_size);
    module.def("tiling_installed_tiler_ExactTileSize_split", &tile_installed_tiler_ExactTileSize_split);
}
