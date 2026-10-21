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
O tile_source_auto_split_auto_split(const O &globals, O v_img, O v_upscale, O v_tiler, O v_overlap, O v_progress);
O tile_source_auto_split_exact_split(const O &globals, O v_img, O v_upscale, O v_starting_tile_size, O v_split_tile_size, O v_overlap, O v_progress);
O tile_source_auto_split_max_split(const O &globals, O v_img, O v_upscale, O v_starting_tile_size, O v_split_tile_size, O v_overlap, O v_progress);
O tile_source_exact_split_pad_image(const O &globals, O v_img, O v_min_size);
O tile_source_exact_split__Segment_length(const O &globals, const O &self);
O tile_source_exact_split__Segment_padded_length(const O &globals, const O &self);
O tile_source_exact_split_exact_split_into_segments(const O &globals, O v_length, O v_exact, O v_overlap);
O tile_source_exact_split_exact_split_into_regions(const O &globals, O v_w, O v_h, O v_exact_w, O v_exact_h, O v_overlap);
O tile_source_exact_split_exact_split_without_padding(const O &globals, O v_img, O v_exact_size, O v_upscale, O v_overlap, O v_progress);
O tile_source_exact_split_exact_split(const O &globals, O v_img, O v_exact_size, O v_upscale, O v_overlap, O v_progress);
O tile_source_tiler_Tiler_allow_smaller_tile_size(const O &globals, const O &self);
O tile_source_tiler_Tiler_starting_tile_size(const O &globals, const O &self, O v_width, O v_height, O v_channels);
O tile_source_tiler_Tiler_split(const O &globals, const O &self, O v_tile_size);
O tile_source_tiler_NoTiling_allow_smaller_tile_size(const O &globals, const O &self);
O tile_source_tiler_NoTiling_starting_tile_size(const O &globals, const O &self, O v_width, O v_height, O v_channels);
O tile_source_tiler_NoTiling_split(const O &globals, const O &self, O v_tile_size);
O tile_source_tiler_MaxTileSize_init(const O &globals, const O &self, O v_tile_size);
O tile_source_tiler_MaxTileSize_allow_smaller_tile_size(const O &globals, const O &self);
O tile_source_tiler_MaxTileSize_starting_tile_size(const O &globals, const O &self, O v_width, O v_height, O v_channels);
O tile_source_tiler_ExactTileSize_init(const O &globals, const O &self, O v_exact_size);
O tile_source_tiler_ExactTileSize_allow_smaller_tile_size(const O &globals, const O &self);
O tile_source_tiler_ExactTileSize_starting_tile_size(const O &globals, const O &self, O v_width, O v_height, O v_channels);
O tile_source_tiler_ExactTileSize_split(const O &globals, const O &self, O v_tile_size);
O tile_source_tiler_BoundedTileSize_init(const O &globals, const O &self, O v_tile_size, O v_min_size, O v_max_size);
O tile_source_tiler_BoundedTileSize_allow_smaller_tile_size(const O &globals, const O &self);
O tile_source_tiler_BoundedTileSize_starting_tile_size(const O &globals, const O &self, O v_width, O v_height, O v_channels);
O tile_source_tiler_BoundedTileSize_split(const O &globals, const O &self, O v_tile_size);
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
O tile_source_auto_split_auto_split(const O &globals, O v_img, O v_upscale, O v_tiler, O v_overlap, O v_progress) {
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
    O cn_568 = unpack(([&]() -> O { O cn_566 = item(globals, py::str("get_h_w_c")); O cn_567 = local(v_img, "img"); return cn_566(cn_567); }()), 3);
    v_h = item(cn_568, py::int_(0));
    v_w = item(cn_568, py::int_(1));
    v_c = item(cn_568, py::int_(2));
    v_split = (truth(([&]() -> O { O cn_569 = attr(local(v_tiler, "tiler"), "allow_smaller_tile_size"); return cn_569(); }())) ? O(item(globals, py::str("_max_split"))) : O(item(globals, py::str("_exact_split"))));
    return ([&]() -> O { O cn_574 = local(v_split, "split"); O cn_575 = local(v_img, "img"); O cn_576 = local(v_upscale, "upscale"); O cn_577 = ([&]() -> O { O cn_570 = attr(local(v_tiler, "tiler"), "starting_tile_size"); O cn_571 = local(v_w, "w"); O cn_572 = local(v_h, "h"); O cn_573 = local(v_c, "c"); return cn_570(cn_571, cn_572, cn_573); }()); O cn_578 = attr(local(v_tiler, "tiler"), "split"); O cn_579 = local(v_overlap, "overlap"); O cn_580 = local(v_progress, "progress"); return cn_574(cn_575, py::arg("upscale") = cn_576, py::arg("starting_tile_size") = cn_577, py::arg("split_tile_size") = cn_578, py::arg("overlap") = cn_579, py::arg("progress") = cn_580); }());
}
O tile_source_auto_split_exact_split(const O &globals, O v_img, O v_upscale, O v_starting_tile_size, O v_split_tile_size, O v_overlap, O v_progress) {
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
    O cn_583 = unpack(([&]() -> O { O cn_581 = item(globals, py::str("get_h_w_c")); O cn_582 = local(v_img, "img"); return cn_581(cn_582); }()), 3);
    v_h = item(cn_583, py::int_(0));
    v_w = item(cn_583, py::int_(1));
    v_c = item(cn_583, py::int_(2));
    ([&]() -> O { O cn_588 = attr(item(globals, py::str("logger")), "debug"); O cn_589 = py::str("Exact size split image (%dx%dpx @ %d) with exact tile size %dx%dpx."); O cn_590 = local(v_w, "w"); O cn_591 = local(v_h, "h"); O cn_592 = local(v_c, "c"); O cn_593 = ([&]() -> O { O cn_584 = local(v_starting_tile_size, "starting_tile_size"); O cn_585 = py::int_(0); return item(cn_584, cn_585); }()); O cn_594 = ([&]() -> O { O cn_586 = local(v_starting_tile_size, "starting_tile_size"); O cn_587 = py::int_(1); return item(cn_586, cn_587); }()); return cn_588(cn_589, cn_590, cn_591, cn_592, cn_593, cn_594); }());
    v_no_split_upscale = py::cpp_function([=](O v_i, O v_r) mutable -> O {
        O v_result;
        v_result = ([&]() -> O { O cn_595 = local(v_upscale, "upscale"); O cn_596 = local(v_i, "i"); O cn_597 = local(v_r, "r"); return cn_595(cn_596, cn_597); }());
        if (truth(([&]() -> O { O cn_598 = builtin("isinstance"); O cn_599 = local(v_result, "result"); O cn_600 = item(globals, py::str("Split")); return cn_598(cn_599, cn_600); }()))) {
            raise_class(item(globals, py::str("_SplitEx")));
        }
        return local(v_result, "result");
    });
    v_MAX_ITER = py::int_(20);
    for (py::handle cn_601 : py::reinterpret_borrow<py::iterable>(([&]() -> O { O cn_602 = builtin("range"); O cn_603 = local(v_MAX_ITER, "MAX_ITER"); return cn_602(cn_603); }()))) {
        v__ = py::reinterpret_borrow<O>(cn_601);
        try {
            v_max_overlap = ([&]() -> O { O cn_606 = ([&]() -> O { O cn_604 = builtin("min"); py::list cn_605; cn_605.attr("extend")(local(v_starting_tile_size, "starting_tile_size")); return cn_604(*py::tuple(cn_605)); }()); O cn_607 = py::int_(4); return binary(cn_606, cn_607, Op::floordiv); }());
            return ([&]() -> O { O cn_611 = item(globals, py::str("exact_split")); O cn_612 = local(v_img, "img"); O cn_613 = local(v_starting_tile_size, "starting_tile_size"); O cn_614 = local(v_no_split_upscale, "no_split_upscale"); O cn_615 = ([&]() -> O { O cn_608 = builtin("min"); O cn_609 = local(v_max_overlap, "max_overlap"); O cn_610 = local(v_overlap, "overlap"); return cn_608(cn_609, cn_610); }()); O cn_616 = local(v_progress, "progress"); return cn_611(py::arg("img") = cn_612, py::arg("exact_size") = cn_613, py::arg("upscale") = cn_614, py::arg("overlap") = cn_615, py::arg("progress") = cn_616); }());
        } catch (const py::error_already_set &error) {
            if (!error.matches(item(globals, py::str("_SplitEx")).ptr())) throw;
            GraphHandledException handled(error);
            v_starting_tile_size = ([&]() -> O { O cn_617 = local(v_split_tile_size, "split_tile_size"); O cn_618 = local(v_starting_tile_size, "starting_tile_size"); return cn_617(cn_618); }());
        }
    }
    raise(PyExc_ValueError, concat_text({py::str("Aborting after "), py::str(local(v_MAX_ITER, "MAX_ITER")), py::str(" splits. Unable to upscale image.")}));
}
O tile_source_auto_split_max_split(const O &globals, O v_img, O v_upscale, O v_starting_tile_size, O v_split_tile_size, O v_overlap, O v_progress) {
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
    O cn_621 = unpack(([&]() -> O { O cn_619 = item(globals, py::str("get_h_w_c")); O cn_620 = local(v_img, "img"); return cn_619(cn_620); }()), 3);
    v_h = item(cn_621, py::int_(0));
    v_w = item(cn_621, py::int_(1));
    v_c = item(cn_621, py::int_(2));
    v_img_region = ([&]() -> O { O cn_622 = item(globals, py::str("Region")); O cn_623 = py::int_(0); O cn_624 = py::int_(0); O cn_625 = local(v_w, "w"); O cn_626 = local(v_h, "h"); return cn_622(cn_623, cn_624, cn_625, cn_626); }());
    v_max_tile_size = local(v_starting_tile_size, "starting_tile_size");
    ([&]() -> O { O cn_627 = attr(item(globals, py::str("logger")), "debug"); O cn_628 = py::str("Auto split image (%dx%dpx @ %d) with initial tile size %s."); O cn_629 = local(v_w, "w"); O cn_630 = local(v_h, "h"); O cn_631 = local(v_c, "c"); O cn_632 = local(v_max_tile_size, "max_tile_size"); return cn_627(cn_628, cn_629, cn_630, cn_631, cn_632); }());
    if ((truth(([&]() -> O { O cn_637 = local(v_w, "w"); O cn_638 = ([&]() -> O { O cn_635 = local(v_max_tile_size, "max_tile_size"); O cn_636 = py::int_(0); return item(cn_635, cn_636); }()); return rich_compare(cn_637, cn_638, Py_LE); }())) && truth(([&]() -> O { O cn_643 = local(v_h, "h"); O cn_644 = ([&]() -> O { O cn_641 = local(v_max_tile_size, "max_tile_size"); O cn_642 = py::int_(1); return item(cn_641, cn_642); }()); return rich_compare(cn_643, cn_644, Py_LE); }())))) {
        v_upscale_result = ([&]() -> O { O cn_645 = local(v_upscale, "upscale"); O cn_646 = local(v_img, "img"); O cn_647 = local(v_img_region, "img_region"); return cn_645(cn_646, cn_647); }());
        if (!(truth(([&]() -> O { O cn_648 = builtin("isinstance"); O cn_649 = local(v_upscale_result, "upscale_result"); O cn_650 = item(globals, py::str("Split")); return cn_648(cn_649, cn_650); }())))) {
            if (truth(([&]() -> O { O cn_651 = local(v_progress, "progress"); O cn_652 = py::none(); return py::bool_(cn_651.ptr() != cn_652.ptr()); }()))) {
                ([&]() -> O { O cn_653 = attr(local(v_progress, "progress"), "set_progress"); O cn_654 = py::float_(1.0); return cn_653(cn_654); }());
            }
            return local(v_upscale_result, "upscale_result");
        }
        v_max_tile_size = ([&]() -> O { O cn_655 = local(v_split_tile_size, "split_tile_size"); O cn_656 = local(v_max_tile_size, "max_tile_size"); return cn_655(cn_656); }());
        ([&]() -> O { O cn_657 = attr(item(globals, py::str("logger")), "warning"); O cn_658 = py::str("Unable to upscale the whole image at once. Reduced tile size to %s."); O cn_659 = local(v_max_tile_size, "max_tile_size"); return cn_657(cn_658, cn_659); }());
    }
    v_start_y = py::int_(0);
    v_result = py::none();
    v_scale = py::int_(0);
    v_out_channels = py::int_(0);
    v_restart = py::bool_(true);
    while (truth(local(v_restart, "restart"))) {
        v_restart = py::bool_(false);
        v_tile_count_x = ([&]() -> O { O cn_664 = attr(item(globals, py::str("math")), "ceil"); O cn_665 = ([&]() -> O { O cn_662 = local(v_w, "w"); O cn_663 = ([&]() -> O { O cn_660 = local(v_max_tile_size, "max_tile_size"); O cn_661 = py::int_(0); return item(cn_660, cn_661); }()); return binary(cn_662, cn_663, Op::div); }()); return cn_664(cn_665); }());
        v_tile_count_y = ([&]() -> O { O cn_670 = attr(item(globals, py::str("math")), "ceil"); O cn_671 = ([&]() -> O { O cn_668 = local(v_h, "h"); O cn_669 = ([&]() -> O { O cn_666 = local(v_max_tile_size, "max_tile_size"); O cn_667 = py::int_(1); return item(cn_666, cn_667); }()); return binary(cn_668, cn_669, Op::div); }()); return cn_670(cn_671); }());
        v_tile_size_x = ([&]() -> O { O cn_674 = attr(item(globals, py::str("math")), "ceil"); O cn_675 = ([&]() -> O { O cn_672 = local(v_w, "w"); O cn_673 = local(v_tile_count_x, "tile_count_x"); return binary(cn_672, cn_673, Op::div); }()); return cn_674(cn_675); }());
        v_tile_size_y = ([&]() -> O { O cn_678 = attr(item(globals, py::str("math")), "ceil"); O cn_679 = ([&]() -> O { O cn_676 = local(v_h, "h"); O cn_677 = local(v_tile_count_y, "tile_count_y"); return binary(cn_676, cn_677, Op::div); }()); return cn_678(cn_679); }());
        v_total_tiles = ([&]() -> O { O cn_680 = local(v_tile_count_x, "tile_count_x"); O cn_681 = local(v_tile_count_y, "tile_count_y"); return binary(cn_680, cn_681, Op::mul); }());
        ([&]() -> O { O cn_682 = attr(item(globals, py::str("logger")), "debug"); O cn_683 = py::str("Currently %dx%d tiles each %dx%dpx."); O cn_684 = local(v_tile_count_x, "tile_count_x"); O cn_685 = local(v_tile_count_y, "tile_count_y"); O cn_686 = local(v_tile_size_x, "tile_size_x"); O cn_687 = local(v_tile_size_y, "tile_size_y"); return cn_682(cn_683, cn_684, cn_685, cn_686, cn_687); }());
        v_prev_row_result = py::none();
        v_tiles_processed = py::int_(0);
        for (py::handle cn_688 : py::reinterpret_borrow<py::iterable>(([&]() -> O { O cn_689 = builtin("range"); O cn_690 = local(v_tile_count_y, "tile_count_y"); return cn_689(cn_690); }()))) {
            v_y = py::reinterpret_borrow<O>(cn_688);
            if (truth(([&]() -> O { O cn_691 = local(v_y, "y"); O cn_692 = local(v_start_y, "start_y"); return rich_compare(cn_691, cn_692, Py_LT); }()))) {
                continue;
            }
            v_row_result = py::none();
            v_row_overlap = py::none();
            for (py::handle cn_693 : py::reinterpret_borrow<py::iterable>(([&]() -> O { O cn_694 = builtin("range"); O cn_695 = local(v_tile_count_x, "tile_count_x"); return cn_694(cn_695); }()))) {
                v_x = py::reinterpret_borrow<O>(cn_693);
                v_tile = ([&]() -> O { O cn_705 = attr(([&]() -> O { O cn_700 = item(globals, py::str("Region")); O cn_701 = ([&]() -> O { O cn_696 = local(v_x, "x"); O cn_697 = local(v_tile_size_x, "tile_size_x"); return binary(cn_696, cn_697, Op::mul); }()); O cn_702 = ([&]() -> O { O cn_698 = local(v_y, "y"); O cn_699 = local(v_tile_size_y, "tile_size_y"); return binary(cn_698, cn_699, Op::mul); }()); O cn_703 = local(v_tile_size_x, "tile_size_x"); O cn_704 = local(v_tile_size_y, "tile_size_y"); return cn_700(cn_701, cn_702, cn_703, cn_704); }()), "intersect"); O cn_706 = local(v_img_region, "img_region"); return cn_705(cn_706); }());
                v_pad = ([&]() -> O { O cn_709 = attr(([&]() -> O { O cn_707 = attr(local(v_img_region, "img_region"), "child_padding"); O cn_708 = local(v_tile, "tile"); return cn_707(cn_708); }()), "min"); O cn_710 = local(v_overlap, "overlap"); return cn_709(cn_710); }());
                v_padded_tile = ([&]() -> O { O cn_711 = attr(local(v_tile, "tile"), "add_padding"); O cn_712 = local(v_pad, "pad"); return cn_711(cn_712); }());
                v_upscale_result = ([&]() -> O { O cn_715 = local(v_upscale, "upscale"); O cn_716 = ([&]() -> O { O cn_713 = attr(local(v_padded_tile, "padded_tile"), "read_from"); O cn_714 = local(v_img, "img"); return cn_713(cn_714); }()); O cn_717 = local(v_padded_tile, "padded_tile"); return cn_715(cn_716, cn_717); }());
                if (truth(([&]() -> O { O cn_718 = builtin("isinstance"); O cn_719 = local(v_upscale_result, "upscale_result"); O cn_720 = item(globals, py::str("Split")); return cn_718(cn_719, cn_720); }()))) {
                    v_max_tile_size = ([&]() -> O { O cn_721 = local(v_split_tile_size, "split_tile_size"); O cn_722 = local(v_max_tile_size, "max_tile_size"); return cn_721(cn_722); }());
                    v_new_tile_count_y = ([&]() -> O { O cn_727 = attr(item(globals, py::str("math")), "ceil"); O cn_728 = ([&]() -> O { O cn_725 = local(v_h, "h"); O cn_726 = ([&]() -> O { O cn_723 = local(v_max_tile_size, "max_tile_size"); O cn_724 = py::int_(1); return item(cn_723, cn_724); }()); return binary(cn_725, cn_726, Op::div); }()); return cn_727(cn_728); }());
                    v_new_tile_size_y = ([&]() -> O { O cn_731 = attr(item(globals, py::str("math")), "ceil"); O cn_732 = ([&]() -> O { O cn_729 = local(v_h, "h"); O cn_730 = local(v_new_tile_count_y, "new_tile_count_y"); return binary(cn_729, cn_730, Op::div); }()); return cn_731(cn_732); }());
                    v_start_y = ([&]() -> O { O cn_735 = ([&]() -> O { O cn_733 = local(v_y, "y"); O cn_734 = local(v_tile_size_y, "tile_size_y"); return binary(cn_733, cn_734, Op::mul); }()); O cn_736 = local(v_new_tile_size_y, "new_tile_size_y"); return binary(cn_735, cn_736, Op::floordiv); }());
                    ([&]() -> O { O cn_737 = attr(item(globals, py::str("logger")), "debug"); O cn_738 = py::str("Split occurred. New tile size is %s. Starting at row %d."); O cn_739 = local(v_max_tile_size, "max_tile_size"); O cn_740 = local(v_start_y, "start_y"); return cn_737(cn_738, cn_739, cn_740); }());
                    if (truth(([&]() -> O { O cn_741 = local(v_result, "result"); O cn_742 = py::none(); return py::bool_(cn_741.ptr() != cn_742.ptr()); }()))) {
                        O cn_747 = ([&]() -> O { O cn_745 = ([&]() -> O { O cn_743 = local(v_start_y, "start_y"); O cn_744 = local(v_new_tile_size_y, "new_tile_size_y"); return binary(cn_743, cn_744, Op::mul); }()); O cn_746 = local(v_scale, "scale"); return binary(cn_745, cn_746, Op::mul); }());
                        set_attr(local(v_result, "result"), "offset", cn_747);
                    }
                    v_restart = py::bool_(true);
                    break;
                }
                O cn_750 = unpack(([&]() -> O { O cn_748 = item(globals, py::str("get_h_w_c")); O cn_749 = local(v_upscale_result, "upscale_result"); return cn_748(cn_749); }()), 3);
                v_up_h = item(cn_750, py::int_(0));
                v_up_w = item(cn_750, py::int_(1));
                v_up_c = item(cn_750, py::int_(2));
                v_current_scale = ([&]() -> O { O cn_751 = local(v_up_h, "up_h"); O cn_752 = attr(local(v_padded_tile, "padded_tile"), "height"); return binary(cn_751, cn_752, Op::floordiv); }());
                if (!(truth(([&]() -> O { O cn_753 = local(v_current_scale, "current_scale"); O cn_754 = py::int_(0); return rich_compare(cn_753, cn_754, Py_GT); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
                if (!(truth(([&]() -> O { O cn_759 = ([&]() -> O { O cn_757 = attr(local(v_padded_tile, "padded_tile"), "height"); O cn_758 = local(v_current_scale, "current_scale"); return binary(cn_757, cn_758, Op::mul); }()); O cn_760 = local(v_up_h, "up_h"); return rich_compare(cn_759, cn_760, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
                if (!(truth(([&]() -> O { O cn_765 = ([&]() -> O { O cn_763 = attr(local(v_padded_tile, "padded_tile"), "width"); O cn_764 = local(v_current_scale, "current_scale"); return binary(cn_763, cn_764, Op::mul); }()); O cn_766 = local(v_up_w, "up_w"); return rich_compare(cn_765, cn_766, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
                if (truth(([&]() -> O { O cn_767 = local(v_row_result, "row_result"); O cn_768 = py::none(); return py::bool_(cn_767.ptr() == cn_768.ptr()); }()))) {
                    v_scale = local(v_current_scale, "current_scale");
                    v_out_channels = local(v_up_c, "up_c");
                    v_row_result = ([&]() -> O { O cn_773 = item(globals, py::str("TileBlender")); O cn_774 = ([&]() -> O { O cn_769 = local(v_w, "w"); O cn_770 = local(v_scale, "scale"); return binary(cn_769, cn_770, Op::mul); }()); O cn_775 = ([&]() -> O { O cn_771 = attr(local(v_padded_tile, "padded_tile"), "height"); O cn_772 = local(v_scale, "scale"); return binary(cn_771, cn_772, Op::mul); }()); O cn_776 = local(v_out_channels, "out_channels"); O cn_777 = attr(item(globals, py::str("BlendDirection")), "X"); O cn_778 = item(globals, py::str("half_sin_blend_fn")); O cn_779 = local(v_prev_row_result, "prev_row_result"); return cn_773(py::arg("width") = cn_774, py::arg("height") = cn_775, py::arg("channels") = cn_776, py::arg("direction") = cn_777, py::arg("blend_fn") = cn_778, py::arg("_prev") = cn_779); }());
                    v_prev_row_result = local(v_row_result, "row_result");
                    v_row_overlap = ([&]() -> O { O cn_784 = item(globals, py::str("TileOverlap")); O cn_785 = ([&]() -> O { O cn_780 = attr(local(v_pad, "pad"), "top"); O cn_781 = local(v_scale, "scale"); return binary(cn_780, cn_781, Op::mul); }()); O cn_786 = ([&]() -> O { O cn_782 = attr(local(v_pad, "pad"), "bottom"); O cn_783 = local(v_scale, "scale"); return binary(cn_782, cn_783, Op::mul); }()); return cn_784(cn_785, cn_786); }());
                }
                if (!(truth(([&]() -> O { O cn_787 = local(v_current_scale, "current_scale"); O cn_788 = local(v_scale, "scale"); return rich_compare(cn_787, cn_788, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
                ([&]() -> O { O cn_796 = attr(local(v_row_result, "row_result"), "add_tile"); O cn_797 = local(v_upscale_result, "upscale_result"); O cn_798 = ([&]() -> O { O cn_793 = item(globals, py::str("TileOverlap")); O cn_794 = ([&]() -> O { O cn_789 = attr(local(v_pad, "pad"), "left"); O cn_790 = local(v_scale, "scale"); return binary(cn_789, cn_790, Op::mul); }()); O cn_795 = ([&]() -> O { O cn_791 = attr(local(v_pad, "pad"), "right"); O cn_792 = local(v_scale, "scale"); return binary(cn_791, cn_792, Op::mul); }()); return cn_793(cn_794, cn_795); }()); return cn_796(cn_797, cn_798); }());
                O cn_799 = local(v_tiles_processed, "tiles_processed");
                O cn_800 = py::int_(1);
                O cn_801 = inplace(cn_799, cn_800, Op::add);
                v_tiles_processed = cn_801;
                if (truth(([&]() -> O { O cn_802 = local(v_progress, "progress"); O cn_803 = py::none(); return py::bool_(cn_802.ptr() != cn_803.ptr()); }()))) {
                    ([&]() -> O { O cn_806 = attr(local(v_progress, "progress"), "set_progress"); O cn_807 = ([&]() -> O { O cn_804 = local(v_tiles_processed, "tiles_processed"); O cn_805 = local(v_total_tiles, "total_tiles"); return binary(cn_804, cn_805, Op::div); }()); return cn_806(cn_807); }());
                }
            }
            if (truth(local(v_restart, "restart"))) {
                break;
            }
            if (!(truth(([&]() -> O { O cn_808 = local(v_row_result, "row_result"); O cn_809 = py::none(); return py::bool_(cn_808.ptr() != cn_809.ptr()); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            if (!(truth(([&]() -> O { O cn_810 = local(v_row_overlap, "row_overlap"); O cn_811 = py::none(); return py::bool_(cn_810.ptr() != cn_811.ptr()); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            if (truth(([&]() -> O { O cn_812 = local(v_result, "result"); O cn_813 = py::none(); return py::bool_(cn_812.ptr() == cn_813.ptr()); }()))) {
                v_result = ([&]() -> O { O cn_818 = item(globals, py::str("TileBlender")); O cn_819 = ([&]() -> O { O cn_814 = local(v_w, "w"); O cn_815 = local(v_scale, "scale"); return binary(cn_814, cn_815, Op::mul); }()); O cn_820 = ([&]() -> O { O cn_816 = local(v_h, "h"); O cn_817 = local(v_scale, "scale"); return binary(cn_816, cn_817, Op::mul); }()); O cn_821 = local(v_out_channels, "out_channels"); O cn_822 = attr(item(globals, py::str("BlendDirection")), "Y"); O cn_823 = item(globals, py::str("half_sin_blend_fn")); return cn_818(py::arg("width") = cn_819, py::arg("height") = cn_820, py::arg("channels") = cn_821, py::arg("direction") = cn_822, py::arg("blend_fn") = cn_823); }());
            }
            ([&]() -> O { O cn_825 = attr(local(v_result, "result"), "add_tile"); O cn_826 = ([&]() -> O { O cn_824 = attr(local(v_row_result, "row_result"), "get_result"); return cn_824(); }()); O cn_827 = local(v_row_overlap, "row_overlap"); return cn_825(cn_826, cn_827); }());
        }
    }
    if (!(truth(([&]() -> O { O cn_828 = local(v_result, "result"); O cn_829 = py::none(); return py::bool_(cn_828.ptr() != cn_829.ptr()); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
    return ([&]() -> O { O cn_830 = attr(local(v_result, "result"), "get_result"); return cn_830(); }());
}
O tile_source_exact_split_pad_image(const O &globals, O v_img, O v_min_size) {
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
    O cn_833 = unpack(([&]() -> O { O cn_831 = item(globals, py::str("get_h_w_c")); O cn_832 = local(v_img, "img"); return cn_831(cn_832); }()), 3);
    v_h = item(cn_833, py::int_(0));
    v_w = item(cn_833, py::int_(1));
    v__ = item(cn_833, py::int_(2));
    O cn_834 = unpack(local(v_min_size, "min_size"), 2);
    v_min_w = item(cn_834, py::int_(0));
    v_min_h = item(cn_834, py::int_(1));
    v_x = ([&]() -> O { O cn_840 = ([&]() -> O { O cn_837 = builtin("max"); O cn_838 = py::int_(0); O cn_839 = ([&]() -> O { O cn_835 = local(v_min_w, "min_w"); O cn_836 = local(v_w, "w"); return binary(cn_835, cn_836, Op::sub); }()); return cn_837(cn_838, cn_839); }()); O cn_841 = py::int_(2); return binary(cn_840, cn_841, Op::div); }());
    v_y = ([&]() -> O { O cn_847 = ([&]() -> O { O cn_844 = builtin("max"); O cn_845 = py::int_(0); O cn_846 = ([&]() -> O { O cn_842 = local(v_min_h, "min_h"); O cn_843 = local(v_h, "h"); return binary(cn_842, cn_843, Op::sub); }()); return cn_844(cn_845, cn_846); }()); O cn_848 = py::int_(2); return binary(cn_847, cn_848, Op::div); }());
    v_padding = ([&]() -> O { O cn_857 = item(globals, py::str("Padding")); O cn_858 = ([&]() -> O { O cn_849 = attr(item(globals, py::str("math")), "floor"); O cn_850 = local(v_y, "y"); return cn_849(cn_850); }()); O cn_859 = ([&]() -> O { O cn_851 = attr(item(globals, py::str("math")), "floor"); O cn_852 = local(v_x, "x"); return cn_851(cn_852); }()); O cn_860 = ([&]() -> O { O cn_853 = attr(item(globals, py::str("math")), "ceil"); O cn_854 = local(v_y, "y"); return cn_853(cn_854); }()); O cn_861 = ([&]() -> O { O cn_855 = attr(item(globals, py::str("math")), "ceil"); O cn_856 = local(v_x, "x"); return cn_855(cn_856); }()); return cn_857(cn_858, cn_859, cn_860, cn_861); }());
    return make_tuple({([&]() -> O { O cn_862 = item(globals, py::str("create_border")); O cn_863 = local(v_img, "img"); O cn_864 = attr(item(globals, py::str("BorderType")), "REFLECT_MIRROR"); O cn_865 = local(v_padding, "padding"); return cn_862(cn_863, cn_864, cn_865); }()), local(v_padding, "padding")});
}
O tile_source_exact_split__Segment_length(const O &globals, const O &self) {
    (void)globals;
    (void)self;
    return ([&]() -> O { O cn_866 = attr(self, "end"); O cn_867 = attr(self, "start"); return binary(cn_866, cn_867, Op::sub); }());
}
O tile_source_exact_split__Segment_padded_length(const O &globals, const O &self) {
    (void)globals;
    (void)self;
    return ([&]() -> O { O cn_872 = ([&]() -> O { O cn_868 = attr(self, "end"); O cn_869 = attr(self, "end_padding"); return binary(cn_868, cn_869, Op::add); }()); O cn_873 = ([&]() -> O { O cn_870 = attr(self, "start"); O cn_871 = attr(self, "start_padding"); return binary(cn_870, cn_871, Op::sub); }()); return binary(cn_872, cn_873, Op::sub); }());
}
O tile_source_exact_split_exact_split_into_segments(const O &globals, O v_length, O v_exact, O v_overlap) {
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
    if (truth(([&]() -> O { O cn_874 = local(v_length, "length"); O cn_875 = local(v_exact, "exact"); return rich_compare(cn_874, cn_875, Py_EQ); }()))) {
        return make_list({([&]() -> O { O cn_876 = item(globals, py::str("_Segment")); O cn_877 = py::int_(0); O cn_878 = local(v_exact, "exact"); O cn_879 = py::int_(0); O cn_880 = py::int_(0); return cn_876(cn_877, cn_878, cn_879, cn_880); }())});
    }
    if (!(truth(([&]() -> O { O cn_881 = local(v_length, "length"); O cn_882 = local(v_exact, "exact"); return rich_compare(cn_881, cn_882, Py_GT); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
    if (!(truth(([&]() -> O { O cn_887 = local(v_exact, "exact"); O cn_888 = ([&]() -> O { O cn_885 = local(v_overlap, "overlap"); O cn_886 = py::int_(2); return binary(cn_885, cn_886, Op::mul); }()); return rich_compare(cn_887, cn_888, Py_GT); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
    v_result = make_list({});
    v_add = py::cpp_function([=](O v_s) mutable -> O {
        if (!(truth(([&]() -> O { O cn_889 = attr(local(v_s, "s"), "padded_length"); O cn_890 = local(v_exact, "exact"); return rich_compare(cn_889, cn_890, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
        ([&]() -> O { O cn_891 = attr(local(v_result, "result"), "append"); O cn_892 = local(v_s, "s"); return cn_891(cn_892); }());
        return py::none();
    });
    ([&]() -> O { O cn_900 = local(v_add, "add"); O cn_901 = ([&]() -> O { O cn_895 = item(globals, py::str("_Segment")); O cn_896 = py::int_(0); O cn_897 = ([&]() -> O { O cn_893 = local(v_exact, "exact"); O cn_894 = local(v_overlap, "overlap"); return binary(cn_893, cn_894, Op::sub); }()); O cn_898 = py::int_(0); O cn_899 = local(v_overlap, "overlap"); return cn_895(cn_896, cn_897, cn_898, cn_899); }()); return cn_900(cn_901); }());
    while (truth(([&]() -> O { O cn_906 = attr(([&]() -> O { O cn_904 = local(v_result, "result"); O cn_905 = negative(py::int_(1)); return item(cn_904, cn_905); }()), "end"); O cn_907 = local(v_length, "length"); return rich_compare(cn_906, cn_907, Py_LT); }()))) {
        v_start_padding = local(v_overlap, "overlap");
        v_start = attr(([&]() -> O { O cn_908 = local(v_result, "result"); O cn_909 = negative(py::int_(1)); return item(cn_908, cn_909); }()), "end");
        v_end = ([&]() -> O { O cn_914 = ([&]() -> O { O cn_910 = local(v_start, "start"); O cn_911 = local(v_exact, "exact"); return binary(cn_910, cn_911, Op::add); }()); O cn_915 = ([&]() -> O { O cn_912 = local(v_overlap, "overlap"); O cn_913 = py::int_(2); return binary(cn_912, cn_913, Op::mul); }()); return binary(cn_914, cn_915, Op::sub); }());
        v_end_padding = local(v_overlap, "overlap");
        if (truth(([&]() -> O { O cn_920 = ([&]() -> O { O cn_918 = local(v_end, "end"); O cn_919 = local(v_end_padding, "end_padding"); return binary(cn_918, cn_919, Op::add); }()); O cn_921 = local(v_length, "length"); return rich_compare(cn_920, cn_921, Py_GE); }()))) {
            v_end_padding = py::int_(0);
            v_end = local(v_length, "length");
            v_start_padding = ([&]() -> O { O cn_924 = local(v_exact, "exact"); O cn_925 = ([&]() -> O { O cn_922 = local(v_end, "end"); O cn_923 = local(v_start, "start"); return binary(cn_922, cn_923, Op::sub); }()); return binary(cn_924, cn_925, Op::sub); }());
        }
        ([&]() -> O { O cn_931 = local(v_add, "add"); O cn_932 = ([&]() -> O { O cn_926 = item(globals, py::str("_Segment")); O cn_927 = local(v_start, "start"); O cn_928 = local(v_end, "end"); O cn_929 = local(v_start_padding, "start_padding"); O cn_930 = local(v_end_padding, "end_padding"); return cn_926(cn_927, cn_928, cn_929, cn_930); }()); return cn_931(cn_932); }());
    }
    return local(v_result, "result");
}
O tile_source_exact_split_exact_split_into_regions(const O &globals, O v_w, O v_h, O v_exact_w, O v_exact_h, O v_overlap) {
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
    v_x_segments = ([&]() -> O { O cn_933 = item(globals, py::str("_exact_split_into_segments")); O cn_934 = local(v_w, "w"); O cn_935 = local(v_exact_w, "exact_w"); O cn_936 = local(v_overlap, "overlap"); return cn_933(cn_934, cn_935, cn_936); }());
    v_y_segments = ([&]() -> O { O cn_937 = item(globals, py::str("_exact_split_into_segments")); O cn_938 = local(v_h, "h"); O cn_939 = local(v_exact_h, "exact_h"); O cn_940 = local(v_overlap, "overlap"); return cn_937(cn_938, cn_939, cn_940); }());
    ([&]() -> O { O cn_941 = attr(item(globals, py::str("logger")), "info"); O cn_942 = py::str("Image is split into %dx%d tiles each exactly %dx%dpx."); O cn_943 = py::int_(py::len(local(v_x_segments, "x_segments"))); O cn_944 = py::int_(py::len(local(v_y_segments, "y_segments"))); O cn_945 = local(v_exact_w, "exact_w"); O cn_946 = local(v_exact_h, "exact_h"); return cn_941(cn_942, cn_943, cn_944, cn_945, cn_946); }());
    v_result = make_list({});
    for (py::handle cn_947 : py::reinterpret_borrow<py::iterable>(local(v_y_segments, "y_segments"))) {
        v_y = py::reinterpret_borrow<O>(cn_947);
        v_row = make_list({});
        for (py::handle cn_948 : py::reinterpret_borrow<py::iterable>(local(v_x_segments, "x_segments"))) {
            v_x = py::reinterpret_borrow<O>(cn_948);
            ([&]() -> O { O cn_959 = attr(local(v_row, "row"), "append"); O cn_960 = make_tuple({([&]() -> O { O cn_949 = item(globals, py::str("Region")); O cn_950 = attr(local(v_x, "x"), "start"); O cn_951 = attr(local(v_y, "y"), "start"); O cn_952 = attr(local(v_x, "x"), "length"); O cn_953 = attr(local(v_y, "y"), "length"); return cn_949(cn_950, cn_951, cn_952, cn_953); }()), ([&]() -> O { O cn_954 = item(globals, py::str("Padding")); O cn_955 = attr(local(v_y, "y"), "start_padding"); O cn_956 = attr(local(v_x, "x"), "end_padding"); O cn_957 = attr(local(v_y, "y"), "end_padding"); O cn_958 = attr(local(v_x, "x"), "start_padding"); return cn_954(cn_955, cn_956, cn_957, cn_958); }())}); return cn_959(cn_960); }());
        }
        ([&]() -> O { O cn_961 = attr(local(v_result, "result"), "append"); O cn_962 = local(v_row, "row"); return cn_961(cn_962); }());
    }
    return local(v_result, "result");
}
O tile_source_exact_split_exact_split_without_padding(const O &globals, O v_img, O v_exact_size, O v_upscale, O v_overlap, O v_progress) {
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
    O cn_965 = unpack(([&]() -> O { O cn_963 = item(globals, py::str("get_h_w_c")); O cn_964 = local(v_img, "img"); return cn_963(cn_964); }()), 3);
    v_h = item(cn_965, py::int_(0));
    v_w = item(cn_965, py::int_(1));
    v__ = item(cn_965, py::int_(2));
    O cn_966 = unpack(local(v_exact_size, "exact_size"), 2);
    v_exact_w = item(cn_966, py::int_(0));
    v_exact_h = item(cn_966, py::int_(1));
    if (!((truth(([&]() -> O { O cn_967 = local(v_w, "w"); O cn_968 = local(v_exact_w, "exact_w"); return rich_compare(cn_967, cn_968, Py_GE); }())) && truth(([&]() -> O { O cn_969 = local(v_h, "h"); O cn_970 = local(v_exact_h, "exact_h"); return rich_compare(cn_969, cn_970, Py_GE); }()))))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
    if (truth(([&]() -> O { O cn_971 = make_tuple({local(v_w, "w"), local(v_h, "h")}); O cn_972 = local(v_exact_size, "exact_size"); return rich_compare(cn_971, cn_972, Py_EQ); }()))) {
        if (truth(([&]() -> O { O cn_973 = local(v_progress, "progress"); O cn_974 = py::none(); return py::bool_(cn_973.ptr() != cn_974.ptr()); }()))) {
            ([&]() -> O { O cn_975 = attr(local(v_progress, "progress"), "set_progress"); O cn_976 = py::float_(1.0); return cn_975(cn_976); }());
        }
        return ([&]() -> O { O cn_982 = local(v_upscale, "upscale"); O cn_983 = local(v_img, "img"); O cn_984 = ([&]() -> O { O cn_977 = item(globals, py::str("Region")); O cn_978 = py::int_(0); O cn_979 = py::int_(0); O cn_980 = local(v_w, "w"); O cn_981 = local(v_h, "h"); return cn_977(cn_978, cn_979, cn_980, cn_981); }()); return cn_982(cn_983, cn_984); }());
    }
    v_result = py::none();
    v_scale = py::int_(0);
    v_out_channels = py::int_(0);
    v_regions = ([&]() -> O { O cn_985 = item(globals, py::str("_exact_split_into_regions")); O cn_986 = local(v_w, "w"); O cn_987 = local(v_h, "h"); O cn_988 = local(v_exact_w, "exact_w"); O cn_989 = local(v_exact_h, "exact_h"); O cn_990 = local(v_overlap, "overlap"); return cn_985(cn_986, cn_987, cn_988, cn_989, cn_990); }());
    v_total_tiles = ([&]() -> O { O cn_991 = py::int_(0); for(py::handle cn_992 : py::reinterpret_borrow<py::iterable>(local(v_regions, "regions"))) { O v_row = py::reinterpret_borrow<O>(cn_992); cn_991 = binary(cn_991, py::int_(py::len(local(v_row, "row"))), Op::add); } return cn_991; }());
    v_tiles_processed = py::int_(0);
    for (py::handle cn_993 : py::reinterpret_borrow<py::iterable>(local(v_regions, "regions"))) {
        v_row = py::reinterpret_borrow<O>(cn_993);
        v_row_result = py::none();
        v_row_overlap = py::none();
        for (py::handle cn_994 : py::reinterpret_borrow<py::iterable>(local(v_row, "row"))) {
            O cn_995 = unpack(py::reinterpret_borrow<O>(cn_994), 2);
            v_tile = item(cn_995, py::int_(0));
            v_pad = item(cn_995, py::int_(1));
            v_padded_tile = ([&]() -> O { O cn_996 = attr(local(v_tile, "tile"), "add_padding"); O cn_997 = local(v_pad, "pad"); return cn_996(cn_997); }());
            if (!(truth(([&]() -> O { O cn_998 = attr(local(v_padded_tile, "padded_tile"), "size"); O cn_999 = local(v_exact_size, "exact_size"); return rich_compare(cn_998, cn_999, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            v_upscale_result = ([&]() -> O { O cn_1002 = local(v_upscale, "upscale"); O cn_1003 = ([&]() -> O { O cn_1000 = attr(local(v_padded_tile, "padded_tile"), "read_from"); O cn_1001 = local(v_img, "img"); return cn_1000(cn_1001); }()); O cn_1004 = local(v_padded_tile, "padded_tile"); return cn_1002(cn_1003, cn_1004); }());
            O cn_1007 = unpack(([&]() -> O { O cn_1005 = item(globals, py::str("get_h_w_c")); O cn_1006 = local(v_upscale_result, "upscale_result"); return cn_1005(cn_1006); }()), 3);
            v_up_h = item(cn_1007, py::int_(0));
            v_up_w = item(cn_1007, py::int_(1));
            v_up_c = item(cn_1007, py::int_(2));
            v_current_scale = ([&]() -> O { O cn_1008 = local(v_up_h, "up_h"); O cn_1009 = local(v_exact_h, "exact_h"); return binary(cn_1008, cn_1009, Op::floordiv); }());
            if (!(truth(([&]() -> O { O cn_1010 = local(v_current_scale, "current_scale"); O cn_1011 = py::int_(0); return rich_compare(cn_1010, cn_1011, Py_GT); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            if (!(truth(([&]() -> O { O cn_1016 = ([&]() -> O { O cn_1014 = local(v_exact_h, "exact_h"); O cn_1015 = local(v_current_scale, "current_scale"); return binary(cn_1014, cn_1015, Op::mul); }()); O cn_1017 = local(v_up_h, "up_h"); return rich_compare(cn_1016, cn_1017, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            if (!(truth(([&]() -> O { O cn_1022 = ([&]() -> O { O cn_1020 = local(v_exact_w, "exact_w"); O cn_1021 = local(v_current_scale, "current_scale"); return binary(cn_1020, cn_1021, Op::mul); }()); O cn_1023 = local(v_up_w, "up_w"); return rich_compare(cn_1022, cn_1023, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            if (truth(([&]() -> O { O cn_1024 = local(v_row_result, "row_result"); O cn_1025 = py::none(); return py::bool_(cn_1024.ptr() == cn_1025.ptr()); }()))) {
                v_scale = local(v_current_scale, "current_scale");
                v_out_channels = local(v_up_c, "up_c");
                v_row_result = ([&]() -> O { O cn_1030 = item(globals, py::str("TileBlender")); O cn_1031 = ([&]() -> O { O cn_1026 = local(v_w, "w"); O cn_1027 = local(v_scale, "scale"); return binary(cn_1026, cn_1027, Op::mul); }()); O cn_1032 = ([&]() -> O { O cn_1028 = local(v_exact_h, "exact_h"); O cn_1029 = local(v_scale, "scale"); return binary(cn_1028, cn_1029, Op::mul); }()); O cn_1033 = local(v_out_channels, "out_channels"); O cn_1034 = attr(item(globals, py::str("BlendDirection")), "X"); O cn_1035 = item(globals, py::str("half_sin_blend_fn")); return cn_1030(py::arg("width") = cn_1031, py::arg("height") = cn_1032, py::arg("channels") = cn_1033, py::arg("direction") = cn_1034, py::arg("blend_fn") = cn_1035); }());
                v_row_overlap = ([&]() -> O { O cn_1040 = item(globals, py::str("TileOverlap")); O cn_1041 = ([&]() -> O { O cn_1036 = attr(local(v_pad, "pad"), "top"); O cn_1037 = local(v_scale, "scale"); return binary(cn_1036, cn_1037, Op::mul); }()); O cn_1042 = ([&]() -> O { O cn_1038 = attr(local(v_pad, "pad"), "bottom"); O cn_1039 = local(v_scale, "scale"); return binary(cn_1038, cn_1039, Op::mul); }()); return cn_1040(cn_1041, cn_1042); }());
            }
            if (!(truth(([&]() -> O { O cn_1043 = local(v_current_scale, "current_scale"); O cn_1044 = local(v_scale, "scale"); return rich_compare(cn_1043, cn_1044, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            ([&]() -> O { O cn_1052 = attr(local(v_row_result, "row_result"), "add_tile"); O cn_1053 = local(v_upscale_result, "upscale_result"); O cn_1054 = ([&]() -> O { O cn_1049 = item(globals, py::str("TileOverlap")); O cn_1050 = ([&]() -> O { O cn_1045 = attr(local(v_pad, "pad"), "left"); O cn_1046 = local(v_scale, "scale"); return binary(cn_1045, cn_1046, Op::mul); }()); O cn_1051 = ([&]() -> O { O cn_1047 = attr(local(v_pad, "pad"), "right"); O cn_1048 = local(v_scale, "scale"); return binary(cn_1047, cn_1048, Op::mul); }()); return cn_1049(cn_1050, cn_1051); }()); return cn_1052(cn_1053, cn_1054); }());
            O cn_1055 = local(v_tiles_processed, "tiles_processed");
            O cn_1056 = py::int_(1);
            O cn_1057 = inplace(cn_1055, cn_1056, Op::add);
            v_tiles_processed = cn_1057;
            if (truth(([&]() -> O { O cn_1058 = local(v_progress, "progress"); O cn_1059 = py::none(); return py::bool_(cn_1058.ptr() != cn_1059.ptr()); }()))) {
                ([&]() -> O { O cn_1062 = attr(local(v_progress, "progress"), "set_progress"); O cn_1063 = ([&]() -> O { O cn_1060 = local(v_tiles_processed, "tiles_processed"); O cn_1061 = local(v_total_tiles, "total_tiles"); return binary(cn_1060, cn_1061, Op::div); }()); return cn_1062(cn_1063); }());
            }
        }
        if (!(truth(([&]() -> O { O cn_1064 = local(v_row_result, "row_result"); O cn_1065 = py::none(); return py::bool_(cn_1064.ptr() != cn_1065.ptr()); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
        if (!(truth(([&]() -> O { O cn_1066 = local(v_row_overlap, "row_overlap"); O cn_1067 = py::none(); return py::bool_(cn_1066.ptr() != cn_1067.ptr()); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
        if (truth(([&]() -> O { O cn_1068 = local(v_result, "result"); O cn_1069 = py::none(); return py::bool_(cn_1068.ptr() == cn_1069.ptr()); }()))) {
            v_result = ([&]() -> O { O cn_1074 = item(globals, py::str("TileBlender")); O cn_1075 = ([&]() -> O { O cn_1070 = local(v_w, "w"); O cn_1071 = local(v_scale, "scale"); return binary(cn_1070, cn_1071, Op::mul); }()); O cn_1076 = ([&]() -> O { O cn_1072 = local(v_h, "h"); O cn_1073 = local(v_scale, "scale"); return binary(cn_1072, cn_1073, Op::mul); }()); O cn_1077 = local(v_out_channels, "out_channels"); O cn_1078 = attr(item(globals, py::str("BlendDirection")), "Y"); O cn_1079 = item(globals, py::str("half_sin_blend_fn")); return cn_1074(py::arg("width") = cn_1075, py::arg("height") = cn_1076, py::arg("channels") = cn_1077, py::arg("direction") = cn_1078, py::arg("blend_fn") = cn_1079); }());
        }
        ([&]() -> O { O cn_1081 = attr(local(v_result, "result"), "add_tile"); O cn_1082 = ([&]() -> O { O cn_1080 = attr(local(v_row_result, "row_result"), "get_result"); return cn_1080(); }()); O cn_1083 = local(v_row_overlap, "row_overlap"); return cn_1081(cn_1082, cn_1083); }());
    }
    if (!(truth(([&]() -> O { O cn_1084 = local(v_result, "result"); O cn_1085 = py::none(); return py::bool_(cn_1084.ptr() != cn_1085.ptr()); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
    return ([&]() -> O { O cn_1086 = attr(local(v_result, "result"), "get_result"); return cn_1086(); }());
}
O tile_source_exact_split_exact_split(const O &globals, O v_img, O v_exact_size, O v_upscale, O v_overlap, O v_progress) {
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
    O cn_1090 = unpack(([&]() -> O { O cn_1087 = item(globals, py::str("_pad_image")); O cn_1088 = local(v_img, "img"); O cn_1089 = local(v_exact_size, "exact_size"); return cn_1087(cn_1088, cn_1089); }()), 2);
    v_img = item(cn_1090, py::int_(0));
    v_base_padding = item(cn_1090, py::int_(1));
    O cn_1093 = unpack(([&]() -> O { O cn_1091 = item(globals, py::str("get_h_w_c")); O cn_1092 = local(v_img, "img"); return cn_1091(cn_1092); }()), 3);
    v_h = item(cn_1093, py::int_(0));
    v_w = item(cn_1093, py::int_(1));
    v__ = item(cn_1093, py::int_(2));
    v_result = ([&]() -> O { O cn_1094 = item(globals, py::str("_exact_split_without_padding")); O cn_1095 = local(v_img, "img"); O cn_1096 = local(v_exact_size, "exact_size"); O cn_1097 = local(v_upscale, "upscale"); O cn_1098 = local(v_overlap, "overlap"); O cn_1099 = local(v_progress, "progress"); return cn_1094(cn_1095, cn_1096, cn_1097, cn_1098, cn_1099); }());
    v_scale = ([&]() -> O { O cn_1104 = ([&]() -> O { O cn_1102 = ([&]() -> O { O cn_1100 = item(globals, py::str("get_h_w_c")); O cn_1101 = local(v_result, "result"); return cn_1100(cn_1101); }()); O cn_1103 = py::int_(0); return item(cn_1102, cn_1103); }()); O cn_1105 = local(v_h, "h"); return binary(cn_1104, cn_1105, Op::floordiv); }());
    if (truth(attr(local(v_base_padding, "base_padding"), "empty"))) {
        return local(v_result, "result");
    }
    return ([&]() -> O { O cn_1115 = attr(([&]() -> O { O cn_1113 = attr(([&]() -> O { O cn_1111 = attr(([&]() -> O { O cn_1106 = item(globals, py::str("Region")); O cn_1107 = py::int_(0); O cn_1108 = py::int_(0); O cn_1109 = local(v_w, "w"); O cn_1110 = local(v_h, "h"); return cn_1106(cn_1107, cn_1108, cn_1109, cn_1110); }()), "remove_padding"); O cn_1112 = local(v_base_padding, "base_padding"); return cn_1111(cn_1112); }()), "scale"); O cn_1114 = local(v_scale, "scale"); return cn_1113(cn_1114); }()), "read_from"); O cn_1116 = local(v_result, "result"); return cn_1115(cn_1116); }());
}
O tile_source_tiler_Tiler_allow_smaller_tile_size(const O &globals, const O &self) {
    (void)globals;
    (void)self;
    return py::none();
}
O tile_source_tiler_Tiler_starting_tile_size(const O &globals, const O &self, O v_width, O v_height, O v_channels) {
    (void)globals;
    (void)self;
    (void)v_width;
    (void)v_height;
    (void)v_channels;
    return py::none();
}
O tile_source_tiler_Tiler_split(const O &globals, const O &self, O v_tile_size) {
    O v_h;
    O v_w;
    (void)globals;
    (void)self;
    (void)v_tile_size;
    O cn_1117 = unpack(local(v_tile_size, "tile_size"), 2);
    v_w = item(cn_1117, py::int_(0));
    v_h = item(cn_1117, py::int_(1));
    if (!((truth(([&]() -> O { O cn_1118 = local(v_w, "w"); O cn_1119 = py::int_(16); return rich_compare(cn_1118, cn_1119, Py_GE); }())) && truth(([&]() -> O { O cn_1120 = local(v_h, "h"); O cn_1121 = py::int_(16); return rich_compare(cn_1120, cn_1121, Py_GE); }()))))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
    return make_tuple({([&]() -> O { O cn_1124 = builtin("max"); O cn_1125 = py::int_(16); O cn_1126 = ([&]() -> O { O cn_1122 = local(v_w, "w"); O cn_1123 = py::int_(2); return binary(cn_1122, cn_1123, Op::floordiv); }()); return cn_1124(cn_1125, cn_1126); }()), ([&]() -> O { O cn_1129 = builtin("max"); O cn_1130 = py::int_(16); O cn_1131 = ([&]() -> O { O cn_1127 = local(v_h, "h"); O cn_1128 = py::int_(2); return binary(cn_1127, cn_1128, Op::floordiv); }()); return cn_1129(cn_1130, cn_1131); }())});
}
O tile_source_tiler_NoTiling_allow_smaller_tile_size(const O &globals, const O &self) {
    (void)globals;
    (void)self;
    return py::bool_(true);
}
O tile_source_tiler_NoTiling_starting_tile_size(const O &globals, const O &self, O v_width, O v_height, O v_channels) {
    O v_size;
    (void)globals;
    (void)self;
    (void)v_width;
    (void)v_height;
    (void)v_channels;
    v_size = ([&]() -> O { O cn_1132 = builtin("max"); O cn_1133 = local(v_width, "width"); O cn_1134 = local(v_height, "height"); return cn_1132(cn_1133, cn_1134); }());
    return make_tuple({local(v_size, "size"), local(v_size, "size")});
}
O tile_source_tiler_NoTiling_split(const O &globals, const O &self, O v_tile_size) {
    (void)globals;
    (void)self;
    (void)v_tile_size;
    raise(PyExc_ValueError, py::str("Image cannot be upscale with No Tiling mode."));
}
O tile_source_tiler_MaxTileSize_init(const O &globals, const O &self, O v_tile_size) {
    (void)globals;
    (void)self;
    (void)v_tile_size;
    O cn_1135 = local(v_tile_size, "tile_size");
    set_attr(self, "tile_size", cn_1135);
    return py::none();
}
O tile_source_tiler_MaxTileSize_allow_smaller_tile_size(const O &globals, const O &self) {
    (void)globals;
    (void)self;
    return py::bool_(true);
}
O tile_source_tiler_MaxTileSize_starting_tile_size(const O &globals, const O &self, O v_width, O v_height, O v_channels) {
    O v_max_tile_size;
    O v_size;
    (void)globals;
    (void)self;
    (void)v_width;
    (void)v_height;
    (void)v_channels;
    v_max_tile_size = ([&]() -> O { O cn_1140 = builtin("max"); O cn_1141 = ([&]() -> O { O cn_1136 = local(v_width, "width"); O cn_1137 = py::int_(10); return binary(cn_1136, cn_1137, Op::add); }()); O cn_1142 = ([&]() -> O { O cn_1138 = local(v_height, "height"); O cn_1139 = py::int_(10); return binary(cn_1138, cn_1139, Op::add); }()); return cn_1140(cn_1141, cn_1142); }());
    v_size = ([&]() -> O { O cn_1143 = builtin("min"); O cn_1144 = attr(self, "tile_size"); O cn_1145 = local(v_max_tile_size, "max_tile_size"); return cn_1143(cn_1144, cn_1145); }());
    return make_tuple({local(v_size, "size"), local(v_size, "size")});
}
O tile_source_tiler_ExactTileSize_init(const O &globals, const O &self, O v_exact_size) {
    (void)globals;
    (void)self;
    (void)v_exact_size;
    O cn_1146 = local(v_exact_size, "exact_size");
    set_attr(self, "exact_size", cn_1146);
    return py::none();
}
O tile_source_tiler_ExactTileSize_allow_smaller_tile_size(const O &globals, const O &self) {
    (void)globals;
    (void)self;
    return py::bool_(false);
}
O tile_source_tiler_ExactTileSize_starting_tile_size(const O &globals, const O &self, O v_width, O v_height, O v_channels) {
    (void)globals;
    (void)self;
    (void)v_width;
    (void)v_height;
    (void)v_channels;
    return attr(self, "exact_size");
}
O tile_source_tiler_ExactTileSize_split(const O &globals, const O &self, O v_tile_size) {
    (void)globals;
    (void)self;
    (void)v_tile_size;
    raise(PyExc_ValueError, concat_text({py::str("Splits are not supported for exact size ("), py::str(([&]() -> O { O cn_1147 = attr(self, "exact_size"); O cn_1148 = py::int_(0); return item(cn_1147, cn_1148); }())), py::str("x"), py::str(([&]() -> O { O cn_1149 = attr(self, "exact_size"); O cn_1150 = py::int_(1); return item(cn_1149, cn_1150); }())), py::str("px) splitting. This typically means that your machine does not have enough VRAM to run the current model.")}));
}
O tile_source_tiler_BoundedTileSize_init(const O &globals, const O &self, O v_tile_size, O v_min_size, O v_max_size) {
    (void)globals;
    (void)self;
    (void)v_tile_size;
    (void)v_min_size;
    (void)v_max_size;
    O cn_1151 = unpack((truth(local(v_min_size, "min_size")) ? O(local(v_min_size, "min_size")) : O(make_tuple({py::int_(1), py::int_(1)}))), 2);
    O cn_1152 = item(cn_1151, py::int_(0));
    set_attr(self, "min_w", cn_1152);
    O cn_1153 = item(cn_1151, py::int_(1));
    set_attr(self, "min_h", cn_1153);
    O cn_1158 = unpack((truth(local(v_max_size, "max_size")) ? O(local(v_max_size, "max_size")) : O(make_tuple({([&]() -> O { O cn_1154 = py::int_(2); O cn_1155 = py::int_(31); return binary(cn_1154, cn_1155, Op::pow); }()), ([&]() -> O { O cn_1156 = py::int_(2); O cn_1157 = py::int_(31); return binary(cn_1156, cn_1157, Op::pow); }())}))), 2);
    O cn_1159 = item(cn_1158, py::int_(0));
    set_attr(self, "max_w", cn_1159);
    O cn_1160 = item(cn_1158, py::int_(1));
    set_attr(self, "max_h", cn_1160);
    O cn_1169 = ([&]() -> O { O cn_1165 = builtin("max"); O cn_1166 = attr(self, "min_w"); O cn_1167 = attr(self, "min_h"); O cn_1168 = ([&]() -> O { O cn_1161 = builtin("min"); O cn_1162 = local(v_tile_size, "tile_size"); O cn_1163 = attr(self, "max_w"); O cn_1164 = attr(self, "max_h"); return cn_1161(cn_1162, cn_1163, cn_1164); }()); return cn_1165(cn_1166, cn_1167, cn_1168); }());
    set_attr(self, "tile_size", cn_1169);
    return py::none();
}
O tile_source_tiler_BoundedTileSize_allow_smaller_tile_size(const O &globals, const O &self) {
    (void)globals;
    (void)self;
    return py::bool_(false);
}
O tile_source_tiler_BoundedTileSize_starting_tile_size(const O &globals, const O &self, O v_width, O v_height, O v_channels) {
    (void)globals;
    (void)self;
    (void)v_width;
    (void)v_height;
    (void)v_channels;
    return make_tuple({attr(self, "tile_size"), attr(self, "tile_size")});
}
O tile_source_tiler_BoundedTileSize_split(const O &globals, const O &self, O v_tile_size) {
    O v_h;
    O v_new_h;
    O v_new_w;
    O v_w;
    (void)globals;
    (void)self;
    (void)v_tile_size;
    O cn_1170 = unpack(local(v_tile_size, "tile_size"), 2);
    v_w = item(cn_1170, py::int_(0));
    v_h = item(cn_1170, py::int_(1));
    v_new_w = ([&]() -> O { O cn_1173 = builtin("max"); O cn_1174 = attr(self, "min_w"); O cn_1175 = ([&]() -> O { O cn_1171 = local(v_w, "w"); O cn_1172 = py::int_(2); return binary(cn_1171, cn_1172, Op::floordiv); }()); return cn_1173(cn_1174, cn_1175); }());
    v_new_h = ([&]() -> O { O cn_1178 = builtin("max"); O cn_1179 = attr(self, "min_h"); O cn_1180 = ([&]() -> O { O cn_1176 = local(v_h, "h"); O cn_1177 = py::int_(2); return binary(cn_1176, cn_1177, Op::floordiv); }()); return cn_1178(cn_1179, cn_1180); }());
    if ((truth(([&]() -> O { O cn_1181 = local(v_new_w, "new_w"); O cn_1182 = local(v_w, "w"); return rich_compare(cn_1181, cn_1182, Py_EQ); }())) && truth(([&]() -> O { O cn_1183 = local(v_new_h, "new_h"); O cn_1184 = local(v_h, "h"); return rich_compare(cn_1183, cn_1184, Py_EQ); }())))) {
        raise(PyExc_ValueError, concat_text({py::str("Cannot reduce tile size below the minimum size ("), py::str(attr(self, "min_w")), py::str("x"), py::str(attr(self, "min_h")), py::str("). This typically means your machine does not have enough VRAM.")}));
    }
    return make_tuple({local(v_new_w, "new_w"), local(v_new_h, "new_h")});
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
    module.def("tiling_source_auto_split_auto_split", &tile_source_auto_split_auto_split);
    module.def("tiling_source_auto_split_exact_split", &tile_source_auto_split_exact_split);
    module.def("tiling_source_auto_split_max_split", &tile_source_auto_split_max_split);
    module.def("tiling_source_exact_split_pad_image", &tile_source_exact_split_pad_image);
    module.def("tiling_source_exact_split__Segment_length", &tile_source_exact_split__Segment_length);
    module.def("tiling_source_exact_split__Segment_padded_length", &tile_source_exact_split__Segment_padded_length);
    module.def("tiling_source_exact_split_exact_split_into_segments", &tile_source_exact_split_exact_split_into_segments);
    module.def("tiling_source_exact_split_exact_split_into_regions", &tile_source_exact_split_exact_split_into_regions);
    module.def("tiling_source_exact_split_exact_split_without_padding", &tile_source_exact_split_exact_split_without_padding);
    module.def("tiling_source_exact_split_exact_split", &tile_source_exact_split_exact_split);
    module.def("tiling_source_tiler_Tiler_allow_smaller_tile_size", &tile_source_tiler_Tiler_allow_smaller_tile_size);
    module.def("tiling_source_tiler_Tiler_starting_tile_size", &tile_source_tiler_Tiler_starting_tile_size);
    module.def("tiling_source_tiler_Tiler_split", &tile_source_tiler_Tiler_split);
    module.def("tiling_source_tiler_NoTiling_allow_smaller_tile_size", &tile_source_tiler_NoTiling_allow_smaller_tile_size);
    module.def("tiling_source_tiler_NoTiling_starting_tile_size", &tile_source_tiler_NoTiling_starting_tile_size);
    module.def("tiling_source_tiler_NoTiling_split", &tile_source_tiler_NoTiling_split);
    module.def("tiling_source_tiler_MaxTileSize_init", &tile_source_tiler_MaxTileSize_init);
    module.def("tiling_source_tiler_MaxTileSize_allow_smaller_tile_size", &tile_source_tiler_MaxTileSize_allow_smaller_tile_size);
    module.def("tiling_source_tiler_MaxTileSize_starting_tile_size", &tile_source_tiler_MaxTileSize_starting_tile_size);
    module.def("tiling_source_tiler_ExactTileSize_init", &tile_source_tiler_ExactTileSize_init);
    module.def("tiling_source_tiler_ExactTileSize_allow_smaller_tile_size", &tile_source_tiler_ExactTileSize_allow_smaller_tile_size);
    module.def("tiling_source_tiler_ExactTileSize_starting_tile_size", &tile_source_tiler_ExactTileSize_starting_tile_size);
    module.def("tiling_source_tiler_ExactTileSize_split", &tile_source_tiler_ExactTileSize_split);
    module.def("tiling_source_tiler_BoundedTileSize_init", &tile_source_tiler_BoundedTileSize_init);
    module.def("tiling_source_tiler_BoundedTileSize_allow_smaller_tile_size", &tile_source_tiler_BoundedTileSize_allow_smaller_tile_size);
    module.def("tiling_source_tiler_BoundedTileSize_starting_tile_size", &tile_source_tiler_BoundedTileSize_starting_tile_size);
    module.def("tiling_source_tiler_BoundedTileSize_split", &tile_source_tiler_BoundedTileSize_split);
}
