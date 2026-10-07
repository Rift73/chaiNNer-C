// Fixed native tile state machines. See generate_tiling_cpp.py and reference_tiling/sources.json.
O tile_installed_auto_split_auto_split(const O &globals, O v_img, O v_upscale, O v_tiler, O v_overlap);
O tile_installed_auto_split_exact_split(const O &globals, O v_img, O v_upscale, O v_starting_tile_size, O v_split_tile_size, O v_overlap);
O tile_installed_auto_split_max_split(const O &globals, O v_img, O v_upscale, O v_starting_tile_size, O v_split_tile_size, O v_overlap);
O tile_installed_exact_split_pad_image(const O &globals, O v_img, O v_min_size);
O tile_installed_exact_split__Segment_length(const O &globals, const O &self);
O tile_installed_exact_split__Segment_padded_length(const O &globals, const O &self);
O tile_installed_exact_split_exact_split_into_segments(const O &globals, O v_length, O v_exact, O v_overlap);
O tile_installed_exact_split_exact_split_into_regions(const O &globals, O v_w, O v_h, O v_exact_w, O v_exact_h, O v_overlap);
O tile_installed_exact_split_exact_split_without_padding(const O &globals, O v_img, O v_exact_size, O v_upscale, O v_overlap);
O tile_installed_exact_split_exact_split(const O &globals, O v_img, O v_exact_size, O v_upscale, O v_overlap);
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
O tile_installed_auto_split_auto_split(const O &globals, O v_img, O v_upscale, O v_tiler, O v_overlap) {
    O v_c;
    O v_h;
    O v_split;
    O v_w;
    (void)globals;
    (void)v_img;
    (void)v_upscale;
    (void)v_tiler;
    (void)v_overlap;
    O cn_3 = unpack(([&]() -> O { O cn_1 = item(globals, py::str("get_h_w_c")); O cn_2 = local(v_img, "img"); return cn_1(cn_2); }()), 3);
    v_h = item(cn_3, py::int_(0));
    v_w = item(cn_3, py::int_(1));
    v_c = item(cn_3, py::int_(2));
    v_split = (truth(([&]() -> O { O cn_4 = attr(local(v_tiler, "tiler"), "allow_smaller_tile_size"); return cn_4(); }())) ? O(item(globals, py::str("_max_split"))) : O(item(globals, py::str("_exact_split"))));
    return ([&]() -> O { O cn_9 = local(v_split, "split"); O cn_10 = local(v_img, "img"); O cn_11 = local(v_upscale, "upscale"); O cn_12 = ([&]() -> O { O cn_5 = attr(local(v_tiler, "tiler"), "starting_tile_size"); O cn_6 = local(v_w, "w"); O cn_7 = local(v_h, "h"); O cn_8 = local(v_c, "c"); return cn_5(cn_6, cn_7, cn_8); }()); O cn_13 = attr(local(v_tiler, "tiler"), "split"); O cn_14 = local(v_overlap, "overlap"); return cn_9(cn_10, py::arg("upscale") = cn_11, py::arg("starting_tile_size") = cn_12, py::arg("split_tile_size") = cn_13, py::arg("overlap") = cn_14); }());
}
O tile_installed_auto_split_exact_split(const O &globals, O v_img, O v_upscale, O v_starting_tile_size, O v_split_tile_size, O v_overlap) {
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
    O cn_17 = unpack(([&]() -> O { O cn_15 = item(globals, py::str("get_h_w_c")); O cn_16 = local(v_img, "img"); return cn_15(cn_16); }()), 3);
    v_h = item(cn_17, py::int_(0));
    v_w = item(cn_17, py::int_(1));
    v_c = item(cn_17, py::int_(2));
    ([&]() -> O { O cn_22 = attr(item(globals, py::str("logger")), "debug"); O cn_23 = concat_text({py::str("Exact size split image ("), py::str(local(v_w, "w")), py::str("x"), py::str(local(v_h, "h")), py::str("px @ "), py::str(local(v_c, "c")), py::str(") with exact tile size "), py::str(([&]() -> O { O cn_18 = local(v_starting_tile_size, "starting_tile_size"); O cn_19 = py::int_(0); return item(cn_18, cn_19); }())), py::str("x"), py::str(([&]() -> O { O cn_20 = local(v_starting_tile_size, "starting_tile_size"); O cn_21 = py::int_(1); return item(cn_20, cn_21); }())), py::str("px.")}); return cn_22(cn_23); }());
    v_no_split_upscale = py::cpp_function([=](O v_i, O v_r) mutable -> O {
        O v_result;
        v_result = ([&]() -> O { O cn_24 = local(v_upscale, "upscale"); O cn_25 = local(v_i, "i"); O cn_26 = local(v_r, "r"); return cn_24(cn_25, cn_26); }());
        if (truth(([&]() -> O { O cn_27 = builtin("isinstance"); O cn_28 = local(v_result, "result"); O cn_29 = item(globals, py::str("Split")); return cn_27(cn_28, cn_29); }()))) {
            raise_class(item(globals, py::str("_SplitEx")));
        }
        return local(v_result, "result");
    });
    v_MAX_ITER = py::int_(20);
    for (py::handle cn_30 : py::reinterpret_borrow<py::iterable>(([&]() -> O { O cn_31 = builtin("range"); O cn_32 = local(v_MAX_ITER, "MAX_ITER"); return cn_31(cn_32); }()))) {
        v__ = py::reinterpret_borrow<O>(cn_30);
        try {
            v_max_overlap = ([&]() -> O { O cn_35 = ([&]() -> O { O cn_33 = builtin("min"); py::list cn_34; cn_34.attr("extend")(local(v_starting_tile_size, "starting_tile_size")); return cn_33(*py::tuple(cn_34)); }()); O cn_36 = py::int_(4); return binary(cn_35, cn_36, Op::floordiv); }());
            return ([&]() -> O { O cn_40 = item(globals, py::str("exact_split")); O cn_41 = local(v_img, "img"); O cn_42 = local(v_starting_tile_size, "starting_tile_size"); O cn_43 = local(v_no_split_upscale, "no_split_upscale"); O cn_44 = ([&]() -> O { O cn_37 = builtin("min"); O cn_38 = local(v_max_overlap, "max_overlap"); O cn_39 = local(v_overlap, "overlap"); return cn_37(cn_38, cn_39); }()); return cn_40(py::arg("img") = cn_41, py::arg("exact_size") = cn_42, py::arg("upscale") = cn_43, py::arg("overlap") = cn_44); }());
        } catch (const py::error_already_set &error) {
            if (!error.matches(item(globals, py::str("_SplitEx")).ptr())) throw;
            GraphHandledException handled(error);
            v_starting_tile_size = ([&]() -> O { O cn_45 = local(v_split_tile_size, "split_tile_size"); O cn_46 = local(v_starting_tile_size, "starting_tile_size"); return cn_45(cn_46); }());
        }
    }
    raise(PyExc_ValueError, concat_text({py::str("Aborting after "), py::str(local(v_MAX_ITER, "MAX_ITER")), py::str(" splits. Unable to upscale image.")}));
}
O tile_installed_auto_split_max_split(const O &globals, O v_img, O v_upscale, O v_starting_tile_size, O v_split_tile_size, O v_overlap) {
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
    O cn_49 = unpack(([&]() -> O { O cn_47 = item(globals, py::str("get_h_w_c")); O cn_48 = local(v_img, "img"); return cn_47(cn_48); }()), 3);
    v_h = item(cn_49, py::int_(0));
    v_w = item(cn_49, py::int_(1));
    v_c = item(cn_49, py::int_(2));
    v_img_region = ([&]() -> O { O cn_50 = item(globals, py::str("Region")); O cn_51 = py::int_(0); O cn_52 = py::int_(0); O cn_53 = local(v_w, "w"); O cn_54 = local(v_h, "h"); return cn_50(cn_51, cn_52, cn_53, cn_54); }());
    v_max_tile_size = local(v_starting_tile_size, "starting_tile_size");
    ([&]() -> O { O cn_55 = attr(item(globals, py::str("logger")), "debug"); O cn_56 = concat_text({py::str("Auto split image ("), py::str(local(v_w, "w")), py::str("x"), py::str(local(v_h, "h")), py::str("px @ "), py::str(local(v_c, "c")), py::str(") with initial tile size "), py::str(local(v_max_tile_size, "max_tile_size")), py::str(".")}); return cn_55(cn_56); }());
    if ((truth(([&]() -> O { O cn_61 = local(v_w, "w"); O cn_62 = ([&]() -> O { O cn_59 = local(v_max_tile_size, "max_tile_size"); O cn_60 = py::int_(0); return item(cn_59, cn_60); }()); return rich_compare(cn_61, cn_62, Py_LE); }())) && truth(([&]() -> O { O cn_67 = local(v_h, "h"); O cn_68 = ([&]() -> O { O cn_65 = local(v_max_tile_size, "max_tile_size"); O cn_66 = py::int_(1); return item(cn_65, cn_66); }()); return rich_compare(cn_67, cn_68, Py_LE); }())))) {
        v_upscale_result = ([&]() -> O { O cn_69 = local(v_upscale, "upscale"); O cn_70 = local(v_img, "img"); O cn_71 = local(v_img_region, "img_region"); return cn_69(cn_70, cn_71); }());
        if (!(truth(([&]() -> O { O cn_72 = builtin("isinstance"); O cn_73 = local(v_upscale_result, "upscale_result"); O cn_74 = item(globals, py::str("Split")); return cn_72(cn_73, cn_74); }())))) {
            return local(v_upscale_result, "upscale_result");
        }
        v_max_tile_size = ([&]() -> O { O cn_75 = local(v_split_tile_size, "split_tile_size"); O cn_76 = local(v_max_tile_size, "max_tile_size"); return cn_75(cn_76); }());
        ([&]() -> O { O cn_77 = attr(item(globals, py::str("logger")), "warn"); O cn_78 = concat_text({py::str("Unable to upscale the whole image at once. Reduced tile size to "), py::str(local(v_max_tile_size, "max_tile_size")), py::str(".")}); return cn_77(cn_78); }());
    }
    v_start_y = py::int_(0);
    v_result = py::none();
    v_scale = py::int_(0);
    v_out_channels = py::int_(0);
    v_restart = py::bool_(true);
    while (truth(local(v_restart, "restart"))) {
        v_restart = py::bool_(false);
        v_tile_count_x = ([&]() -> O { O cn_83 = attr(item(globals, py::str("math")), "ceil"); O cn_84 = ([&]() -> O { O cn_81 = local(v_w, "w"); O cn_82 = ([&]() -> O { O cn_79 = local(v_max_tile_size, "max_tile_size"); O cn_80 = py::int_(0); return item(cn_79, cn_80); }()); return binary(cn_81, cn_82, Op::div); }()); return cn_83(cn_84); }());
        v_tile_count_y = ([&]() -> O { O cn_89 = attr(item(globals, py::str("math")), "ceil"); O cn_90 = ([&]() -> O { O cn_87 = local(v_h, "h"); O cn_88 = ([&]() -> O { O cn_85 = local(v_max_tile_size, "max_tile_size"); O cn_86 = py::int_(1); return item(cn_85, cn_86); }()); return binary(cn_87, cn_88, Op::div); }()); return cn_89(cn_90); }());
        v_tile_size_x = ([&]() -> O { O cn_93 = attr(item(globals, py::str("math")), "ceil"); O cn_94 = ([&]() -> O { O cn_91 = local(v_w, "w"); O cn_92 = local(v_tile_count_x, "tile_count_x"); return binary(cn_91, cn_92, Op::div); }()); return cn_93(cn_94); }());
        v_tile_size_y = ([&]() -> O { O cn_97 = attr(item(globals, py::str("math")), "ceil"); O cn_98 = ([&]() -> O { O cn_95 = local(v_h, "h"); O cn_96 = local(v_tile_count_y, "tile_count_y"); return binary(cn_95, cn_96, Op::div); }()); return cn_97(cn_98); }());
        ([&]() -> O { O cn_99 = attr(item(globals, py::str("logger")), "debug"); O cn_100 = concat_text({py::str("Currently "), py::str(local(v_tile_count_x, "tile_count_x")), py::str("x"), py::str(local(v_tile_count_y, "tile_count_y")), py::str(" tiles each "), py::str(local(v_tile_size_x, "tile_size_x")), py::str("x"), py::str(local(v_tile_size_y, "tile_size_y")), py::str("px.")}); return cn_99(cn_100); }());
        v_prev_row_result = py::none();
        for (py::handle cn_101 : py::reinterpret_borrow<py::iterable>(([&]() -> O { O cn_102 = builtin("range"); O cn_103 = local(v_tile_count_y, "tile_count_y"); return cn_102(cn_103); }()))) {
            v_y = py::reinterpret_borrow<O>(cn_101);
            if (truth(([&]() -> O { O cn_104 = local(v_y, "y"); O cn_105 = local(v_start_y, "start_y"); return rich_compare(cn_104, cn_105, Py_LT); }()))) {
                continue;
            }
            v_row_result = py::none();
            v_row_overlap = py::none();
            for (py::handle cn_106 : py::reinterpret_borrow<py::iterable>(([&]() -> O { O cn_107 = builtin("range"); O cn_108 = local(v_tile_count_x, "tile_count_x"); return cn_107(cn_108); }()))) {
                v_x = py::reinterpret_borrow<O>(cn_106);
                v_tile = ([&]() -> O { O cn_118 = attr(([&]() -> O { O cn_113 = item(globals, py::str("Region")); O cn_114 = ([&]() -> O { O cn_109 = local(v_x, "x"); O cn_110 = local(v_tile_size_x, "tile_size_x"); return binary(cn_109, cn_110, Op::mul); }()); O cn_115 = ([&]() -> O { O cn_111 = local(v_y, "y"); O cn_112 = local(v_tile_size_y, "tile_size_y"); return binary(cn_111, cn_112, Op::mul); }()); O cn_116 = local(v_tile_size_x, "tile_size_x"); O cn_117 = local(v_tile_size_y, "tile_size_y"); return cn_113(cn_114, cn_115, cn_116, cn_117); }()), "intersect"); O cn_119 = local(v_img_region, "img_region"); return cn_118(cn_119); }());
                v_pad = ([&]() -> O { O cn_122 = attr(([&]() -> O { O cn_120 = attr(local(v_img_region, "img_region"), "child_padding"); O cn_121 = local(v_tile, "tile"); return cn_120(cn_121); }()), "min"); O cn_123 = local(v_overlap, "overlap"); return cn_122(cn_123); }());
                v_padded_tile = ([&]() -> O { O cn_124 = attr(local(v_tile, "tile"), "add_padding"); O cn_125 = local(v_pad, "pad"); return cn_124(cn_125); }());
                v_upscale_result = ([&]() -> O { O cn_128 = local(v_upscale, "upscale"); O cn_129 = ([&]() -> O { O cn_126 = attr(local(v_padded_tile, "padded_tile"), "read_from"); O cn_127 = local(v_img, "img"); return cn_126(cn_127); }()); O cn_130 = local(v_padded_tile, "padded_tile"); return cn_128(cn_129, cn_130); }());
                if (truth(([&]() -> O { O cn_131 = builtin("isinstance"); O cn_132 = local(v_upscale_result, "upscale_result"); O cn_133 = item(globals, py::str("Split")); return cn_131(cn_132, cn_133); }()))) {
                    v_max_tile_size = ([&]() -> O { O cn_134 = local(v_split_tile_size, "split_tile_size"); O cn_135 = local(v_max_tile_size, "max_tile_size"); return cn_134(cn_135); }());
                    v_new_tile_count_y = ([&]() -> O { O cn_140 = attr(item(globals, py::str("math")), "ceil"); O cn_141 = ([&]() -> O { O cn_138 = local(v_h, "h"); O cn_139 = ([&]() -> O { O cn_136 = local(v_max_tile_size, "max_tile_size"); O cn_137 = py::int_(1); return item(cn_136, cn_137); }()); return binary(cn_138, cn_139, Op::div); }()); return cn_140(cn_141); }());
                    v_new_tile_size_y = ([&]() -> O { O cn_144 = attr(item(globals, py::str("math")), "ceil"); O cn_145 = ([&]() -> O { O cn_142 = local(v_h, "h"); O cn_143 = local(v_new_tile_count_y, "new_tile_count_y"); return binary(cn_142, cn_143, Op::div); }()); return cn_144(cn_145); }());
                    v_start_y = ([&]() -> O { O cn_148 = ([&]() -> O { O cn_146 = local(v_y, "y"); O cn_147 = local(v_tile_size_x, "tile_size_x"); return binary(cn_146, cn_147, Op::mul); }()); O cn_149 = local(v_new_tile_size_y, "new_tile_size_y"); return binary(cn_148, cn_149, Op::floordiv); }());
                    ([&]() -> O { O cn_150 = attr(item(globals, py::str("logger")), "debug"); O cn_151 = concat_text({py::str("Split occurred. New tile size is "), py::str(local(v_max_tile_size, "max_tile_size")), py::str(". Starting at row "), py::str(local(v_start_y, "start_y")), py::str(".")}); return cn_150(cn_151); }());
                    if (truth(([&]() -> O { O cn_152 = local(v_result, "result"); O cn_153 = py::none(); return py::bool_(cn_152.ptr() != cn_153.ptr()); }()))) {
                        O cn_156 = ([&]() -> O { O cn_154 = local(v_start_y, "start_y"); O cn_155 = local(v_new_tile_size_y, "new_tile_size_y"); return binary(cn_154, cn_155, Op::mul); }());
                        set_attr(local(v_result, "result"), "offset", cn_156);
                    }
                    v_restart = py::bool_(true);
                    break;
                }
                O cn_159 = unpack(([&]() -> O { O cn_157 = item(globals, py::str("get_h_w_c")); O cn_158 = local(v_upscale_result, "upscale_result"); return cn_157(cn_158); }()), 3);
                v_up_h = item(cn_159, py::int_(0));
                v_up_w = item(cn_159, py::int_(1));
                v_up_c = item(cn_159, py::int_(2));
                v_current_scale = ([&]() -> O { O cn_160 = local(v_up_h, "up_h"); O cn_161 = attr(local(v_padded_tile, "padded_tile"), "height"); return binary(cn_160, cn_161, Op::floordiv); }());
                if (!(truth(([&]() -> O { O cn_162 = local(v_current_scale, "current_scale"); O cn_163 = py::int_(0); return rich_compare(cn_162, cn_163, Py_GT); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
                if (!(truth(([&]() -> O { O cn_168 = ([&]() -> O { O cn_166 = attr(local(v_padded_tile, "padded_tile"), "height"); O cn_167 = local(v_current_scale, "current_scale"); return binary(cn_166, cn_167, Op::mul); }()); O cn_169 = local(v_up_h, "up_h"); return rich_compare(cn_168, cn_169, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
                if (!(truth(([&]() -> O { O cn_174 = ([&]() -> O { O cn_172 = attr(local(v_padded_tile, "padded_tile"), "width"); O cn_173 = local(v_current_scale, "current_scale"); return binary(cn_172, cn_173, Op::mul); }()); O cn_175 = local(v_up_w, "up_w"); return rich_compare(cn_174, cn_175, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
                if (truth(([&]() -> O { O cn_176 = local(v_row_result, "row_result"); O cn_177 = py::none(); return py::bool_(cn_176.ptr() == cn_177.ptr()); }()))) {
                    v_scale = local(v_current_scale, "current_scale");
                    v_out_channels = local(v_up_c, "up_c");
                    v_row_result = ([&]() -> O { O cn_182 = item(globals, py::str("TileBlender")); O cn_183 = ([&]() -> O { O cn_178 = local(v_w, "w"); O cn_179 = local(v_scale, "scale"); return binary(cn_178, cn_179, Op::mul); }()); O cn_184 = ([&]() -> O { O cn_180 = attr(local(v_padded_tile, "padded_tile"), "height"); O cn_181 = local(v_scale, "scale"); return binary(cn_180, cn_181, Op::mul); }()); O cn_185 = local(v_out_channels, "out_channels"); O cn_186 = attr(item(globals, py::str("BlendDirection")), "X"); O cn_187 = item(globals, py::str("half_sin_blend_fn")); O cn_188 = local(v_prev_row_result, "prev_row_result"); return cn_182(py::arg("width") = cn_183, py::arg("height") = cn_184, py::arg("channels") = cn_185, py::arg("direction") = cn_186, py::arg("blend_fn") = cn_187, py::arg("_prev") = cn_188); }());
                    v_prev_row_result = local(v_row_result, "row_result");
                    v_row_overlap = ([&]() -> O { O cn_193 = item(globals, py::str("TileOverlap")); O cn_194 = ([&]() -> O { O cn_189 = attr(local(v_pad, "pad"), "top"); O cn_190 = local(v_scale, "scale"); return binary(cn_189, cn_190, Op::mul); }()); O cn_195 = ([&]() -> O { O cn_191 = attr(local(v_pad, "pad"), "bottom"); O cn_192 = local(v_scale, "scale"); return binary(cn_191, cn_192, Op::mul); }()); return cn_193(cn_194, cn_195); }());
                }
                if (!(truth(([&]() -> O { O cn_196 = local(v_current_scale, "current_scale"); O cn_197 = local(v_scale, "scale"); return rich_compare(cn_196, cn_197, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
                ([&]() -> O { O cn_205 = attr(local(v_row_result, "row_result"), "add_tile"); O cn_206 = local(v_upscale_result, "upscale_result"); O cn_207 = ([&]() -> O { O cn_202 = item(globals, py::str("TileOverlap")); O cn_203 = ([&]() -> O { O cn_198 = attr(local(v_pad, "pad"), "left"); O cn_199 = local(v_scale, "scale"); return binary(cn_198, cn_199, Op::mul); }()); O cn_204 = ([&]() -> O { O cn_200 = attr(local(v_pad, "pad"), "right"); O cn_201 = local(v_scale, "scale"); return binary(cn_200, cn_201, Op::mul); }()); return cn_202(cn_203, cn_204); }()); return cn_205(cn_206, cn_207); }());
            }
            if (truth(local(v_restart, "restart"))) {
                break;
            }
            if (!(truth(([&]() -> O { O cn_208 = local(v_row_result, "row_result"); O cn_209 = py::none(); return py::bool_(cn_208.ptr() != cn_209.ptr()); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            if (!(truth(([&]() -> O { O cn_210 = local(v_row_overlap, "row_overlap"); O cn_211 = py::none(); return py::bool_(cn_210.ptr() != cn_211.ptr()); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            if (truth(([&]() -> O { O cn_212 = local(v_result, "result"); O cn_213 = py::none(); return py::bool_(cn_212.ptr() == cn_213.ptr()); }()))) {
                v_result = ([&]() -> O { O cn_218 = item(globals, py::str("TileBlender")); O cn_219 = ([&]() -> O { O cn_214 = local(v_w, "w"); O cn_215 = local(v_scale, "scale"); return binary(cn_214, cn_215, Op::mul); }()); O cn_220 = ([&]() -> O { O cn_216 = local(v_h, "h"); O cn_217 = local(v_scale, "scale"); return binary(cn_216, cn_217, Op::mul); }()); O cn_221 = local(v_out_channels, "out_channels"); O cn_222 = attr(item(globals, py::str("BlendDirection")), "Y"); O cn_223 = item(globals, py::str("half_sin_blend_fn")); return cn_218(py::arg("width") = cn_219, py::arg("height") = cn_220, py::arg("channels") = cn_221, py::arg("direction") = cn_222, py::arg("blend_fn") = cn_223); }());
            }
            ([&]() -> O { O cn_225 = attr(local(v_result, "result"), "add_tile"); O cn_226 = ([&]() -> O { O cn_224 = attr(local(v_row_result, "row_result"), "get_result"); return cn_224(); }()); O cn_227 = local(v_row_overlap, "row_overlap"); return cn_225(cn_226, cn_227); }());
        }
    }
    if (!(truth(([&]() -> O { O cn_228 = local(v_result, "result"); O cn_229 = py::none(); return py::bool_(cn_228.ptr() != cn_229.ptr()); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
    return ([&]() -> O { O cn_230 = attr(local(v_result, "result"), "get_result"); return cn_230(); }());
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
    O cn_233 = unpack(([&]() -> O { O cn_231 = item(globals, py::str("get_h_w_c")); O cn_232 = local(v_img, "img"); return cn_231(cn_232); }()), 3);
    v_h = item(cn_233, py::int_(0));
    v_w = item(cn_233, py::int_(1));
    v__ = item(cn_233, py::int_(2));
    O cn_234 = unpack(local(v_min_size, "min_size"), 2);
    v_min_w = item(cn_234, py::int_(0));
    v_min_h = item(cn_234, py::int_(1));
    v_x = ([&]() -> O { O cn_240 = ([&]() -> O { O cn_237 = builtin("max"); O cn_238 = py::int_(0); O cn_239 = ([&]() -> O { O cn_235 = local(v_min_w, "min_w"); O cn_236 = local(v_w, "w"); return binary(cn_235, cn_236, Op::sub); }()); return cn_237(cn_238, cn_239); }()); O cn_241 = py::int_(2); return binary(cn_240, cn_241, Op::div); }());
    v_y = ([&]() -> O { O cn_247 = ([&]() -> O { O cn_244 = builtin("max"); O cn_245 = py::int_(0); O cn_246 = ([&]() -> O { O cn_242 = local(v_min_h, "min_h"); O cn_243 = local(v_h, "h"); return binary(cn_242, cn_243, Op::sub); }()); return cn_244(cn_245, cn_246); }()); O cn_248 = py::int_(2); return binary(cn_247, cn_248, Op::div); }());
    v_padding = ([&]() -> O { O cn_257 = item(globals, py::str("Padding")); O cn_258 = ([&]() -> O { O cn_249 = attr(item(globals, py::str("math")), "floor"); O cn_250 = local(v_y, "y"); return cn_249(cn_250); }()); O cn_259 = ([&]() -> O { O cn_251 = attr(item(globals, py::str("math")), "floor"); O cn_252 = local(v_x, "x"); return cn_251(cn_252); }()); O cn_260 = ([&]() -> O { O cn_253 = attr(item(globals, py::str("math")), "ceil"); O cn_254 = local(v_y, "y"); return cn_253(cn_254); }()); O cn_261 = ([&]() -> O { O cn_255 = attr(item(globals, py::str("math")), "ceil"); O cn_256 = local(v_x, "x"); return cn_255(cn_256); }()); return cn_257(cn_258, cn_259, cn_260, cn_261); }());
    return make_tuple({([&]() -> O { O cn_262 = item(globals, py::str("create_border")); O cn_263 = local(v_img, "img"); O cn_264 = attr(item(globals, py::str("BorderType")), "REFLECT_MIRROR"); O cn_265 = local(v_padding, "padding"); return cn_262(cn_263, cn_264, cn_265); }()), local(v_padding, "padding")});
}
O tile_installed_exact_split__Segment_length(const O &globals, const O &self) {
    (void)globals;
    (void)self;
    return ([&]() -> O { O cn_266 = attr(self, "end"); O cn_267 = attr(self, "start"); return binary(cn_266, cn_267, Op::sub); }());
}
O tile_installed_exact_split__Segment_padded_length(const O &globals, const O &self) {
    (void)globals;
    (void)self;
    return ([&]() -> O { O cn_272 = ([&]() -> O { O cn_268 = attr(self, "end"); O cn_269 = attr(self, "end_padding"); return binary(cn_268, cn_269, Op::add); }()); O cn_273 = ([&]() -> O { O cn_270 = attr(self, "start"); O cn_271 = attr(self, "start_padding"); return binary(cn_270, cn_271, Op::sub); }()); return binary(cn_272, cn_273, Op::sub); }());
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
    if (truth(([&]() -> O { O cn_274 = local(v_length, "length"); O cn_275 = local(v_exact, "exact"); return rich_compare(cn_274, cn_275, Py_EQ); }()))) {
        return make_list({([&]() -> O { O cn_276 = item(globals, py::str("_Segment")); O cn_277 = py::int_(0); O cn_278 = local(v_exact, "exact"); O cn_279 = py::int_(0); O cn_280 = py::int_(0); return cn_276(cn_277, cn_278, cn_279, cn_280); }())});
    }
    if (!(truth(([&]() -> O { O cn_281 = local(v_length, "length"); O cn_282 = local(v_exact, "exact"); return rich_compare(cn_281, cn_282, Py_GT); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
    if (!(truth(([&]() -> O { O cn_287 = local(v_exact, "exact"); O cn_288 = ([&]() -> O { O cn_285 = local(v_overlap, "overlap"); O cn_286 = py::int_(2); return binary(cn_285, cn_286, Op::mul); }()); return rich_compare(cn_287, cn_288, Py_GT); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
    v_result = make_list({});
    v_add = py::cpp_function([=](O v_s) mutable -> O {
        if (!(truth(([&]() -> O { O cn_289 = attr(local(v_s, "s"), "padded_length"); O cn_290 = local(v_exact, "exact"); return rich_compare(cn_289, cn_290, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
        ([&]() -> O { O cn_291 = attr(local(v_result, "result"), "append"); O cn_292 = local(v_s, "s"); return cn_291(cn_292); }());
        return py::none();
    });
    ([&]() -> O { O cn_300 = local(v_add, "add"); O cn_301 = ([&]() -> O { O cn_295 = item(globals, py::str("_Segment")); O cn_296 = py::int_(0); O cn_297 = ([&]() -> O { O cn_293 = local(v_exact, "exact"); O cn_294 = local(v_overlap, "overlap"); return binary(cn_293, cn_294, Op::sub); }()); O cn_298 = py::int_(0); O cn_299 = local(v_overlap, "overlap"); return cn_295(cn_296, cn_297, cn_298, cn_299); }()); return cn_300(cn_301); }());
    while (truth(([&]() -> O { O cn_306 = attr(([&]() -> O { O cn_304 = local(v_result, "result"); O cn_305 = negative(py::int_(1)); return item(cn_304, cn_305); }()), "end"); O cn_307 = local(v_length, "length"); return rich_compare(cn_306, cn_307, Py_LT); }()))) {
        v_start_padding = local(v_overlap, "overlap");
        v_start = attr(([&]() -> O { O cn_308 = local(v_result, "result"); O cn_309 = negative(py::int_(1)); return item(cn_308, cn_309); }()), "end");
        v_end = ([&]() -> O { O cn_314 = ([&]() -> O { O cn_310 = local(v_start, "start"); O cn_311 = local(v_exact, "exact"); return binary(cn_310, cn_311, Op::add); }()); O cn_315 = ([&]() -> O { O cn_312 = local(v_overlap, "overlap"); O cn_313 = py::int_(2); return binary(cn_312, cn_313, Op::mul); }()); return binary(cn_314, cn_315, Op::sub); }());
        v_end_padding = local(v_overlap, "overlap");
        if (truth(([&]() -> O { O cn_320 = ([&]() -> O { O cn_318 = local(v_end, "end"); O cn_319 = local(v_end_padding, "end_padding"); return binary(cn_318, cn_319, Op::add); }()); O cn_321 = local(v_length, "length"); return rich_compare(cn_320, cn_321, Py_GE); }()))) {
            v_end_padding = py::int_(0);
            v_end = local(v_length, "length");
            v_start_padding = ([&]() -> O { O cn_324 = local(v_exact, "exact"); O cn_325 = ([&]() -> O { O cn_322 = local(v_end, "end"); O cn_323 = local(v_start, "start"); return binary(cn_322, cn_323, Op::sub); }()); return binary(cn_324, cn_325, Op::sub); }());
        }
        ([&]() -> O { O cn_331 = local(v_add, "add"); O cn_332 = ([&]() -> O { O cn_326 = item(globals, py::str("_Segment")); O cn_327 = local(v_start, "start"); O cn_328 = local(v_end, "end"); O cn_329 = local(v_start_padding, "start_padding"); O cn_330 = local(v_end_padding, "end_padding"); return cn_326(cn_327, cn_328, cn_329, cn_330); }()); return cn_331(cn_332); }());
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
    v_x_segments = ([&]() -> O { O cn_333 = item(globals, py::str("_exact_split_into_segments")); O cn_334 = local(v_w, "w"); O cn_335 = local(v_exact_w, "exact_w"); O cn_336 = local(v_overlap, "overlap"); return cn_333(cn_334, cn_335, cn_336); }());
    v_y_segments = ([&]() -> O { O cn_337 = item(globals, py::str("_exact_split_into_segments")); O cn_338 = local(v_h, "h"); O cn_339 = local(v_exact_h, "exact_h"); O cn_340 = local(v_overlap, "overlap"); return cn_337(cn_338, cn_339, cn_340); }());
    ([&]() -> O { O cn_341 = attr(item(globals, py::str("logger")), "info"); O cn_342 = concat_text({py::str("Image is split into "), py::str(py::int_(py::len(local(v_x_segments, "x_segments")))), py::str("x"), py::str(py::int_(py::len(local(v_y_segments, "y_segments")))), py::str(" tiles each exactly "), py::str(local(v_exact_w, "exact_w")), py::str("x"), py::str(local(v_exact_h, "exact_h")), py::str("px.")}); return cn_341(cn_342); }());
    v_result = make_list({});
    for (py::handle cn_343 : py::reinterpret_borrow<py::iterable>(local(v_y_segments, "y_segments"))) {
        v_y = py::reinterpret_borrow<O>(cn_343);
        v_row = make_list({});
        for (py::handle cn_344 : py::reinterpret_borrow<py::iterable>(local(v_x_segments, "x_segments"))) {
            v_x = py::reinterpret_borrow<O>(cn_344);
            ([&]() -> O { O cn_355 = attr(local(v_row, "row"), "append"); O cn_356 = make_tuple({([&]() -> O { O cn_345 = item(globals, py::str("Region")); O cn_346 = attr(local(v_x, "x"), "start"); O cn_347 = attr(local(v_y, "y"), "start"); O cn_348 = attr(local(v_x, "x"), "length"); O cn_349 = attr(local(v_y, "y"), "length"); return cn_345(cn_346, cn_347, cn_348, cn_349); }()), ([&]() -> O { O cn_350 = item(globals, py::str("Padding")); O cn_351 = attr(local(v_y, "y"), "start_padding"); O cn_352 = attr(local(v_x, "x"), "end_padding"); O cn_353 = attr(local(v_y, "y"), "end_padding"); O cn_354 = attr(local(v_x, "x"), "start_padding"); return cn_350(cn_351, cn_352, cn_353, cn_354); }())}); return cn_355(cn_356); }());
        }
        ([&]() -> O { O cn_357 = attr(local(v_result, "result"), "append"); O cn_358 = local(v_row, "row"); return cn_357(cn_358); }());
    }
    return local(v_result, "result");
}
O tile_installed_exact_split_exact_split_without_padding(const O &globals, O v_img, O v_exact_size, O v_upscale, O v_overlap) {
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
    O cn_361 = unpack(([&]() -> O { O cn_359 = item(globals, py::str("get_h_w_c")); O cn_360 = local(v_img, "img"); return cn_359(cn_360); }()), 3);
    v_h = item(cn_361, py::int_(0));
    v_w = item(cn_361, py::int_(1));
    v__ = item(cn_361, py::int_(2));
    O cn_362 = unpack(local(v_exact_size, "exact_size"), 2);
    v_exact_w = item(cn_362, py::int_(0));
    v_exact_h = item(cn_362, py::int_(1));
    if (!((truth(([&]() -> O { O cn_363 = local(v_w, "w"); O cn_364 = local(v_exact_w, "exact_w"); return rich_compare(cn_363, cn_364, Py_GE); }())) && truth(([&]() -> O { O cn_365 = local(v_h, "h"); O cn_366 = local(v_exact_h, "exact_h"); return rich_compare(cn_365, cn_366, Py_GE); }()))))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
    if (truth(([&]() -> O { O cn_367 = make_tuple({local(v_w, "w"), local(v_h, "h")}); O cn_368 = local(v_exact_size, "exact_size"); return rich_compare(cn_367, cn_368, Py_EQ); }()))) {
        return ([&]() -> O { O cn_374 = local(v_upscale, "upscale"); O cn_375 = local(v_img, "img"); O cn_376 = ([&]() -> O { O cn_369 = item(globals, py::str("Region")); O cn_370 = py::int_(0); O cn_371 = py::int_(0); O cn_372 = local(v_w, "w"); O cn_373 = local(v_h, "h"); return cn_369(cn_370, cn_371, cn_372, cn_373); }()); return cn_374(cn_375, cn_376); }());
    }
    v_result = py::none();
    v_scale = py::int_(0);
    v_out_channels = py::int_(0);
    v_regions = ([&]() -> O { O cn_377 = item(globals, py::str("_exact_split_into_regions")); O cn_378 = local(v_w, "w"); O cn_379 = local(v_h, "h"); O cn_380 = local(v_exact_w, "exact_w"); O cn_381 = local(v_exact_h, "exact_h"); O cn_382 = local(v_overlap, "overlap"); return cn_377(cn_378, cn_379, cn_380, cn_381, cn_382); }());
    for (py::handle cn_383 : py::reinterpret_borrow<py::iterable>(local(v_regions, "regions"))) {
        v_row = py::reinterpret_borrow<O>(cn_383);
        v_row_result = py::none();
        v_row_overlap = py::none();
        for (py::handle cn_384 : py::reinterpret_borrow<py::iterable>(local(v_row, "row"))) {
            O cn_385 = unpack(py::reinterpret_borrow<O>(cn_384), 2);
            v_tile = item(cn_385, py::int_(0));
            v_pad = item(cn_385, py::int_(1));
            v_padded_tile = ([&]() -> O { O cn_386 = attr(local(v_tile, "tile"), "add_padding"); O cn_387 = local(v_pad, "pad"); return cn_386(cn_387); }());
            if (!(truth(([&]() -> O { O cn_388 = attr(local(v_padded_tile, "padded_tile"), "size"); O cn_389 = local(v_exact_size, "exact_size"); return rich_compare(cn_388, cn_389, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            v_upscale_result = ([&]() -> O { O cn_392 = local(v_upscale, "upscale"); O cn_393 = ([&]() -> O { O cn_390 = attr(local(v_padded_tile, "padded_tile"), "read_from"); O cn_391 = local(v_img, "img"); return cn_390(cn_391); }()); O cn_394 = local(v_padded_tile, "padded_tile"); return cn_392(cn_393, cn_394); }());
            O cn_397 = unpack(([&]() -> O { O cn_395 = item(globals, py::str("get_h_w_c")); O cn_396 = local(v_upscale_result, "upscale_result"); return cn_395(cn_396); }()), 3);
            v_up_h = item(cn_397, py::int_(0));
            v_up_w = item(cn_397, py::int_(1));
            v_up_c = item(cn_397, py::int_(2));
            v_current_scale = ([&]() -> O { O cn_398 = local(v_up_h, "up_h"); O cn_399 = local(v_exact_h, "exact_h"); return binary(cn_398, cn_399, Op::floordiv); }());
            if (!(truth(([&]() -> O { O cn_400 = local(v_current_scale, "current_scale"); O cn_401 = py::int_(0); return rich_compare(cn_400, cn_401, Py_GT); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            if (!(truth(([&]() -> O { O cn_406 = ([&]() -> O { O cn_404 = local(v_exact_h, "exact_h"); O cn_405 = local(v_current_scale, "current_scale"); return binary(cn_404, cn_405, Op::mul); }()); O cn_407 = local(v_up_h, "up_h"); return rich_compare(cn_406, cn_407, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            if (!(truth(([&]() -> O { O cn_412 = ([&]() -> O { O cn_410 = local(v_exact_w, "exact_w"); O cn_411 = local(v_current_scale, "current_scale"); return binary(cn_410, cn_411, Op::mul); }()); O cn_413 = local(v_up_w, "up_w"); return rich_compare(cn_412, cn_413, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            if (truth(([&]() -> O { O cn_414 = local(v_row_result, "row_result"); O cn_415 = py::none(); return py::bool_(cn_414.ptr() == cn_415.ptr()); }()))) {
                v_scale = local(v_current_scale, "current_scale");
                v_out_channels = local(v_up_c, "up_c");
                v_row_result = ([&]() -> O { O cn_420 = item(globals, py::str("TileBlender")); O cn_421 = ([&]() -> O { O cn_416 = local(v_w, "w"); O cn_417 = local(v_scale, "scale"); return binary(cn_416, cn_417, Op::mul); }()); O cn_422 = ([&]() -> O { O cn_418 = local(v_exact_h, "exact_h"); O cn_419 = local(v_scale, "scale"); return binary(cn_418, cn_419, Op::mul); }()); O cn_423 = local(v_out_channels, "out_channels"); O cn_424 = attr(item(globals, py::str("BlendDirection")), "X"); O cn_425 = item(globals, py::str("half_sin_blend_fn")); return cn_420(py::arg("width") = cn_421, py::arg("height") = cn_422, py::arg("channels") = cn_423, py::arg("direction") = cn_424, py::arg("blend_fn") = cn_425); }());
                v_row_overlap = ([&]() -> O { O cn_430 = item(globals, py::str("TileOverlap")); O cn_431 = ([&]() -> O { O cn_426 = attr(local(v_pad, "pad"), "top"); O cn_427 = local(v_scale, "scale"); return binary(cn_426, cn_427, Op::mul); }()); O cn_432 = ([&]() -> O { O cn_428 = attr(local(v_pad, "pad"), "bottom"); O cn_429 = local(v_scale, "scale"); return binary(cn_428, cn_429, Op::mul); }()); return cn_430(cn_431, cn_432); }());
            }
            if (!(truth(([&]() -> O { O cn_433 = local(v_current_scale, "current_scale"); O cn_434 = local(v_scale, "scale"); return rich_compare(cn_433, cn_434, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            ([&]() -> O { O cn_442 = attr(local(v_row_result, "row_result"), "add_tile"); O cn_443 = local(v_upscale_result, "upscale_result"); O cn_444 = ([&]() -> O { O cn_439 = item(globals, py::str("TileOverlap")); O cn_440 = ([&]() -> O { O cn_435 = attr(local(v_pad, "pad"), "left"); O cn_436 = local(v_scale, "scale"); return binary(cn_435, cn_436, Op::mul); }()); O cn_441 = ([&]() -> O { O cn_437 = attr(local(v_pad, "pad"), "right"); O cn_438 = local(v_scale, "scale"); return binary(cn_437, cn_438, Op::mul); }()); return cn_439(cn_440, cn_441); }()); return cn_442(cn_443, cn_444); }());
        }
        if (!(truth(([&]() -> O { O cn_445 = local(v_row_result, "row_result"); O cn_446 = py::none(); return py::bool_(cn_445.ptr() != cn_446.ptr()); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
        if (!(truth(([&]() -> O { O cn_447 = local(v_row_overlap, "row_overlap"); O cn_448 = py::none(); return py::bool_(cn_447.ptr() != cn_448.ptr()); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
        if (truth(([&]() -> O { O cn_449 = local(v_result, "result"); O cn_450 = py::none(); return py::bool_(cn_449.ptr() == cn_450.ptr()); }()))) {
            v_result = ([&]() -> O { O cn_455 = item(globals, py::str("TileBlender")); O cn_456 = ([&]() -> O { O cn_451 = local(v_w, "w"); O cn_452 = local(v_scale, "scale"); return binary(cn_451, cn_452, Op::mul); }()); O cn_457 = ([&]() -> O { O cn_453 = local(v_h, "h"); O cn_454 = local(v_scale, "scale"); return binary(cn_453, cn_454, Op::mul); }()); O cn_458 = local(v_out_channels, "out_channels"); O cn_459 = attr(item(globals, py::str("BlendDirection")), "Y"); O cn_460 = item(globals, py::str("half_sin_blend_fn")); return cn_455(py::arg("width") = cn_456, py::arg("height") = cn_457, py::arg("channels") = cn_458, py::arg("direction") = cn_459, py::arg("blend_fn") = cn_460); }());
        }
        ([&]() -> O { O cn_462 = attr(local(v_result, "result"), "add_tile"); O cn_463 = ([&]() -> O { O cn_461 = attr(local(v_row_result, "row_result"), "get_result"); return cn_461(); }()); O cn_464 = local(v_row_overlap, "row_overlap"); return cn_462(cn_463, cn_464); }());
    }
    if (!(truth(([&]() -> O { O cn_465 = local(v_result, "result"); O cn_466 = py::none(); return py::bool_(cn_465.ptr() != cn_466.ptr()); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
    return ([&]() -> O { O cn_467 = attr(local(v_result, "result"), "get_result"); return cn_467(); }());
}
O tile_installed_exact_split_exact_split(const O &globals, O v_img, O v_exact_size, O v_upscale, O v_overlap) {
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
    O cn_471 = unpack(([&]() -> O { O cn_468 = item(globals, py::str("_pad_image")); O cn_469 = local(v_img, "img"); O cn_470 = local(v_exact_size, "exact_size"); return cn_468(cn_469, cn_470); }()), 2);
    v_img = item(cn_471, py::int_(0));
    v_base_padding = item(cn_471, py::int_(1));
    O cn_474 = unpack(([&]() -> O { O cn_472 = item(globals, py::str("get_h_w_c")); O cn_473 = local(v_img, "img"); return cn_472(cn_473); }()), 3);
    v_h = item(cn_474, py::int_(0));
    v_w = item(cn_474, py::int_(1));
    v__ = item(cn_474, py::int_(2));
    v_result = ([&]() -> O { O cn_475 = item(globals, py::str("_exact_split_without_padding")); O cn_476 = local(v_img, "img"); O cn_477 = local(v_exact_size, "exact_size"); O cn_478 = local(v_upscale, "upscale"); O cn_479 = local(v_overlap, "overlap"); return cn_475(cn_476, cn_477, cn_478, cn_479); }());
    v_scale = ([&]() -> O { O cn_484 = ([&]() -> O { O cn_482 = ([&]() -> O { O cn_480 = item(globals, py::str("get_h_w_c")); O cn_481 = local(v_result, "result"); return cn_480(cn_481); }()); O cn_483 = py::int_(0); return item(cn_482, cn_483); }()); O cn_485 = local(v_h, "h"); return binary(cn_484, cn_485, Op::floordiv); }());
    if (truth(attr(local(v_base_padding, "base_padding"), "empty"))) {
        return local(v_result, "result");
    }
    return ([&]() -> O { O cn_495 = attr(([&]() -> O { O cn_493 = attr(([&]() -> O { O cn_491 = attr(([&]() -> O { O cn_486 = item(globals, py::str("Region")); O cn_487 = py::int_(0); O cn_488 = py::int_(0); O cn_489 = local(v_w, "w"); O cn_490 = local(v_h, "h"); return cn_486(cn_487, cn_488, cn_489, cn_490); }()), "remove_padding"); O cn_492 = local(v_base_padding, "base_padding"); return cn_491(cn_492); }()), "scale"); O cn_494 = local(v_scale, "scale"); return cn_493(cn_494); }()), "read_from"); O cn_496 = local(v_result, "result"); return cn_495(cn_496); }());
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
    O cn_497 = unpack(local(v_tile_size, "tile_size"), 2);
    v_w = item(cn_497, py::int_(0));
    v_h = item(cn_497, py::int_(1));
    if (!((truth(([&]() -> O { O cn_498 = local(v_w, "w"); O cn_499 = py::int_(16); return rich_compare(cn_498, cn_499, Py_GE); }())) && truth(([&]() -> O { O cn_500 = local(v_h, "h"); O cn_501 = py::int_(16); return rich_compare(cn_500, cn_501, Py_GE); }()))))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
    return make_tuple({([&]() -> O { O cn_504 = builtin("max"); O cn_505 = py::int_(16); O cn_506 = ([&]() -> O { O cn_502 = local(v_w, "w"); O cn_503 = py::int_(2); return binary(cn_502, cn_503, Op::floordiv); }()); return cn_504(cn_505, cn_506); }()), ([&]() -> O { O cn_509 = builtin("max"); O cn_510 = py::int_(16); O cn_511 = ([&]() -> O { O cn_507 = local(v_h, "h"); O cn_508 = py::int_(2); return binary(cn_507, cn_508, Op::floordiv); }()); return cn_509(cn_510, cn_511); }())});
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
    v_size = ([&]() -> O { O cn_512 = builtin("max"); O cn_513 = local(v_width, "width"); O cn_514 = local(v_height, "height"); return cn_512(cn_513, cn_514); }());
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
    O cn_515 = local(v_tile_size, "tile_size");
    set_attr(self, "tile_size", cn_515);
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
    v_max_tile_size = ([&]() -> O { O cn_520 = builtin("max"); O cn_521 = ([&]() -> O { O cn_516 = local(v_width, "width"); O cn_517 = py::int_(10); return binary(cn_516, cn_517, Op::add); }()); O cn_522 = ([&]() -> O { O cn_518 = local(v_height, "height"); O cn_519 = py::int_(10); return binary(cn_518, cn_519, Op::add); }()); return cn_520(cn_521, cn_522); }());
    v_size = ([&]() -> O { O cn_523 = builtin("min"); O cn_524 = attr(self, "tile_size"); O cn_525 = local(v_max_tile_size, "max_tile_size"); return cn_523(cn_524, cn_525); }());
    return make_tuple({local(v_size, "size"), local(v_size, "size")});
}
O tile_installed_tiler_ExactTileSize_init(const O &globals, const O &self, O v_exact_size) {
    (void)globals;
    (void)self;
    (void)v_exact_size;
    O cn_526 = local(v_exact_size, "exact_size");
    set_attr(self, "exact_size", cn_526);
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
    raise(PyExc_ValueError, concat_text({py::str("Splits are not supported for exact size ("), py::str(([&]() -> O { O cn_527 = attr(self, "exact_size"); O cn_528 = py::int_(0); return item(cn_527, cn_528); }())), py::str("x"), py::str(([&]() -> O { O cn_529 = attr(self, "exact_size"); O cn_530 = py::int_(1); return item(cn_529, cn_530); }())), py::str("px) splitting. This typically means that your machine does not have enough VRAM to run the current model.")}));
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
    O cn_533 = unpack(([&]() -> O { O cn_531 = item(globals, py::str("get_h_w_c")); O cn_532 = local(v_img, "img"); return cn_531(cn_532); }()), 3);
    v_h = item(cn_533, py::int_(0));
    v_w = item(cn_533, py::int_(1));
    v_c = item(cn_533, py::int_(2));
    v_split = (truth(([&]() -> O { O cn_534 = attr(local(v_tiler, "tiler"), "allow_smaller_tile_size"); return cn_534(); }())) ? O(item(globals, py::str("_max_split"))) : O(item(globals, py::str("_exact_split"))));
    return ([&]() -> O { O cn_539 = local(v_split, "split"); O cn_540 = local(v_img, "img"); O cn_541 = local(v_upscale, "upscale"); O cn_542 = ([&]() -> O { O cn_535 = attr(local(v_tiler, "tiler"), "starting_tile_size"); O cn_536 = local(v_w, "w"); O cn_537 = local(v_h, "h"); O cn_538 = local(v_c, "c"); return cn_535(cn_536, cn_537, cn_538); }()); O cn_543 = attr(local(v_tiler, "tiler"), "split"); O cn_544 = local(v_overlap, "overlap"); O cn_545 = local(v_progress, "progress"); return cn_539(cn_540, py::arg("upscale") = cn_541, py::arg("starting_tile_size") = cn_542, py::arg("split_tile_size") = cn_543, py::arg("overlap") = cn_544, py::arg("progress") = cn_545); }());
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
    O cn_548 = unpack(([&]() -> O { O cn_546 = item(globals, py::str("get_h_w_c")); O cn_547 = local(v_img, "img"); return cn_546(cn_547); }()), 3);
    v_h = item(cn_548, py::int_(0));
    v_w = item(cn_548, py::int_(1));
    v_c = item(cn_548, py::int_(2));
    ([&]() -> O { O cn_553 = attr(item(globals, py::str("logger")), "debug"); O cn_554 = py::str("Exact size split image (%dx%dpx @ %d) with exact tile size %dx%dpx."); O cn_555 = local(v_w, "w"); O cn_556 = local(v_h, "h"); O cn_557 = local(v_c, "c"); O cn_558 = ([&]() -> O { O cn_549 = local(v_starting_tile_size, "starting_tile_size"); O cn_550 = py::int_(0); return item(cn_549, cn_550); }()); O cn_559 = ([&]() -> O { O cn_551 = local(v_starting_tile_size, "starting_tile_size"); O cn_552 = py::int_(1); return item(cn_551, cn_552); }()); return cn_553(cn_554, cn_555, cn_556, cn_557, cn_558, cn_559); }());
    v_no_split_upscale = py::cpp_function([=](O v_i, O v_r) mutable -> O {
        O v_result;
        v_result = ([&]() -> O { O cn_560 = local(v_upscale, "upscale"); O cn_561 = local(v_i, "i"); O cn_562 = local(v_r, "r"); return cn_560(cn_561, cn_562); }());
        if (truth(([&]() -> O { O cn_563 = builtin("isinstance"); O cn_564 = local(v_result, "result"); O cn_565 = item(globals, py::str("Split")); return cn_563(cn_564, cn_565); }()))) {
            raise_class(item(globals, py::str("_SplitEx")));
        }
        return local(v_result, "result");
    });
    v_MAX_ITER = py::int_(20);
    for (py::handle cn_566 : py::reinterpret_borrow<py::iterable>(([&]() -> O { O cn_567 = builtin("range"); O cn_568 = local(v_MAX_ITER, "MAX_ITER"); return cn_567(cn_568); }()))) {
        v__ = py::reinterpret_borrow<O>(cn_566);
        try {
            v_max_overlap = ([&]() -> O { O cn_571 = ([&]() -> O { O cn_569 = builtin("min"); py::list cn_570; cn_570.attr("extend")(local(v_starting_tile_size, "starting_tile_size")); return cn_569(*py::tuple(cn_570)); }()); O cn_572 = py::int_(4); return binary(cn_571, cn_572, Op::floordiv); }());
            return ([&]() -> O { O cn_576 = item(globals, py::str("exact_split")); O cn_577 = local(v_img, "img"); O cn_578 = local(v_starting_tile_size, "starting_tile_size"); O cn_579 = local(v_no_split_upscale, "no_split_upscale"); O cn_580 = ([&]() -> O { O cn_573 = builtin("min"); O cn_574 = local(v_max_overlap, "max_overlap"); O cn_575 = local(v_overlap, "overlap"); return cn_573(cn_574, cn_575); }()); O cn_581 = local(v_progress, "progress"); return cn_576(py::arg("img") = cn_577, py::arg("exact_size") = cn_578, py::arg("upscale") = cn_579, py::arg("overlap") = cn_580, py::arg("progress") = cn_581); }());
        } catch (const py::error_already_set &error) {
            if (!error.matches(item(globals, py::str("_SplitEx")).ptr())) throw;
            GraphHandledException handled(error);
            v_starting_tile_size = ([&]() -> O { O cn_582 = local(v_split_tile_size, "split_tile_size"); O cn_583 = local(v_starting_tile_size, "starting_tile_size"); return cn_582(cn_583); }());
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
    O cn_586 = unpack(([&]() -> O { O cn_584 = item(globals, py::str("get_h_w_c")); O cn_585 = local(v_img, "img"); return cn_584(cn_585); }()), 3);
    v_h = item(cn_586, py::int_(0));
    v_w = item(cn_586, py::int_(1));
    v_c = item(cn_586, py::int_(2));
    v_img_region = ([&]() -> O { O cn_587 = item(globals, py::str("Region")); O cn_588 = py::int_(0); O cn_589 = py::int_(0); O cn_590 = local(v_w, "w"); O cn_591 = local(v_h, "h"); return cn_587(cn_588, cn_589, cn_590, cn_591); }());
    v_max_tile_size = local(v_starting_tile_size, "starting_tile_size");
    ([&]() -> O { O cn_592 = attr(item(globals, py::str("logger")), "debug"); O cn_593 = py::str("Auto split image (%dx%dpx @ %d) with initial tile size %s."); O cn_594 = local(v_w, "w"); O cn_595 = local(v_h, "h"); O cn_596 = local(v_c, "c"); O cn_597 = local(v_max_tile_size, "max_tile_size"); return cn_592(cn_593, cn_594, cn_595, cn_596, cn_597); }());
    if ((truth(([&]() -> O { O cn_602 = local(v_w, "w"); O cn_603 = ([&]() -> O { O cn_600 = local(v_max_tile_size, "max_tile_size"); O cn_601 = py::int_(0); return item(cn_600, cn_601); }()); return rich_compare(cn_602, cn_603, Py_LE); }())) && truth(([&]() -> O { O cn_608 = local(v_h, "h"); O cn_609 = ([&]() -> O { O cn_606 = local(v_max_tile_size, "max_tile_size"); O cn_607 = py::int_(1); return item(cn_606, cn_607); }()); return rich_compare(cn_608, cn_609, Py_LE); }())))) {
        v_upscale_result = ([&]() -> O { O cn_610 = local(v_upscale, "upscale"); O cn_611 = local(v_img, "img"); O cn_612 = local(v_img_region, "img_region"); return cn_610(cn_611, cn_612); }());
        if (!(truth(([&]() -> O { O cn_613 = builtin("isinstance"); O cn_614 = local(v_upscale_result, "upscale_result"); O cn_615 = item(globals, py::str("Split")); return cn_613(cn_614, cn_615); }())))) {
            if (truth(([&]() -> O { O cn_616 = local(v_progress, "progress"); O cn_617 = py::none(); return py::bool_(cn_616.ptr() != cn_617.ptr()); }()))) {
                ([&]() -> O { O cn_618 = attr(local(v_progress, "progress"), "set_progress"); O cn_619 = py::float_(1.0); return cn_618(cn_619); }());
            }
            return local(v_upscale_result, "upscale_result");
        }
        v_max_tile_size = ([&]() -> O { O cn_620 = local(v_split_tile_size, "split_tile_size"); O cn_621 = local(v_max_tile_size, "max_tile_size"); return cn_620(cn_621); }());
        ([&]() -> O { O cn_622 = attr(item(globals, py::str("logger")), "warning"); O cn_623 = py::str("Unable to upscale the whole image at once. Reduced tile size to %s."); O cn_624 = local(v_max_tile_size, "max_tile_size"); return cn_622(cn_623, cn_624); }());
    }
    v_start_y = py::int_(0);
    v_result = py::none();
    v_scale = py::int_(0);
    v_out_channels = py::int_(0);
    v_restart = py::bool_(true);
    while (truth(local(v_restart, "restart"))) {
        v_restart = py::bool_(false);
        v_tile_count_x = ([&]() -> O { O cn_629 = attr(item(globals, py::str("math")), "ceil"); O cn_630 = ([&]() -> O { O cn_627 = local(v_w, "w"); O cn_628 = ([&]() -> O { O cn_625 = local(v_max_tile_size, "max_tile_size"); O cn_626 = py::int_(0); return item(cn_625, cn_626); }()); return binary(cn_627, cn_628, Op::div); }()); return cn_629(cn_630); }());
        v_tile_count_y = ([&]() -> O { O cn_635 = attr(item(globals, py::str("math")), "ceil"); O cn_636 = ([&]() -> O { O cn_633 = local(v_h, "h"); O cn_634 = ([&]() -> O { O cn_631 = local(v_max_tile_size, "max_tile_size"); O cn_632 = py::int_(1); return item(cn_631, cn_632); }()); return binary(cn_633, cn_634, Op::div); }()); return cn_635(cn_636); }());
        v_tile_size_x = ([&]() -> O { O cn_639 = attr(item(globals, py::str("math")), "ceil"); O cn_640 = ([&]() -> O { O cn_637 = local(v_w, "w"); O cn_638 = local(v_tile_count_x, "tile_count_x"); return binary(cn_637, cn_638, Op::div); }()); return cn_639(cn_640); }());
        v_tile_size_y = ([&]() -> O { O cn_643 = attr(item(globals, py::str("math")), "ceil"); O cn_644 = ([&]() -> O { O cn_641 = local(v_h, "h"); O cn_642 = local(v_tile_count_y, "tile_count_y"); return binary(cn_641, cn_642, Op::div); }()); return cn_643(cn_644); }());
        v_total_tiles = ([&]() -> O { O cn_645 = local(v_tile_count_x, "tile_count_x"); O cn_646 = local(v_tile_count_y, "tile_count_y"); return binary(cn_645, cn_646, Op::mul); }());
        ([&]() -> O { O cn_647 = attr(item(globals, py::str("logger")), "debug"); O cn_648 = py::str("Currently %dx%d tiles each %dx%dpx."); O cn_649 = local(v_tile_count_x, "tile_count_x"); O cn_650 = local(v_tile_count_y, "tile_count_y"); O cn_651 = local(v_tile_size_x, "tile_size_x"); O cn_652 = local(v_tile_size_y, "tile_size_y"); return cn_647(cn_648, cn_649, cn_650, cn_651, cn_652); }());
        v_prev_row_result = py::none();
        v_tiles_processed = py::int_(0);
        for (py::handle cn_653 : py::reinterpret_borrow<py::iterable>(([&]() -> O { O cn_654 = builtin("range"); O cn_655 = local(v_tile_count_y, "tile_count_y"); return cn_654(cn_655); }()))) {
            v_y = py::reinterpret_borrow<O>(cn_653);
            if (truth(([&]() -> O { O cn_656 = local(v_y, "y"); O cn_657 = local(v_start_y, "start_y"); return rich_compare(cn_656, cn_657, Py_LT); }()))) {
                continue;
            }
            v_row_result = py::none();
            v_row_overlap = py::none();
            for (py::handle cn_658 : py::reinterpret_borrow<py::iterable>(([&]() -> O { O cn_659 = builtin("range"); O cn_660 = local(v_tile_count_x, "tile_count_x"); return cn_659(cn_660); }()))) {
                v_x = py::reinterpret_borrow<O>(cn_658);
                v_tile = ([&]() -> O { O cn_670 = attr(([&]() -> O { O cn_665 = item(globals, py::str("Region")); O cn_666 = ([&]() -> O { O cn_661 = local(v_x, "x"); O cn_662 = local(v_tile_size_x, "tile_size_x"); return binary(cn_661, cn_662, Op::mul); }()); O cn_667 = ([&]() -> O { O cn_663 = local(v_y, "y"); O cn_664 = local(v_tile_size_y, "tile_size_y"); return binary(cn_663, cn_664, Op::mul); }()); O cn_668 = local(v_tile_size_x, "tile_size_x"); O cn_669 = local(v_tile_size_y, "tile_size_y"); return cn_665(cn_666, cn_667, cn_668, cn_669); }()), "intersect"); O cn_671 = local(v_img_region, "img_region"); return cn_670(cn_671); }());
                v_pad = ([&]() -> O { O cn_674 = attr(([&]() -> O { O cn_672 = attr(local(v_img_region, "img_region"), "child_padding"); O cn_673 = local(v_tile, "tile"); return cn_672(cn_673); }()), "min"); O cn_675 = local(v_overlap, "overlap"); return cn_674(cn_675); }());
                v_padded_tile = ([&]() -> O { O cn_676 = attr(local(v_tile, "tile"), "add_padding"); O cn_677 = local(v_pad, "pad"); return cn_676(cn_677); }());
                v_upscale_result = ([&]() -> O { O cn_680 = local(v_upscale, "upscale"); O cn_681 = ([&]() -> O { O cn_678 = attr(local(v_padded_tile, "padded_tile"), "read_from"); O cn_679 = local(v_img, "img"); return cn_678(cn_679); }()); O cn_682 = local(v_padded_tile, "padded_tile"); return cn_680(cn_681, cn_682); }());
                if (truth(([&]() -> O { O cn_683 = builtin("isinstance"); O cn_684 = local(v_upscale_result, "upscale_result"); O cn_685 = item(globals, py::str("Split")); return cn_683(cn_684, cn_685); }()))) {
                    v_max_tile_size = ([&]() -> O { O cn_686 = local(v_split_tile_size, "split_tile_size"); O cn_687 = local(v_max_tile_size, "max_tile_size"); return cn_686(cn_687); }());
                    v_new_tile_count_y = ([&]() -> O { O cn_692 = attr(item(globals, py::str("math")), "ceil"); O cn_693 = ([&]() -> O { O cn_690 = local(v_h, "h"); O cn_691 = ([&]() -> O { O cn_688 = local(v_max_tile_size, "max_tile_size"); O cn_689 = py::int_(1); return item(cn_688, cn_689); }()); return binary(cn_690, cn_691, Op::div); }()); return cn_692(cn_693); }());
                    v_new_tile_size_y = ([&]() -> O { O cn_696 = attr(item(globals, py::str("math")), "ceil"); O cn_697 = ([&]() -> O { O cn_694 = local(v_h, "h"); O cn_695 = local(v_new_tile_count_y, "new_tile_count_y"); return binary(cn_694, cn_695, Op::div); }()); return cn_696(cn_697); }());
                    v_start_y = ([&]() -> O { O cn_700 = ([&]() -> O { O cn_698 = local(v_y, "y"); O cn_699 = local(v_tile_size_x, "tile_size_x"); return binary(cn_698, cn_699, Op::mul); }()); O cn_701 = local(v_new_tile_size_y, "new_tile_size_y"); return binary(cn_700, cn_701, Op::floordiv); }());
                    ([&]() -> O { O cn_702 = attr(item(globals, py::str("logger")), "debug"); O cn_703 = py::str("Split occurred. New tile size is %s. Starting at row %d."); O cn_704 = local(v_max_tile_size, "max_tile_size"); O cn_705 = local(v_start_y, "start_y"); return cn_702(cn_703, cn_704, cn_705); }());
                    if (truth(([&]() -> O { O cn_706 = local(v_result, "result"); O cn_707 = py::none(); return py::bool_(cn_706.ptr() != cn_707.ptr()); }()))) {
                        O cn_710 = ([&]() -> O { O cn_708 = local(v_start_y, "start_y"); O cn_709 = local(v_new_tile_size_y, "new_tile_size_y"); return binary(cn_708, cn_709, Op::mul); }());
                        set_attr(local(v_result, "result"), "offset", cn_710);
                    }
                    v_restart = py::bool_(true);
                    break;
                }
                O cn_713 = unpack(([&]() -> O { O cn_711 = item(globals, py::str("get_h_w_c")); O cn_712 = local(v_upscale_result, "upscale_result"); return cn_711(cn_712); }()), 3);
                v_up_h = item(cn_713, py::int_(0));
                v_up_w = item(cn_713, py::int_(1));
                v_up_c = item(cn_713, py::int_(2));
                v_current_scale = ([&]() -> O { O cn_714 = local(v_up_h, "up_h"); O cn_715 = attr(local(v_padded_tile, "padded_tile"), "height"); return binary(cn_714, cn_715, Op::floordiv); }());
                if (!(truth(([&]() -> O { O cn_716 = local(v_current_scale, "current_scale"); O cn_717 = py::int_(0); return rich_compare(cn_716, cn_717, Py_GT); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
                if (!(truth(([&]() -> O { O cn_722 = ([&]() -> O { O cn_720 = attr(local(v_padded_tile, "padded_tile"), "height"); O cn_721 = local(v_current_scale, "current_scale"); return binary(cn_720, cn_721, Op::mul); }()); O cn_723 = local(v_up_h, "up_h"); return rich_compare(cn_722, cn_723, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
                if (!(truth(([&]() -> O { O cn_728 = ([&]() -> O { O cn_726 = attr(local(v_padded_tile, "padded_tile"), "width"); O cn_727 = local(v_current_scale, "current_scale"); return binary(cn_726, cn_727, Op::mul); }()); O cn_729 = local(v_up_w, "up_w"); return rich_compare(cn_728, cn_729, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
                if (truth(([&]() -> O { O cn_730 = local(v_row_result, "row_result"); O cn_731 = py::none(); return py::bool_(cn_730.ptr() == cn_731.ptr()); }()))) {
                    v_scale = local(v_current_scale, "current_scale");
                    v_out_channels = local(v_up_c, "up_c");
                    v_row_result = ([&]() -> O { O cn_736 = item(globals, py::str("TileBlender")); O cn_737 = ([&]() -> O { O cn_732 = local(v_w, "w"); O cn_733 = local(v_scale, "scale"); return binary(cn_732, cn_733, Op::mul); }()); O cn_738 = ([&]() -> O { O cn_734 = attr(local(v_padded_tile, "padded_tile"), "height"); O cn_735 = local(v_scale, "scale"); return binary(cn_734, cn_735, Op::mul); }()); O cn_739 = local(v_out_channels, "out_channels"); O cn_740 = attr(item(globals, py::str("BlendDirection")), "X"); O cn_741 = item(globals, py::str("half_sin_blend_fn")); O cn_742 = local(v_prev_row_result, "prev_row_result"); return cn_736(py::arg("width") = cn_737, py::arg("height") = cn_738, py::arg("channels") = cn_739, py::arg("direction") = cn_740, py::arg("blend_fn") = cn_741, py::arg("_prev") = cn_742); }());
                    v_prev_row_result = local(v_row_result, "row_result");
                    v_row_overlap = ([&]() -> O { O cn_747 = item(globals, py::str("TileOverlap")); O cn_748 = ([&]() -> O { O cn_743 = attr(local(v_pad, "pad"), "top"); O cn_744 = local(v_scale, "scale"); return binary(cn_743, cn_744, Op::mul); }()); O cn_749 = ([&]() -> O { O cn_745 = attr(local(v_pad, "pad"), "bottom"); O cn_746 = local(v_scale, "scale"); return binary(cn_745, cn_746, Op::mul); }()); return cn_747(cn_748, cn_749); }());
                }
                if (!(truth(([&]() -> O { O cn_750 = local(v_current_scale, "current_scale"); O cn_751 = local(v_scale, "scale"); return rich_compare(cn_750, cn_751, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
                ([&]() -> O { O cn_759 = attr(local(v_row_result, "row_result"), "add_tile"); O cn_760 = local(v_upscale_result, "upscale_result"); O cn_761 = ([&]() -> O { O cn_756 = item(globals, py::str("TileOverlap")); O cn_757 = ([&]() -> O { O cn_752 = attr(local(v_pad, "pad"), "left"); O cn_753 = local(v_scale, "scale"); return binary(cn_752, cn_753, Op::mul); }()); O cn_758 = ([&]() -> O { O cn_754 = attr(local(v_pad, "pad"), "right"); O cn_755 = local(v_scale, "scale"); return binary(cn_754, cn_755, Op::mul); }()); return cn_756(cn_757, cn_758); }()); return cn_759(cn_760, cn_761); }());
                O cn_762 = local(v_tiles_processed, "tiles_processed");
                O cn_763 = py::int_(1);
                O cn_764 = inplace(cn_762, cn_763, Op::add);
                v_tiles_processed = cn_764;
                if (truth(([&]() -> O { O cn_765 = local(v_progress, "progress"); O cn_766 = py::none(); return py::bool_(cn_765.ptr() != cn_766.ptr()); }()))) {
                    ([&]() -> O { O cn_769 = attr(local(v_progress, "progress"), "set_progress"); O cn_770 = ([&]() -> O { O cn_767 = local(v_tiles_processed, "tiles_processed"); O cn_768 = local(v_total_tiles, "total_tiles"); return binary(cn_767, cn_768, Op::div); }()); return cn_769(cn_770); }());
                }
            }
            if (truth(local(v_restart, "restart"))) {
                break;
            }
            if (!(truth(([&]() -> O { O cn_771 = local(v_row_result, "row_result"); O cn_772 = py::none(); return py::bool_(cn_771.ptr() != cn_772.ptr()); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            if (!(truth(([&]() -> O { O cn_773 = local(v_row_overlap, "row_overlap"); O cn_774 = py::none(); return py::bool_(cn_773.ptr() != cn_774.ptr()); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            if (truth(([&]() -> O { O cn_775 = local(v_result, "result"); O cn_776 = py::none(); return py::bool_(cn_775.ptr() == cn_776.ptr()); }()))) {
                v_result = ([&]() -> O { O cn_781 = item(globals, py::str("TileBlender")); O cn_782 = ([&]() -> O { O cn_777 = local(v_w, "w"); O cn_778 = local(v_scale, "scale"); return binary(cn_777, cn_778, Op::mul); }()); O cn_783 = ([&]() -> O { O cn_779 = local(v_h, "h"); O cn_780 = local(v_scale, "scale"); return binary(cn_779, cn_780, Op::mul); }()); O cn_784 = local(v_out_channels, "out_channels"); O cn_785 = attr(item(globals, py::str("BlendDirection")), "Y"); O cn_786 = item(globals, py::str("half_sin_blend_fn")); return cn_781(py::arg("width") = cn_782, py::arg("height") = cn_783, py::arg("channels") = cn_784, py::arg("direction") = cn_785, py::arg("blend_fn") = cn_786); }());
            }
            ([&]() -> O { O cn_788 = attr(local(v_result, "result"), "add_tile"); O cn_789 = ([&]() -> O { O cn_787 = attr(local(v_row_result, "row_result"), "get_result"); return cn_787(); }()); O cn_790 = local(v_row_overlap, "row_overlap"); return cn_788(cn_789, cn_790); }());
        }
    }
    if (!(truth(([&]() -> O { O cn_791 = local(v_result, "result"); O cn_792 = py::none(); return py::bool_(cn_791.ptr() != cn_792.ptr()); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
    return ([&]() -> O { O cn_793 = attr(local(v_result, "result"), "get_result"); return cn_793(); }());
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
    O cn_796 = unpack(([&]() -> O { O cn_794 = item(globals, py::str("get_h_w_c")); O cn_795 = local(v_img, "img"); return cn_794(cn_795); }()), 3);
    v_h = item(cn_796, py::int_(0));
    v_w = item(cn_796, py::int_(1));
    v__ = item(cn_796, py::int_(2));
    O cn_797 = unpack(local(v_min_size, "min_size"), 2);
    v_min_w = item(cn_797, py::int_(0));
    v_min_h = item(cn_797, py::int_(1));
    v_x = ([&]() -> O { O cn_803 = ([&]() -> O { O cn_800 = builtin("max"); O cn_801 = py::int_(0); O cn_802 = ([&]() -> O { O cn_798 = local(v_min_w, "min_w"); O cn_799 = local(v_w, "w"); return binary(cn_798, cn_799, Op::sub); }()); return cn_800(cn_801, cn_802); }()); O cn_804 = py::int_(2); return binary(cn_803, cn_804, Op::div); }());
    v_y = ([&]() -> O { O cn_810 = ([&]() -> O { O cn_807 = builtin("max"); O cn_808 = py::int_(0); O cn_809 = ([&]() -> O { O cn_805 = local(v_min_h, "min_h"); O cn_806 = local(v_h, "h"); return binary(cn_805, cn_806, Op::sub); }()); return cn_807(cn_808, cn_809); }()); O cn_811 = py::int_(2); return binary(cn_810, cn_811, Op::div); }());
    v_padding = ([&]() -> O { O cn_820 = item(globals, py::str("Padding")); O cn_821 = ([&]() -> O { O cn_812 = attr(item(globals, py::str("math")), "floor"); O cn_813 = local(v_y, "y"); return cn_812(cn_813); }()); O cn_822 = ([&]() -> O { O cn_814 = attr(item(globals, py::str("math")), "floor"); O cn_815 = local(v_x, "x"); return cn_814(cn_815); }()); O cn_823 = ([&]() -> O { O cn_816 = attr(item(globals, py::str("math")), "ceil"); O cn_817 = local(v_y, "y"); return cn_816(cn_817); }()); O cn_824 = ([&]() -> O { O cn_818 = attr(item(globals, py::str("math")), "ceil"); O cn_819 = local(v_x, "x"); return cn_818(cn_819); }()); return cn_820(cn_821, cn_822, cn_823, cn_824); }());
    return make_tuple({([&]() -> O { O cn_825 = item(globals, py::str("create_border")); O cn_826 = local(v_img, "img"); O cn_827 = attr(item(globals, py::str("BorderType")), "REFLECT_MIRROR"); O cn_828 = local(v_padding, "padding"); return cn_825(cn_826, cn_827, cn_828); }()), local(v_padding, "padding")});
}
O tile_source_exact_split__Segment_length(const O &globals, const O &self) {
    (void)globals;
    (void)self;
    return ([&]() -> O { O cn_829 = attr(self, "end"); O cn_830 = attr(self, "start"); return binary(cn_829, cn_830, Op::sub); }());
}
O tile_source_exact_split__Segment_padded_length(const O &globals, const O &self) {
    (void)globals;
    (void)self;
    return ([&]() -> O { O cn_835 = ([&]() -> O { O cn_831 = attr(self, "end"); O cn_832 = attr(self, "end_padding"); return binary(cn_831, cn_832, Op::add); }()); O cn_836 = ([&]() -> O { O cn_833 = attr(self, "start"); O cn_834 = attr(self, "start_padding"); return binary(cn_833, cn_834, Op::sub); }()); return binary(cn_835, cn_836, Op::sub); }());
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
    if (truth(([&]() -> O { O cn_837 = local(v_length, "length"); O cn_838 = local(v_exact, "exact"); return rich_compare(cn_837, cn_838, Py_EQ); }()))) {
        return make_list({([&]() -> O { O cn_839 = item(globals, py::str("_Segment")); O cn_840 = py::int_(0); O cn_841 = local(v_exact, "exact"); O cn_842 = py::int_(0); O cn_843 = py::int_(0); return cn_839(cn_840, cn_841, cn_842, cn_843); }())});
    }
    if (!(truth(([&]() -> O { O cn_844 = local(v_length, "length"); O cn_845 = local(v_exact, "exact"); return rich_compare(cn_844, cn_845, Py_GT); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
    if (!(truth(([&]() -> O { O cn_850 = local(v_exact, "exact"); O cn_851 = ([&]() -> O { O cn_848 = local(v_overlap, "overlap"); O cn_849 = py::int_(2); return binary(cn_848, cn_849, Op::mul); }()); return rich_compare(cn_850, cn_851, Py_GT); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
    v_result = make_list({});
    v_add = py::cpp_function([=](O v_s) mutable -> O {
        if (!(truth(([&]() -> O { O cn_852 = attr(local(v_s, "s"), "padded_length"); O cn_853 = local(v_exact, "exact"); return rich_compare(cn_852, cn_853, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
        ([&]() -> O { O cn_854 = attr(local(v_result, "result"), "append"); O cn_855 = local(v_s, "s"); return cn_854(cn_855); }());
        return py::none();
    });
    ([&]() -> O { O cn_863 = local(v_add, "add"); O cn_864 = ([&]() -> O { O cn_858 = item(globals, py::str("_Segment")); O cn_859 = py::int_(0); O cn_860 = ([&]() -> O { O cn_856 = local(v_exact, "exact"); O cn_857 = local(v_overlap, "overlap"); return binary(cn_856, cn_857, Op::sub); }()); O cn_861 = py::int_(0); O cn_862 = local(v_overlap, "overlap"); return cn_858(cn_859, cn_860, cn_861, cn_862); }()); return cn_863(cn_864); }());
    while (truth(([&]() -> O { O cn_869 = attr(([&]() -> O { O cn_867 = local(v_result, "result"); O cn_868 = negative(py::int_(1)); return item(cn_867, cn_868); }()), "end"); O cn_870 = local(v_length, "length"); return rich_compare(cn_869, cn_870, Py_LT); }()))) {
        v_start_padding = local(v_overlap, "overlap");
        v_start = attr(([&]() -> O { O cn_871 = local(v_result, "result"); O cn_872 = negative(py::int_(1)); return item(cn_871, cn_872); }()), "end");
        v_end = ([&]() -> O { O cn_877 = ([&]() -> O { O cn_873 = local(v_start, "start"); O cn_874 = local(v_exact, "exact"); return binary(cn_873, cn_874, Op::add); }()); O cn_878 = ([&]() -> O { O cn_875 = local(v_overlap, "overlap"); O cn_876 = py::int_(2); return binary(cn_875, cn_876, Op::mul); }()); return binary(cn_877, cn_878, Op::sub); }());
        v_end_padding = local(v_overlap, "overlap");
        if (truth(([&]() -> O { O cn_883 = ([&]() -> O { O cn_881 = local(v_end, "end"); O cn_882 = local(v_end_padding, "end_padding"); return binary(cn_881, cn_882, Op::add); }()); O cn_884 = local(v_length, "length"); return rich_compare(cn_883, cn_884, Py_GE); }()))) {
            v_end_padding = py::int_(0);
            v_end = local(v_length, "length");
            v_start_padding = ([&]() -> O { O cn_887 = local(v_exact, "exact"); O cn_888 = ([&]() -> O { O cn_885 = local(v_end, "end"); O cn_886 = local(v_start, "start"); return binary(cn_885, cn_886, Op::sub); }()); return binary(cn_887, cn_888, Op::sub); }());
        }
        ([&]() -> O { O cn_894 = local(v_add, "add"); O cn_895 = ([&]() -> O { O cn_889 = item(globals, py::str("_Segment")); O cn_890 = local(v_start, "start"); O cn_891 = local(v_end, "end"); O cn_892 = local(v_start_padding, "start_padding"); O cn_893 = local(v_end_padding, "end_padding"); return cn_889(cn_890, cn_891, cn_892, cn_893); }()); return cn_894(cn_895); }());
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
    v_x_segments = ([&]() -> O { O cn_896 = item(globals, py::str("_exact_split_into_segments")); O cn_897 = local(v_w, "w"); O cn_898 = local(v_exact_w, "exact_w"); O cn_899 = local(v_overlap, "overlap"); return cn_896(cn_897, cn_898, cn_899); }());
    v_y_segments = ([&]() -> O { O cn_900 = item(globals, py::str("_exact_split_into_segments")); O cn_901 = local(v_h, "h"); O cn_902 = local(v_exact_h, "exact_h"); O cn_903 = local(v_overlap, "overlap"); return cn_900(cn_901, cn_902, cn_903); }());
    ([&]() -> O { O cn_904 = attr(item(globals, py::str("logger")), "info"); O cn_905 = py::str("Image is split into %dx%d tiles each exactly %dx%dpx."); O cn_906 = py::int_(py::len(local(v_x_segments, "x_segments"))); O cn_907 = py::int_(py::len(local(v_y_segments, "y_segments"))); O cn_908 = local(v_exact_w, "exact_w"); O cn_909 = local(v_exact_h, "exact_h"); return cn_904(cn_905, cn_906, cn_907, cn_908, cn_909); }());
    v_result = make_list({});
    for (py::handle cn_910 : py::reinterpret_borrow<py::iterable>(local(v_y_segments, "y_segments"))) {
        v_y = py::reinterpret_borrow<O>(cn_910);
        v_row = make_list({});
        for (py::handle cn_911 : py::reinterpret_borrow<py::iterable>(local(v_x_segments, "x_segments"))) {
            v_x = py::reinterpret_borrow<O>(cn_911);
            ([&]() -> O { O cn_922 = attr(local(v_row, "row"), "append"); O cn_923 = make_tuple({([&]() -> O { O cn_912 = item(globals, py::str("Region")); O cn_913 = attr(local(v_x, "x"), "start"); O cn_914 = attr(local(v_y, "y"), "start"); O cn_915 = attr(local(v_x, "x"), "length"); O cn_916 = attr(local(v_y, "y"), "length"); return cn_912(cn_913, cn_914, cn_915, cn_916); }()), ([&]() -> O { O cn_917 = item(globals, py::str("Padding")); O cn_918 = attr(local(v_y, "y"), "start_padding"); O cn_919 = attr(local(v_x, "x"), "end_padding"); O cn_920 = attr(local(v_y, "y"), "end_padding"); O cn_921 = attr(local(v_x, "x"), "start_padding"); return cn_917(cn_918, cn_919, cn_920, cn_921); }())}); return cn_922(cn_923); }());
        }
        ([&]() -> O { O cn_924 = attr(local(v_result, "result"), "append"); O cn_925 = local(v_row, "row"); return cn_924(cn_925); }());
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
    O cn_928 = unpack(([&]() -> O { O cn_926 = item(globals, py::str("get_h_w_c")); O cn_927 = local(v_img, "img"); return cn_926(cn_927); }()), 3);
    v_h = item(cn_928, py::int_(0));
    v_w = item(cn_928, py::int_(1));
    v__ = item(cn_928, py::int_(2));
    O cn_929 = unpack(local(v_exact_size, "exact_size"), 2);
    v_exact_w = item(cn_929, py::int_(0));
    v_exact_h = item(cn_929, py::int_(1));
    if (!((truth(([&]() -> O { O cn_930 = local(v_w, "w"); O cn_931 = local(v_exact_w, "exact_w"); return rich_compare(cn_930, cn_931, Py_GE); }())) && truth(([&]() -> O { O cn_932 = local(v_h, "h"); O cn_933 = local(v_exact_h, "exact_h"); return rich_compare(cn_932, cn_933, Py_GE); }()))))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
    if (truth(([&]() -> O { O cn_934 = make_tuple({local(v_w, "w"), local(v_h, "h")}); O cn_935 = local(v_exact_size, "exact_size"); return rich_compare(cn_934, cn_935, Py_EQ); }()))) {
        if (truth(([&]() -> O { O cn_936 = local(v_progress, "progress"); O cn_937 = py::none(); return py::bool_(cn_936.ptr() != cn_937.ptr()); }()))) {
            ([&]() -> O { O cn_938 = attr(local(v_progress, "progress"), "set_progress"); O cn_939 = py::float_(1.0); return cn_938(cn_939); }());
        }
        return ([&]() -> O { O cn_945 = local(v_upscale, "upscale"); O cn_946 = local(v_img, "img"); O cn_947 = ([&]() -> O { O cn_940 = item(globals, py::str("Region")); O cn_941 = py::int_(0); O cn_942 = py::int_(0); O cn_943 = local(v_w, "w"); O cn_944 = local(v_h, "h"); return cn_940(cn_941, cn_942, cn_943, cn_944); }()); return cn_945(cn_946, cn_947); }());
    }
    v_result = py::none();
    v_scale = py::int_(0);
    v_out_channels = py::int_(0);
    v_regions = ([&]() -> O { O cn_948 = item(globals, py::str("_exact_split_into_regions")); O cn_949 = local(v_w, "w"); O cn_950 = local(v_h, "h"); O cn_951 = local(v_exact_w, "exact_w"); O cn_952 = local(v_exact_h, "exact_h"); O cn_953 = local(v_overlap, "overlap"); return cn_948(cn_949, cn_950, cn_951, cn_952, cn_953); }());
    v_total_tiles = ([&]() -> O { O cn_954 = py::int_(0); for(py::handle cn_955 : py::reinterpret_borrow<py::iterable>(local(v_regions, "regions"))) { O v_row = py::reinterpret_borrow<O>(cn_955); cn_954 = binary(cn_954, py::int_(py::len(local(v_row, "row"))), Op::add); } return cn_954; }());
    v_tiles_processed = py::int_(0);
    for (py::handle cn_956 : py::reinterpret_borrow<py::iterable>(local(v_regions, "regions"))) {
        v_row = py::reinterpret_borrow<O>(cn_956);
        v_row_result = py::none();
        v_row_overlap = py::none();
        for (py::handle cn_957 : py::reinterpret_borrow<py::iterable>(local(v_row, "row"))) {
            O cn_958 = unpack(py::reinterpret_borrow<O>(cn_957), 2);
            v_tile = item(cn_958, py::int_(0));
            v_pad = item(cn_958, py::int_(1));
            v_padded_tile = ([&]() -> O { O cn_959 = attr(local(v_tile, "tile"), "add_padding"); O cn_960 = local(v_pad, "pad"); return cn_959(cn_960); }());
            if (!(truth(([&]() -> O { O cn_961 = attr(local(v_padded_tile, "padded_tile"), "size"); O cn_962 = local(v_exact_size, "exact_size"); return rich_compare(cn_961, cn_962, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            v_upscale_result = ([&]() -> O { O cn_965 = local(v_upscale, "upscale"); O cn_966 = ([&]() -> O { O cn_963 = attr(local(v_padded_tile, "padded_tile"), "read_from"); O cn_964 = local(v_img, "img"); return cn_963(cn_964); }()); O cn_967 = local(v_padded_tile, "padded_tile"); return cn_965(cn_966, cn_967); }());
            O cn_970 = unpack(([&]() -> O { O cn_968 = item(globals, py::str("get_h_w_c")); O cn_969 = local(v_upscale_result, "upscale_result"); return cn_968(cn_969); }()), 3);
            v_up_h = item(cn_970, py::int_(0));
            v_up_w = item(cn_970, py::int_(1));
            v_up_c = item(cn_970, py::int_(2));
            v_current_scale = ([&]() -> O { O cn_971 = local(v_up_h, "up_h"); O cn_972 = local(v_exact_h, "exact_h"); return binary(cn_971, cn_972, Op::floordiv); }());
            if (!(truth(([&]() -> O { O cn_973 = local(v_current_scale, "current_scale"); O cn_974 = py::int_(0); return rich_compare(cn_973, cn_974, Py_GT); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            if (!(truth(([&]() -> O { O cn_979 = ([&]() -> O { O cn_977 = local(v_exact_h, "exact_h"); O cn_978 = local(v_current_scale, "current_scale"); return binary(cn_977, cn_978, Op::mul); }()); O cn_980 = local(v_up_h, "up_h"); return rich_compare(cn_979, cn_980, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            if (!(truth(([&]() -> O { O cn_985 = ([&]() -> O { O cn_983 = local(v_exact_w, "exact_w"); O cn_984 = local(v_current_scale, "current_scale"); return binary(cn_983, cn_984, Op::mul); }()); O cn_986 = local(v_up_w, "up_w"); return rich_compare(cn_985, cn_986, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            if (truth(([&]() -> O { O cn_987 = local(v_row_result, "row_result"); O cn_988 = py::none(); return py::bool_(cn_987.ptr() == cn_988.ptr()); }()))) {
                v_scale = local(v_current_scale, "current_scale");
                v_out_channels = local(v_up_c, "up_c");
                v_row_result = ([&]() -> O { O cn_993 = item(globals, py::str("TileBlender")); O cn_994 = ([&]() -> O { O cn_989 = local(v_w, "w"); O cn_990 = local(v_scale, "scale"); return binary(cn_989, cn_990, Op::mul); }()); O cn_995 = ([&]() -> O { O cn_991 = local(v_exact_h, "exact_h"); O cn_992 = local(v_scale, "scale"); return binary(cn_991, cn_992, Op::mul); }()); O cn_996 = local(v_out_channels, "out_channels"); O cn_997 = attr(item(globals, py::str("BlendDirection")), "X"); O cn_998 = item(globals, py::str("half_sin_blend_fn")); return cn_993(py::arg("width") = cn_994, py::arg("height") = cn_995, py::arg("channels") = cn_996, py::arg("direction") = cn_997, py::arg("blend_fn") = cn_998); }());
                v_row_overlap = ([&]() -> O { O cn_1003 = item(globals, py::str("TileOverlap")); O cn_1004 = ([&]() -> O { O cn_999 = attr(local(v_pad, "pad"), "top"); O cn_1000 = local(v_scale, "scale"); return binary(cn_999, cn_1000, Op::mul); }()); O cn_1005 = ([&]() -> O { O cn_1001 = attr(local(v_pad, "pad"), "bottom"); O cn_1002 = local(v_scale, "scale"); return binary(cn_1001, cn_1002, Op::mul); }()); return cn_1003(cn_1004, cn_1005); }());
            }
            if (!(truth(([&]() -> O { O cn_1006 = local(v_current_scale, "current_scale"); O cn_1007 = local(v_scale, "scale"); return rich_compare(cn_1006, cn_1007, Py_EQ); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
            ([&]() -> O { O cn_1015 = attr(local(v_row_result, "row_result"), "add_tile"); O cn_1016 = local(v_upscale_result, "upscale_result"); O cn_1017 = ([&]() -> O { O cn_1012 = item(globals, py::str("TileOverlap")); O cn_1013 = ([&]() -> O { O cn_1008 = attr(local(v_pad, "pad"), "left"); O cn_1009 = local(v_scale, "scale"); return binary(cn_1008, cn_1009, Op::mul); }()); O cn_1014 = ([&]() -> O { O cn_1010 = attr(local(v_pad, "pad"), "right"); O cn_1011 = local(v_scale, "scale"); return binary(cn_1010, cn_1011, Op::mul); }()); return cn_1012(cn_1013, cn_1014); }()); return cn_1015(cn_1016, cn_1017); }());
            O cn_1018 = local(v_tiles_processed, "tiles_processed");
            O cn_1019 = py::int_(1);
            O cn_1020 = inplace(cn_1018, cn_1019, Op::add);
            v_tiles_processed = cn_1020;
            if (truth(([&]() -> O { O cn_1021 = local(v_progress, "progress"); O cn_1022 = py::none(); return py::bool_(cn_1021.ptr() != cn_1022.ptr()); }()))) {
                ([&]() -> O { O cn_1025 = attr(local(v_progress, "progress"), "set_progress"); O cn_1026 = ([&]() -> O { O cn_1023 = local(v_tiles_processed, "tiles_processed"); O cn_1024 = local(v_total_tiles, "total_tiles"); return binary(cn_1023, cn_1024, Op::div); }()); return cn_1025(cn_1026); }());
            }
        }
        if (!(truth(([&]() -> O { O cn_1027 = local(v_row_result, "row_result"); O cn_1028 = py::none(); return py::bool_(cn_1027.ptr() != cn_1028.ptr()); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
        if (!(truth(([&]() -> O { O cn_1029 = local(v_row_overlap, "row_overlap"); O cn_1030 = py::none(); return py::bool_(cn_1029.ptr() != cn_1030.ptr()); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
        if (truth(([&]() -> O { O cn_1031 = local(v_result, "result"); O cn_1032 = py::none(); return py::bool_(cn_1031.ptr() == cn_1032.ptr()); }()))) {
            v_result = ([&]() -> O { O cn_1037 = item(globals, py::str("TileBlender")); O cn_1038 = ([&]() -> O { O cn_1033 = local(v_w, "w"); O cn_1034 = local(v_scale, "scale"); return binary(cn_1033, cn_1034, Op::mul); }()); O cn_1039 = ([&]() -> O { O cn_1035 = local(v_h, "h"); O cn_1036 = local(v_scale, "scale"); return binary(cn_1035, cn_1036, Op::mul); }()); O cn_1040 = local(v_out_channels, "out_channels"); O cn_1041 = attr(item(globals, py::str("BlendDirection")), "Y"); O cn_1042 = item(globals, py::str("half_sin_blend_fn")); return cn_1037(py::arg("width") = cn_1038, py::arg("height") = cn_1039, py::arg("channels") = cn_1040, py::arg("direction") = cn_1041, py::arg("blend_fn") = cn_1042); }());
        }
        ([&]() -> O { O cn_1044 = attr(local(v_result, "result"), "add_tile"); O cn_1045 = ([&]() -> O { O cn_1043 = attr(local(v_row_result, "row_result"), "get_result"); return cn_1043(); }()); O cn_1046 = local(v_row_overlap, "row_overlap"); return cn_1044(cn_1045, cn_1046); }());
    }
    if (!(truth(([&]() -> O { O cn_1047 = local(v_result, "result"); O cn_1048 = py::none(); return py::bool_(cn_1047.ptr() != cn_1048.ptr()); }())))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
    return ([&]() -> O { O cn_1049 = attr(local(v_result, "result"), "get_result"); return cn_1049(); }());
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
    O cn_1053 = unpack(([&]() -> O { O cn_1050 = item(globals, py::str("_pad_image")); O cn_1051 = local(v_img, "img"); O cn_1052 = local(v_exact_size, "exact_size"); return cn_1050(cn_1051, cn_1052); }()), 2);
    v_img = item(cn_1053, py::int_(0));
    v_base_padding = item(cn_1053, py::int_(1));
    O cn_1056 = unpack(([&]() -> O { O cn_1054 = item(globals, py::str("get_h_w_c")); O cn_1055 = local(v_img, "img"); return cn_1054(cn_1055); }()), 3);
    v_h = item(cn_1056, py::int_(0));
    v_w = item(cn_1056, py::int_(1));
    v__ = item(cn_1056, py::int_(2));
    v_result = ([&]() -> O { O cn_1057 = item(globals, py::str("_exact_split_without_padding")); O cn_1058 = local(v_img, "img"); O cn_1059 = local(v_exact_size, "exact_size"); O cn_1060 = local(v_upscale, "upscale"); O cn_1061 = local(v_overlap, "overlap"); O cn_1062 = local(v_progress, "progress"); return cn_1057(cn_1058, cn_1059, cn_1060, cn_1061, cn_1062); }());
    v_scale = ([&]() -> O { O cn_1067 = ([&]() -> O { O cn_1065 = ([&]() -> O { O cn_1063 = item(globals, py::str("get_h_w_c")); O cn_1064 = local(v_result, "result"); return cn_1063(cn_1064); }()); O cn_1066 = py::int_(0); return item(cn_1065, cn_1066); }()); O cn_1068 = local(v_h, "h"); return binary(cn_1067, cn_1068, Op::floordiv); }());
    if (truth(attr(local(v_base_padding, "base_padding"), "empty"))) {
        return local(v_result, "result");
    }
    return ([&]() -> O { O cn_1078 = attr(([&]() -> O { O cn_1076 = attr(([&]() -> O { O cn_1074 = attr(([&]() -> O { O cn_1069 = item(globals, py::str("Region")); O cn_1070 = py::int_(0); O cn_1071 = py::int_(0); O cn_1072 = local(v_w, "w"); O cn_1073 = local(v_h, "h"); return cn_1069(cn_1070, cn_1071, cn_1072, cn_1073); }()), "remove_padding"); O cn_1075 = local(v_base_padding, "base_padding"); return cn_1074(cn_1075); }()), "scale"); O cn_1077 = local(v_scale, "scale"); return cn_1076(cn_1077); }()), "read_from"); O cn_1079 = local(v_result, "result"); return cn_1078(cn_1079); }());
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
    O cn_1080 = unpack(local(v_tile_size, "tile_size"), 2);
    v_w = item(cn_1080, py::int_(0));
    v_h = item(cn_1080, py::int_(1));
    if (!((truth(([&]() -> O { O cn_1081 = local(v_w, "w"); O cn_1082 = py::int_(16); return rich_compare(cn_1081, cn_1082, Py_GE); }())) && truth(([&]() -> O { O cn_1083 = local(v_h, "h"); O cn_1084 = py::int_(16); return rich_compare(cn_1083, cn_1084, Py_GE); }()))))) { PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }
    return make_tuple({([&]() -> O { O cn_1087 = builtin("max"); O cn_1088 = py::int_(16); O cn_1089 = ([&]() -> O { O cn_1085 = local(v_w, "w"); O cn_1086 = py::int_(2); return binary(cn_1085, cn_1086, Op::floordiv); }()); return cn_1087(cn_1088, cn_1089); }()), ([&]() -> O { O cn_1092 = builtin("max"); O cn_1093 = py::int_(16); O cn_1094 = ([&]() -> O { O cn_1090 = local(v_h, "h"); O cn_1091 = py::int_(2); return binary(cn_1090, cn_1091, Op::floordiv); }()); return cn_1092(cn_1093, cn_1094); }())});
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
    v_size = ([&]() -> O { O cn_1095 = builtin("max"); O cn_1096 = local(v_width, "width"); O cn_1097 = local(v_height, "height"); return cn_1095(cn_1096, cn_1097); }());
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
    O cn_1098 = local(v_tile_size, "tile_size");
    set_attr(self, "tile_size", cn_1098);
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
    v_max_tile_size = ([&]() -> O { O cn_1103 = builtin("max"); O cn_1104 = ([&]() -> O { O cn_1099 = local(v_width, "width"); O cn_1100 = py::int_(10); return binary(cn_1099, cn_1100, Op::add); }()); O cn_1105 = ([&]() -> O { O cn_1101 = local(v_height, "height"); O cn_1102 = py::int_(10); return binary(cn_1101, cn_1102, Op::add); }()); return cn_1103(cn_1104, cn_1105); }());
    v_size = ([&]() -> O { O cn_1106 = builtin("min"); O cn_1107 = attr(self, "tile_size"); O cn_1108 = local(v_max_tile_size, "max_tile_size"); return cn_1106(cn_1107, cn_1108); }());
    return make_tuple({local(v_size, "size"), local(v_size, "size")});
}
O tile_source_tiler_ExactTileSize_init(const O &globals, const O &self, O v_exact_size) {
    (void)globals;
    (void)self;
    (void)v_exact_size;
    O cn_1109 = local(v_exact_size, "exact_size");
    set_attr(self, "exact_size", cn_1109);
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
    raise(PyExc_ValueError, concat_text({py::str("Splits are not supported for exact size ("), py::str(([&]() -> O { O cn_1110 = attr(self, "exact_size"); O cn_1111 = py::int_(0); return item(cn_1110, cn_1111); }())), py::str("x"), py::str(([&]() -> O { O cn_1112 = attr(self, "exact_size"); O cn_1113 = py::int_(1); return item(cn_1112, cn_1113); }())), py::str("px) splitting. This typically means that your machine does not have enough VRAM to run the current model.")}));
}
O tile_source_tiler_BoundedTileSize_init(const O &globals, const O &self, O v_tile_size, O v_min_size, O v_max_size) {
    (void)globals;
    (void)self;
    (void)v_tile_size;
    (void)v_min_size;
    (void)v_max_size;
    O cn_1114 = unpack((truth(local(v_min_size, "min_size")) ? O(local(v_min_size, "min_size")) : O(make_tuple({py::int_(1), py::int_(1)}))), 2);
    O cn_1115 = item(cn_1114, py::int_(0));
    set_attr(self, "min_w", cn_1115);
    O cn_1116 = item(cn_1114, py::int_(1));
    set_attr(self, "min_h", cn_1116);
    O cn_1121 = unpack((truth(local(v_max_size, "max_size")) ? O(local(v_max_size, "max_size")) : O(make_tuple({([&]() -> O { O cn_1117 = py::int_(2); O cn_1118 = py::int_(31); return binary(cn_1117, cn_1118, Op::pow); }()), ([&]() -> O { O cn_1119 = py::int_(2); O cn_1120 = py::int_(31); return binary(cn_1119, cn_1120, Op::pow); }())}))), 2);
    O cn_1122 = item(cn_1121, py::int_(0));
    set_attr(self, "max_w", cn_1122);
    O cn_1123 = item(cn_1121, py::int_(1));
    set_attr(self, "max_h", cn_1123);
    O cn_1132 = ([&]() -> O { O cn_1128 = builtin("max"); O cn_1129 = attr(self, "min_w"); O cn_1130 = attr(self, "min_h"); O cn_1131 = ([&]() -> O { O cn_1124 = builtin("min"); O cn_1125 = local(v_tile_size, "tile_size"); O cn_1126 = attr(self, "max_w"); O cn_1127 = attr(self, "max_h"); return cn_1124(cn_1125, cn_1126, cn_1127); }()); return cn_1128(cn_1129, cn_1130, cn_1131); }());
    set_attr(self, "tile_size", cn_1132);
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
    O cn_1133 = unpack(local(v_tile_size, "tile_size"), 2);
    v_w = item(cn_1133, py::int_(0));
    v_h = item(cn_1133, py::int_(1));
    v_new_w = ([&]() -> O { O cn_1136 = builtin("max"); O cn_1137 = attr(self, "min_w"); O cn_1138 = ([&]() -> O { O cn_1134 = local(v_w, "w"); O cn_1135 = py::int_(2); return binary(cn_1134, cn_1135, Op::floordiv); }()); return cn_1136(cn_1137, cn_1138); }());
    v_new_h = ([&]() -> O { O cn_1141 = builtin("max"); O cn_1142 = attr(self, "min_h"); O cn_1143 = ([&]() -> O { O cn_1139 = local(v_h, "h"); O cn_1140 = py::int_(2); return binary(cn_1139, cn_1140, Op::floordiv); }()); return cn_1141(cn_1142, cn_1143); }());
    if ((truth(([&]() -> O { O cn_1144 = local(v_new_w, "new_w"); O cn_1145 = local(v_w, "w"); return rich_compare(cn_1144, cn_1145, Py_EQ); }())) && truth(([&]() -> O { O cn_1146 = local(v_new_h, "new_h"); O cn_1147 = local(v_h, "h"); return rich_compare(cn_1146, cn_1147, Py_EQ); }())))) {
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
