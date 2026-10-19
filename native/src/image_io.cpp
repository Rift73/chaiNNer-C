/* chaiNNer image I/O application algorithms (GPL-3.0).
 * Codec engines are fpng/OpenCV/Pillow/DirectXTex. C++ owns decoder ordering,
 * buffer/channel preparation, save policy/options, and owned job lifecycles.
 * CPython pathlib and OS APIs preserve the installed host semantics. */
#include "atomic_file.hpp"
#include "graph_python.hpp"
#include "graph_exception.hpp"
#include "image_codecs.h"
#include "image_decode.hpp"
#include "image_encode.hpp"
#include "image_preview.hpp"
#include "numpy_pool.hpp"
#include <pybind11/numpy.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace py = pybind11;
namespace {
using graphpy::O;
using graphpy::truth;
using graphpy::contains;
using cn::image_decode::exif_block;
using cn::image_decode::exif_orientation;
O own(PyObject* result) { if (!result) throw py::error_already_set(); return py::reinterpret_steal<O>(result); }
O name(const py::dict& g, const char* key) { return g[py::str(key)]; }
O index(const O& value, Py_ssize_t i) { return value[py::int_(i)]; }
bool equal(const O& a, const O& b) { return truth(own(PyObject_RichCompare(a.ptr(), b.ptr(), Py_EQ))); }
bool option(const py::dict& g, const O& value, const char* type, const char* member) { return equal(value, name(g,type).attr(member)); }
O add(const O& a, const O& b) { return own(PyNumber_Add(a.ptr(),b.ptr())); }
O path_join(const O& a, const O& b) { return own(PyNumber_TrueDivide(a.ptr(),b.ptr())); }
template<typename... A> O format(const char* text, A&&... args) { return py::str(text).attr("format")(std::forward<A>(args)...); }
[[noreturn]] void raise(PyObject* type, const O& message) { PyErr_SetObject(type,message.ptr()); throw py::error_already_set(); }
[[noreturn]] void rethrow_value(const O& error) { PyErr_SetObject(reinterpret_cast<PyObject*>(Py_TYPE(error.ptr())),error.ptr()); throw py::error_already_set(); }
void finally_do(const std::function<void()>& body, const std::function<void()>& cleanup) {
    try { body(); }
    catch (py::error_already_set& error) { GraphHandledException context(error); cleanup(); throw; }
    catch (...) { cleanup(); throw; }
    cleanup();
}
void with_object(O context, const std::function<void(O)>& body) {
    O type = py::type::of(context);
    O exit = type.attr("__exit__");
    O value = type.attr("__enter__")(context);
    try { body(value); }
    catch (py::error_already_set& error) {
        GraphHandledException handled(error);
        if (!truth(exit(context,error.type(),error.value(),error.trace()))) throw;
        return;
    }
    exit(context,py::none(),py::none(),py::none());
}
O get_ext(py::dict g, O path) { return index(name(g,"split_file_path")(path),2).attr("lower")(); }

// fpng (cn::image_encode::fpng_png) for a uint8 BGR(A) array; None when fpng declines. All codec dispatch
// decisions happen before compression; a failure in the selected encoder is reported, never retried through
// another codec. Other dtypes/channels/options retain OpenCV semantics.
O encode_fpng(const O& image) {
    if (!py::isinstance<py::array>(image)) return py::none();
    py::array input=py::reinterpret_borrow<py::array>(image);
    if (input.ndim()!=3 || input.itemsize()!=1 || input.dtype().kind()!='u') return py::none();
    std::vector<uint8_t> encoded;
    cn::image_encode::Fpng result;
    {
        // Keep input alive and snapshot every Python-owned field before release.
        const auto* source=static_cast<const uint8_t*>(input.data());
        const ptrdiff_t height=input.shape(0),width=input.shape(1),channels=input.shape(2);
        const ptrdiff_t row_stride=input.strides(0),pixel_stride=input.strides(1),channel_stride=input.strides(2);
        py::gil_scoped_release release;
        result=cn::image_encode::fpng_png(source,height,width,channels,row_stride,pixel_stride,channel_stride,encoded);
    }
    if(result==cn::image_encode::Fpng::declined)return py::none();
    if(result==cn::image_encode::Fpng::failed)throw std::runtime_error("fpng failed to encode the PNG image");
    return py::bytes(reinterpret_cast<const char*>(encoded.data()),encoded.size());
}

// Copy a channel permutation with the codec's native-endian dtype. Reading
// through byte pointers also supports unaligned, negative and readonly strides.
O swap_channels(py::dict g, O image, const char* code) {
    O cv = name(g,"cv2");
    if (!py::isinstance<py::array>(image)) return cv.attr("cvtColor")(image,cv.attr(code));
    py::array a = py::reinterpret_borrow<py::array>(image);
    const std::string kind = py::str(a.dtype().attr("kind"));
    const auto bytes = a.itemsize();
    if (a.ndim()!=3 || (a.shape(2)!=3 && a.shape(2)!=4) || a.shape(0)==0 || a.shape(1)==0 ||
        !((kind=="u" && (bytes==1 || bytes==2)) || (kind=="f" && bytes==4)))
        return cv.attr("cvtColor")(image,cv.attr(code)); // preserve incompatible codec contracts
    py::dtype dtype = kind=="f" ? py::dtype::of<float>() : (bytes==1 ? py::dtype::of<uint8_t>() : py::dtype::of<uint16_t>());
    const py::ssize_t height=a.shape(0),width=a.shape(1),channels=a.shape(2);
    const py::ssize_t row_stride=a.strides(0),pixel_stride=a.strides(1),channel_stride=a.strides(2);
    const char* data=static_cast<const char*>(a.data());
    py::array output(dtype, {height,width,channels});
    auto* dst = static_cast<char*>(output.mutable_data());
    {
        // Owners outlive this scope; only snapshotted buffer metadata is used.
        py::gil_scoped_release release;
        for (py::ssize_t y=0;y<height;++y) for (py::ssize_t x=0;x<width;++x) {
            for (py::ssize_t c=0;c<channels;++c) {
                const py::ssize_t source = c<3 ? 2-c : c;
                const char* src=data+y*row_stride+x*pixel_stride+source*channel_stride;
                std::memcpy(dst,src,static_cast<size_t>(bytes)); dst+=bytes;
            }
        }
    }
    return output;
}
O remove_alpha(py::dict g, O image) {
    if (!equal(index(name(g,"get_h_w_c")(image),2),py::int_(4))) return image;
    O dtype=image.attr("dtype"), np=name(g,"np");
    int mode=0;
    if(equal(dtype,np.attr("uint8")))mode=1;
    else if(equal(dtype,np.attr("uint16")))mode=2;
    else if(equal(dtype,np.attr("float32")))mode=3;
    else if(equal(dtype,np.attr("float64")))mode=4;
    if(!mode)return image;
    O all=py::slice(py::none(),py::none(),py::none());
    O alpha=image[py::make_tuple(all,all,3)];
    // The uint16 comparison intentionally preserves the installed 65536
    // sentinel. A nonempty uint16 alpha plane can therefore never be removed.
    py::array plane=py::array::ensure(alpha);
    if(!plane)throw py::type_error("Expected an array alpha plane");
    bool opaque=true;
    const char* data=static_cast<const char*>(plane.data());
    const py::ssize_t count=plane.size(),dimensions=plane.ndim();
    const std::vector<py::ssize_t> shape(plane.shape(),plane.shape()+dimensions);
    const std::vector<py::ssize_t> strides(plane.strides(),plane.strides()+dimensions);
    const auto is_opaque=[mode](const char* sample) {
        bool valid=false;
        if(mode==1) valid=static_cast<unsigned char>(*sample)==255;
        if(mode==3){uint32_t value;std::memcpy(&value,sample,4);valid=value==UINT32_C(0x3f800000);}
        if(mode==4){uint64_t value;std::memcpy(&value,sample,8);valid=value==UINT64_C(0x3ff0000000000000);}
        return valid;
    };
    {
        // Keep image/alpha/plane alive, and reacquire before slicing the result.
        py::gil_scoped_release release;
        if(mode==2 && count!=0) opaque=false;
        else if(dimensions==2) {
            for(py::ssize_t y=0;y<shape[0]&&opaque;++y) {
                const char* row=data+y*strides[0];
                for(py::ssize_t x=0;x<shape[1];++x)
                    if(!is_opaque(row+x*strides[1])){opaque=false;break;}
            }
        } else for(py::ssize_t i=0;i<count;++i) {
            py::ssize_t rest=i,offset=0;
            for(py::ssize_t d=dimensions;d-->0;){offset+=(rest%shape[static_cast<size_t>(d)])*strides[static_cast<size_t>(d)];rest/=shape[static_cast<size_t>(d)];}
            if(!is_opaque(data+offset)){opaque=false;break;}
        }
    }
    return opaque ? O(image[py::make_tuple(all,all,py::slice(py::none(),py::int_(3),py::none()))]) : image;
}
// OpenCV's imdecode decodes these through a temp file, and concurrent decodes
// could be handed the same temp name (SP3b decoder audit U2).
bool temp_file_suffix(const O& ext) {
    if(!PyUnicode_Check(ext.ptr()))return false;
    for(const char* suffix:{".sr",".ras",".hdr",".pic",".exr"})
        if(PyUnicode_CompareWithASCIIString(ext.ptr(),suffix)==0)return true;
    return false;
}
// imdecode picks its decoder by the leading bytes, whatever the file's name. In
// OpenCV 4.8.0 the decoders that cannot read from memory (m_buf_supported left
// false, so imdecode_ writes a temp file) are HDR, Sun raster, PFM and OpenEXR;
// these are their checkSignature rules (grfmt_hdr.cpp, grfmt_sunras.cpp,
// grfmt_pfm.cpp, grfmt_exr.cpp with grfmt_base.cpp's prefix compare).
bool temp_file_signature(const O& data) {
    if(!py::isinstance<py::array>(data))return false;
    py::array buffer=py::reinterpret_borrow<py::array>(data);
    if(buffer.ndim()!=1 || buffer.itemsize()!=1)return false;
    std::string head;
    const char* bytes=static_cast<const char*>(buffer.data());
    for(py::ssize_t i=0;i<std::min<py::ssize_t>(buffer.shape(0),10);++i)head.push_back(bytes[i*buffer.strides(0)]);
    const std::string_view view(head);
    // PFM's isspace(byte 2) depends on the CRT locale only above ASCII; lock those too.
    const auto space=[](unsigned char c){return c==' ' || (c>='\t' && c<='\r') || c>=0x80;};
    return view.starts_with("#?RGBE") || view.starts_with("#?RADIANCE") ||
        view.starts_with("\x59\xA6\x6A\x95") || view.starts_with("\x76\x2F\x31\x01") ||
        (view.size()>=3 && view[0]=='P' && (view[1]=='f' || view[1]=='F') && space(static_cast<unsigned char>(view[2])));
}
// Turns decoded pixels upright as EXIF Orientation 2-8 asks, as ImageOps.exif_transpose
// does (5-8 swap height and width), keeping the channel order; 1 returns the image itself.
O orient(py::dict g,O image,int orientation) {
    if(orientation==1)return image;
    const O all=py::slice(py::none(),py::none(),py::none()),reverse=py::slice(py::none(),py::none(),py::int_(-1));
    const bool rows=orientation==3||orientation==4||orientation==6||orientation==7;
    const bool columns=orientation==2||orientation==3||orientation==7||orientation==8;
    O view=image[py::make_tuple(rows?reverse:all,columns?reverse:all)];
    if(orientation>=5)view=view.attr("swapaxes")(0,1);
    return name(g,"np").attr("ascontiguousarray")(view);
}
// orient() for an Orientation tag value read elsewhere: out of range is 1, as in
// exif_orientation.
O orient_tag(py::dict g,O image,long long orientation) {
    return orient(g,image,orientation>=2&&orientation<=8?int(orientation):1);
}
// Today's decoders that decode natively: Pillow's for a JPEG, OpenCV's imdecode, the straight-alpha TIFF one.
enum class NativeDecoder {pillow_jpeg,opencv,pillow_tiff_alpha};
// The native decode of the file at `path` as today's decoder would give it (cn::image_decode); None when it
// declines, and today's path then runs. The file is read and decoded without the GIL; it is taken only to create
// the array the pixels go into.
O native_decode(py::dict g,O path,NativeDecoder decoder) {
    const bool pillow=decoder==NativeDecoder::pillow_jpeg;
    if(!cn::image_decode::enabled())return py::none();
    uint64_t max_pixels=0;
    if(pillow) {
        O limit=name(g,"Image").attr("MAX_IMAGE_PIXELS");
        if(!limit.is_none()) {
            if(!PyLong_CheckExact(limit.ptr()))return py::none();
            const long long value=PyLong_AsLongLong(limit.ptr());
            if(value<=0) {PyErr_Clear();return py::none();}
            max_pixels=uint64_t(value);
        }
    }
    const std::filesystem::path file(py::str(path).cast<std::wstring>());
    O result=py::none();
    bool decoded=false;
    {
        py::gil_scoped_release release;
        const cn::image_decode::Allocator allocate=[&](const cn::image_decode::Shape& shape)->uint8_t* {
            py::gil_scoped_acquire acquire;
            using cn::image_decode::Sample;
            py::dtype dtype=py::dtype::of<uint8_t>();
            switch(shape.sample) {
                case Sample::u8:break;
                case Sample::u16:dtype=py::dtype::of<uint16_t>();break;
                case Sample::u32:dtype=py::dtype::of<uint32_t>();break;
                case Sample::u64:dtype=py::dtype::of<uint64_t>();break;
                case Sample::i8:dtype=py::dtype::of<int8_t>();break;
                case Sample::i16:dtype=py::dtype::of<int16_t>();break;
                case Sample::i32:dtype=py::dtype::of<int32_t>();break;
                case Sample::i64:dtype=py::dtype::of<int64_t>();break;
                case Sample::f32:dtype=py::dtype::of<float>();break;
                case Sample::f64:dtype=py::dtype::of<double>();break;
            }
            py::array array=shape.channels==1?py::array(dtype,{py::ssize_t(shape.height),py::ssize_t(shape.width)}):
                py::array(dtype,{py::ssize_t(shape.height),py::ssize_t(shape.width),py::ssize_t(shape.channels)});
            result=array;
            return static_cast<uint8_t*>(array.mutable_data());
        };
        try {
            std::vector<uint8_t> data;
            decoded=cn::image_decode::read_file(file,data) &&
                (pillow?cn::image_decode::decode_pillow_jpeg(data,max_pixels,allocate):
                 decoder==NativeDecoder::opencv?cn::image_decode::decode_opencv(data,allocate):
                 cn::image_decode::decode_pillow_tiff_alpha(data,allocate));
        } catch(const std::bad_alloc&) {decoded=false;} // today's path reports it
    }
    return decoded?result:O(py::none());
}
bool native_jpeg_extension(const O& ext) {
    for(const char* suffix:{".jpg",".jpeg",".jfif"})
        if(PyUnicode_Check(ext.ptr()) && PyUnicode_CompareWithASCIIString(ext.ptr(),suffix)==0)return true;
    return false;
}
O read_cv(py::dict g,O path) {
    O ext=name(g,"get_ext")(path);
    if(!contains(name(g,"get_opencv_formats")(),ext))return py::none();
    if(O native=native_decode(g,path,NativeDecoder::opencv);!native.is_none())return native;
    O image=py::none(),cv=name(g,"cv2");
    // IMREAD_UNCHANGED leaves EXIF Orientation to the caller; read it from the file's bytes.
    int orientation=1;
    try {
        O data=name(g,"np").attr("fromfile")(path,py::arg("dtype")=name(g,"np").attr("uint8"));
        py::array bytes=py::reinterpret_borrow<py::array>(data);
        const std::string_view exif=exif_block(static_cast<const uint8_t*>(bytes.data()),static_cast<size_t>(bytes.size()));
        orientation=exif_orientation(reinterpret_cast<const uint8_t*>(exif.data()),exif.size());
        // One temp-file decode at a time, process-wide, keyed on the name and on
        // the content. Wait without the GIL: the holder needs it to call into cv2.
        static std::mutex temp_file_decodes;
        std::unique_lock<std::mutex> serialized(temp_file_decodes,std::defer_lock);
        if(temp_file_suffix(ext) || temp_file_signature(data)) { py::gil_scoped_release release; serialized.lock(); }
        image=cv.attr("imdecode")(data,cv.attr("IMREAD_UNCHANGED"));
    }
    catch(py::error_already_set& error) {
        if(!error.matches(PyExc_Exception))throw;
        GraphHandledException handled(error);
        name(g,"logger").attr("warning")(format("Error loading image, trying with imdecode: {}",error.value()));
    }
    if(image.is_none()) {
        try { image=cv.attr("imread")(py::str(path),cv.attr("IMREAD_UNCHANGED")); }
        catch(py::error_already_set& error) {
            if(!error.matches(PyExc_Exception))throw;
            GraphHandledException handled(error);
            O exception=graphpy::builtin("RuntimeError")(format("Error reading image image from path \"{}\". Image may be corrupt.",path));
            PyException_SetCause(exception.ptr(),Py_NewRef(error.value().ptr()));
            rethrow_value(exception);
        }
    }
    if(image.is_none())raise(PyExc_RuntimeError,format("Error reading image image from path \"{}\". Image may be corrupt.",path));
    return orient(g,image,orientation);
}
O read_pil(py::dict g,O path) {
    O ext=name(g,"get_ext")(path);
    if(!contains(name(g,"get_pil_formats")(),ext))return py::none();
    if(native_jpeg_extension(ext))
        if(O native=native_decode(g,path,NativeDecoder::pillow_jpeg);!native.is_none())return native;
    O im=name(g,"Image").attr("open")(path);
    if(equal(im.attr("mode"),py::str("P")))im=im.attr("convert")(im.attr("palette").attr("mode"));
    // Four CMYK samples are not RGBA: K would become the alpha. Pillow's RGB of them.
    else if(equal(im.attr("mode"),py::str("CMYK")))im=im.attr("convert")("RGB");
    O image=name(g,"np").attr("array")(im);
    // Read after the decode, which also collects a PNG's eXIf after IDAT. Pillow keeps
    // a TIFF's tags elsewhere, so TIFFs stay as decoded.
    O exif=im.attr("info").attr("get")("exif");
    const int orientation=PyBytes_Check(exif.ptr())?exif_orientation(reinterpret_cast<const uint8_t*>(PyBytes_AS_STRING(exif.ptr())),static_cast<size_t>(PyBytes_GET_SIZE(exif.ptr()))):1;
    O channels=index(name(g,"get_h_w_c")(image),2);
    if(equal(channels,py::int_(3)))image=swap_channels(g,image,"COLOR_RGB2BGR");
    else if(equal(channels,py::int_(4)))image=swap_channels(g,image,"COLOR_RGBA2BGRA");
    return orient(g,image,orientation);
}
// The straight-alpha TIFF decoder's native half (load_image.py's _read_tiff_straight_alpha calls it first):
// BGRA, turned, or None for the Python decoder to run.
O read_tiff_alpha(py::dict g,O path) {
    return native_decode(g,path,NativeDecoder::pillow_tiff_alpha);
}
O read_dds(py::dict g,O path) {
    if(!equal(name(g,"get_ext")(path),py::str(".dds")))return py::none();
    if(!equal(name(g,"platform").attr("system")(),py::str("Windows")))return py::none();
    return name(g,"remove_unnecessary_alpha")(name(g,"read_dds_texconv")(path));
}
O decode_ext(O extensions,O decoder,O extension_fn,O path) {
    return contains(extensions,extension_fn(path)) ? decoder(path) : O(py::none());
}
O for_ext(O extensions,O decoder,O extension_fn) {
    py::set selected;
    int is_str=PyObject_IsInstance(extensions.ptr(),reinterpret_cast<PyObject*>(&PyUnicode_Type));
    if(is_str<0)throw py::error_already_set();
    if(is_str)selected.add(extensions);else selected.attr("update")(extensions);
    // partial is GC-tracked; unlike a C++ closure it does not hide a cycle
    // through a decoder function and its module globals from Python's GC.
    return py::module_::import("functools").attr("partial")(py::cpp_function(&decode_ext),selected,decoder,extension_fn);
}
O load_image(py::dict g,O path) {
    name(g,"logger").attr("debug")(format("Reading image from path: {}",path));
    O parts=name(g,"split_file_path")(path), image=py::none(),failure=py::none();
    for(py::handle pair_handle:name(g,"_decoders")) {
        O pair=py::reinterpret_borrow<O>(pair_handle),decoder=index(pair,1);
        try {image=decoder(name(g,"Path")(path));}
        catch(py::error_already_set& error) {
            if(!error.matches(PyExc_Exception))throw;
            GraphHandledException handled(error);failure=error.value();
            name(g,"logger").attr("warning")(format("Decoder {} failed",index(pair,0)));
        }
        if(!image.is_none())break;
    }
    if(image.is_none()) {
        if(!failure.is_none())rethrow_value(failure);
        raise(PyExc_RuntimeError,format("The image \"{}\" you are trying to read cannot be read by chaiNNer.",path));
    }
    return py::make_tuple(image,index(parts,0),index(parts,1));
}
O full_path(O base,O relative,O filename,O format_type) {
    O file=format("{}.{}",filename,format_type.attr("extension"));
    if(truth(relative)&&!equal(relative,py::str(".")))base=path_join(base,relative);
    return path_join(base,file).attr("resolve")();
}
// Save Image's conversion for every format but DDS, in today's order: Pillow
// formats (GIF, TGA, AVIF) and lossless WebP with alpha get a uint8 image in
// RGB(A) order and their save options; OpenCV formats get their parameters and
// the precision conversion.
struct Encoding {O image; bool pillow=false,exact_webp=false; py::dict options; py::list params;};
// Save Image's inputs in order: 0-11 as upstream's, 12-22 the DDS inputs (save_dds_inputs),
// then AVIF's chroma subsampling and skip existing files.
constexpr size_t save_inputs=25,avif_subsampling=23,skip_existing=24;
bool pillow_format(const py::dict& g,const O& format_type) {
    return option(g,format_type,"ImageFormat","GIF")||option(g,format_type,"ImageFormat","TGA")||option(g,format_type,"ImageFormat","AVIF");
}
Encoding encoding(py::dict g,O image,py::tuple args) {
    O format_type=args[4];
    Encoding result;
    // OpenCV's lossless WebP lets libwebp change the colour under alpha 0; Pillow's
    // exact=True keeps every sample (an RGB image has no such pixels).
    const bool exact_webp=option(g,format_type,"ImageFormat","WEBP") && truth(args[6]) &&
        equal(index(name(g,"get_h_w_c")(image),2),py::int_(4));
    if(pillow_format(g,format_type) || exact_webp) {
        image=name(g,"to_uint8")(image,py::arg("normalized")=true);
        if(option(g,format_type,"ImageFormat","AVIF")){result.options["quality"]=args[7];result.options["subsampling"]=O(args[avif_subsampling]).attr("value");}
        if(exact_webp){result.options["lossless"]=py::bool_(true);result.options["exact"]=py::bool_(true);}
        O channels=index(name(g,"get_h_w_c")(image),2);
        if(equal(channels,py::int_(3)))image=swap_channels(g,image,"COLOR_BGR2RGB");
        else if(equal(channels,py::int_(4)))image=swap_channels(g,image,"COLOR_BGRA2RGBA");
        else if(!equal(channels,py::int_(1)))raise(PyExc_RuntimeError,format("Unsupported number of channels. Saving .{} images is only supported for grayscale, RGB, and RGBA images.",format_type.attr("extension")));
        result.image=image;result.pillow=true;result.exact_webp=exact_webp;
        return result;
    }
    O cv=name(g,"cv2");py::list& params=result.params;
    if(option(g,format_type,"ImageFormat","JPG")) {
        params.append(cv.attr("IMWRITE_JPEG_QUALITY"));params.append(args[7]);
        params.append(cv.attr("IMWRITE_JPEG_SAMPLING_FACTOR"));params.append(O(args[8]).attr("value"));
        params.append(cv.attr("IMWRITE_JPEG_PROGRESSIVE"));params.append(graphpy::builtin("int")(args[9]));
    } else if(option(g,format_type,"ImageFormat","WEBP")) {params.append(cv.attr("IMWRITE_WEBP_QUALITY"));params.append(truth(args[6])?O(py::int_(101)):O(args[7]));}
    else if(option(g,format_type,"ImageFormat","TIFF")&&!option(g,args[10],"TiffColorDepth","F32")) {params.append(cv.attr("IMWRITE_TIFF_COMPRESSION"));params.append(O(args[11]).attr("cv2_code"));}
    int precision=8;
    if(option(g,format_type,"ImageFormat","PNG")){if(option(g,args[5],"PngColorDepth","U16"))precision=16;}
    else if(option(g,format_type,"ImageFormat","TIFF")) {
        if(option(g,args[10],"TiffColorDepth","U16"))precision=16;
        else if(option(g,args[10],"TiffColorDepth","F32"))precision=32;
    }
    if(precision!=32)image=name(g,precision==8?"to_uint8":"to_uint16")(image,py::arg("normalized")=true);
    result.image=image;
    return result;
}
// chaiNNer's ExtraSamples fix of an encoded TIFF (cn::image_encode::tiff_extra_samples) for a bytes or array
// buffer: the fixed file as bytes, or the buffer as it is.
O add_extra_samples(O buffer) {
    py::array array=py::reinterpret_borrow<py::array>(buffer);
    std::vector<uint8_t> fixed;
    if(!cn::image_encode::tiff_extra_samples({static_cast<const uint8_t*>(array.data()),static_cast<size_t>(array.nbytes())},fixed))
        return buffer;
    return py::bytes(reinterpret_cast<const char*>(fixed.data()),fixed.size());
}
// The image of an encoder, as cv2 takes a uint8, uint16 or float32 array of 1, 3 or 4 channels: C-contiguous
// pixels only (anything else declines). The array must outlive the view.
bool codec_image(const O& value,cn_codec_image& image) {
    if(!py::isinstance<py::array>(value))return false;
    py::array a=py::reinterpret_borrow<py::array>(value);
    const char kind=a.dtype().kind();
    const py::ssize_t bytes=a.itemsize();
    if(!(kind=='u' && (bytes==1 || bytes==2)) && !(kind=='f' && bytes==4))return false;
    if(a.ndim()!=2 && a.ndim()!=3)return false;
    const py::ssize_t channels=a.ndim()==2?1:a.shape(2);
    if((channels!=1 && channels!=3 && channels!=4) || a.shape(0)<=0 || a.shape(1)<=0 ||
        a.shape(0)>INT32_MAX || a.shape(1)>INT32_MAX)return false;
    if(a.strides(1)!=channels*bytes || (a.ndim()==3 && a.strides(2)!=bytes) || a.strides(0)!=a.shape(1)*channels*bytes)return false;
    image=cn_codec_image{static_cast<const uint8_t*>(a.data()),uint32_t(a.shape(0)),uint32_t(a.shape(1)),uint32_t(channels),
        uint32_t(bytes),size_t(a.strides(0))};
    return true;
}
// Runs an encoder without the GIL into a NumPy-pool block; the encoded bytes as a uint8 array that owns the block
// (as cv2.imencode returns its buffer), or None when it declines.
O run_encoder(const std::function<cn_codec_status(cn_codec_output&)>& encoder) {
    cn_codec_output out{};
    out.allocate=cn_numpy_pool_allocate;
    out.release=cn_numpy_pool_free;
    cn_codec_status status;
    {
        py::gil_scoped_release release;
        status=encoder(out);
    }
    if(status!=CN_CODEC_OK) {
        cn_codec_output_free(&out);
        return py::none();
    }
    py::capsule owner(out.data,[](void* data){cn_numpy_pool_free(data);});
    return py::array_t<uint8_t>({py::ssize_t(out.size)},{py::ssize_t(1)},out.data,owner);
}
bool extension_is(const O& extension,std::initializer_list<const char*> names) {
    if(!PyUnicode_Check(extension.ptr()))return false;
    for(const char* name:names)if(PyUnicode_CompareWithASCIIString(extension.ptr(),name)==0)return true;
    return false;
}
// cv2.imencode's bytes from the vendored encoders (image_codecs.h) for the formats and parameters Save Image uses:
// JPEG (quality, sampling factor, progressive), PNG and BMP (no parameters), WebP (quality), TIFF (compression);
// None for anything else, which keeps cv2.imencode.
O native_encode(py::dict g,O extension,O image,O params) {
    cn_codec_image view;
    if(!cn::image_decode::enabled() || !codec_image(image,view))return py::none();
    if(!PyList_CheckExact(params.ptr()) && !PyTuple_CheckExact(params.ptr()))return py::none();
    const py::sequence list=py::reinterpret_borrow<py::sequence>(params);
    const py::ssize_t count=py::len(list);
    if(count%2)return py::none();
    O lower=extension.attr("lower")(),cv=name(g,"cv2");
    std::vector<std::pair<long long,long long>> pairs;
    for(py::ssize_t i=0;i<count;i+=2) {
        O key=list[i],value=list[i+1];
        if(!PyLong_CheckExact(key.ptr()) || !PyLong_CheckExact(value.ptr()))return py::none();
        pairs.emplace_back(py::cast<long long>(key),py::cast<long long>(value));
    }
    const auto constant=[&](const char* key){return py::cast<long long>(cv.attr(key));};
    if(extension_is(lower,{".jpg",".jpeg",".jpe"})) {
        long long quality=95,sampling=0x221111,progressive=0;
        for(const auto& [key,value]:pairs) {
            if(key==constant("IMWRITE_JPEG_QUALITY"))quality=value;
            else if(key==constant("IMWRITE_JPEG_SAMPLING_FACTOR"))sampling=value;
            else if(key==constant("IMWRITE_JPEG_PROGRESSIVE"))progressive=value;
            else return py::none();
        }
        if(quality<0 || quality>100 || sampling<0 || sampling>INT32_MAX || (progressive!=0 && progressive!=1))return py::none();
        return run_encoder([&](cn_codec_output& out){return cn_jpeg_encode(&view,int(quality),uint32_t(sampling),int(progressive),&out);});
    }
    if(extension_is(lower,{".webp"})) {
        long long quality=0;
        for(const auto& [key,value]:pairs) {
            if(key!=constant("IMWRITE_WEBP_QUALITY"))return py::none();
            quality=value;
        }
        // No quality, or one above 100, is lossless; below 1 is 1 (OpenCV says so on stderr).
        if(pairs.empty() || quality>100)quality=101;
        if(quality<1)quality=1;
        return run_encoder([&](cn_codec_output& out){return cn_webp_encode(&view,int(quality),&out);});
    }
    if(extension_is(lower,{".tif",".tiff"})) {
        // IMWRITE_TIFF_COMPRESSION as Save Image passes it (none, LZW, Adobe Deflate), or no parameters.
        long long compression=0;
        for(const auto& [key,value]:pairs) {
            if(key!=constant("IMWRITE_TIFF_COMPRESSION") || (value!=1 && value!=5 && value!=8))return py::none();
            compression=value;
        }
        return run_encoder([&](cn_codec_output& out){return cn_tiff_encode(&view,int(compression),&out);});
    }
    if(!pairs.empty())return py::none();
    if(extension_is(lower,{".png"}))return run_encoder([&](cn_codec_output& out){return cn_png_encode(&view,&out);});
    if(extension_is(lower,{".bmp",".dib"}))return run_encoder([&](cn_codec_output& out){return cn_bmp_encode(&view,&out);});
    return py::none();
}
// Pillow's lossless WebP with exact=True of an RGBA image (Encoding's exact_webp) from the vendored libwebp
// (cn_webp_encode_exact); None for every other encoding or when it declines.
O native_exact_webp(const Encoding& encoded) {
    cn_codec_image view;
    if(!encoded.exact_webp || !cn::image_decode::enabled() || !codec_image(encoded.image,view))return py::none();
    return run_encoder([&](cn_codec_output& out){return cn_webp_encode_exact(&view,&out);});
}
// The codec of cv_save for a file extension: fpng for a default PNG, else the vendored encoder of OpenCV's
// (native_encode), else OpenCV's imencode; a TIFF then gets add_extra_samples. Returns bytes, or the encoder's buffer
// array.
O encode(py::dict g,O extension,O image,O params) {
    O buffer=py::none();
    // Explicit codec parameters are a distinct OpenCV contract (compression,
    // bilevel, strategy). Only default PNG encoding is fpng-compatible.
    if(equal(extension.attr("lower")(),py::str(".png")) &&
        (PyList_CheckExact(params.ptr()) || PyTuple_CheckExact(params.ptr())) && py::len(params)==0)
        buffer=encode_fpng(image);
    if(buffer.is_none())buffer=native_encode(g,extension,image,params);
    if(buffer.is_none())buffer=index(name(g,"cv2").attr("imencode")(format(".{}",extension),image,params),1);
    O lower=extension.attr("lower")();
    if(equal(lower,py::str(".tiff")) || equal(lower,py::str(".tif")))buffer=add_extra_samples(buffer);
    return buffer;
}
// Writes a file's bytes (bytes or a contiguous array) to path through a temporary file in its folder
// (cn::files::write_atomic, without the GIL): no partial file is ever left under path, and a file that already holds
// these bytes is left untouched. A failure raises the OSError open() or write() raises, naming path.
void write_file(const O& path,const O& data) {
    const std::filesystem::path final(py::str(path).cast<std::wstring>());
    Py_buffer view;
    if(PyObject_GetBuffer(data.ptr(),&view,PyBUF_SIMPLE)!=0)throw py::error_already_set();
    struct Release {Py_buffer& view; ~Release(){PyBuffer_Release(&view);}} release_view{view};
    py::gil_scoped_release release;
    cn::files::write_atomic(final,{static_cast<const uint8_t*>(view.buf),size_t(view.len)});
}
// The bytes a Pillow format (GIF, TGA, AVIF, lossless WebP with alpha) writes for an encoding: the native exact
// WebP, else Image.save into a buffer with the plugin named, picked as Image.save(path) picks it (Pillow 9.2):
// preinit(), init() when the extension is unknown, then EXTENSION[extension]; the registry may not be initialized yet.
O pillow_bytes(py::dict g,const Encoding& encoded,const O& extension) {
    if(O data=native_exact_webp(encoded);!data.is_none())return data.attr("tobytes")();
    O image_module=name(g,"Image"),buffer=py::module_::import("io").attr("BytesIO")();
    image_module.attr("preinit")();
    if(!contains(image_module.attr("EXTENSION"),extension))image_module.attr("init")();
    O pillow=image_module.attr("EXTENSION")[extension];
    with_object(image_module.attr("fromarray")(encoded.image),[&](O im){im.attr("save")(buffer,py::arg("format")=pillow,**encoded.options);});
    return buffer.attr("getvalue")();
}
// The DDS inputs of Save Image and Save DDS Texture, in order: format, BC7 compression,
// error metric, dithering, Generate Mip Maps, mip levels, mip filter, separate alpha,
// colour conversion, premultiplied alpha, GPU compression; as save_as_dds's options.
// Legacy and PREFER_DX9 formats take a DX9 header, but a DX9 header records
// premultiplied alpha only as DXT2 and DXT4 (other formats take DX10's alpha mode then)
// and has no arrays.
void save_dds_inputs(py::dict g,O path,O image,O kind,O dds) {
    O format=dds[py::int_(0)];
    const bool array=equal(kind,py::str("array"))&&PyList_Check(image.ptr())&&py::len(image)>1;
    const bool dx9=!array&&(contains(name(g,"LEGACY_TO_DXGI"),format)||(contains(name(g,"PREFER_DX9"),format)&&!truth(dds[py::int_(9)])));
    name(g,"save_as_dds")(path,image,name(g,"to_dxgi")(format),
        // Generate Mip Maps: 0 (yes) takes the level count (0: the full chain), 1 (no) one level.
        py::arg("mipmap_levels")=equal(dds[py::int_(4)],py::int_(1))?O(py::int_(1)):O(dds[py::int_(5)]),py::arg("dithering")=dds[py::int_(3)],
        py::arg("uniform_weighting")=option(g,dds[py::int_(2)],"DDSErrorMetric","UNIFORM"),
        py::arg("minimal_compression")=option(g,dds[py::int_(1)],"BC7Compression","BEST_SPEED"),
        py::arg("maximum_compression")=option(g,dds[py::int_(1)],"BC7Compression","BEST_QUALITY"),
        py::arg("dx9")=dx9,py::arg("separate_alpha")=dds[py::int_(7)],
        py::arg("colour_conversion")=O(dds[py::int_(8)]).attr("value"),py::arg("premultiply_alpha")=dds[py::int_(9)],
        py::arg("gpu")=dds[py::int_(10)],py::arg("mip_filter")=O(dds[py::int_(6)]).attr("value"),py::arg("kind")=kind);
}
void save_image(py::dict g,py::tuple args,O prepared) {
    if(args.size()!=save_inputs)raise(PyExc_TypeError,format("Image save expects {} input values",save_inputs));
    O lazy=args[0],base=args[1],relative=args[2],filename=args[3],format_type=args[4];
    O path=name(g,"get_full_path")(base,relative,filename,format_type);
    if(truth(path.attr("exists")())) {
        if(truth(args[skip_existing])) {name(g,"logger").attr("debug")(format("Skipping existing file: {}",path));return;}
    } else path.attr("parent").attr("mkdir")(py::arg("parents")=true,py::arg("exist_ok")=true);
    name(g,"logger").attr("debug")(format("Writing image to path: {}",path));
    O image=lazy.attr("value");
    // A prepared value (SP3-P9) holds the bytes the branches below would write, as
    // encoded for the format's own extension, written as each branch writes its file.
    // A path that resolved to another spelling (an existing file or a link) encodes
    // as today and leaves the value unused: a node-level exception to the seam's one
    // decision point, since only the commit knows the resolved path.
    if(!prepared.is_none() && equal(index(py::module_::import("os").attr("path").attr("splitext")(path),1),add(py::str("."),format_type.attr("extension")))) {
        write_file(path,prepared.attr("result")());
        return;
    }
    if(option(g,format_type,"ImageFormat","DDS")) {
        save_dds_inputs(g,path,image,py::str("array"),args[py::slice(12,23,1)]);
        return;
    }
    Encoding encoded=encoding(g,image,args);
    if(encoded.pillow) {
        write_file(path,pillow_bytes(g,encoded,add(py::str("."),format_type.attr("extension"))));
        return;
    }
    name(g,"cv_save_image")(path,encoded.image,encoded.params);
}
// Save Image's prepare phase (SP3-P9): the bytes save_image writes for these
// inputs (input 0 the image itself) to a path with the format's own extension,
// encoded in memory; None for DDS, which texconv writes itself, and for a file
// the commit skips as existing, which it never encodes.
O save_prepare(py::dict g,py::tuple args) {
    if(args.size()!=save_inputs)raise(PyExc_TypeError,format("Image save expects {} input values",save_inputs));
    O format_type=args[4];
    if(option(g,format_type,"ImageFormat","DDS"))return py::none();
    if(truth(args[skip_existing]) && truth(name(g,"get_full_path")(args[1],args[2],args[3],format_type).attr("exists")()))return py::none();
    O extension=add(py::str("."),format_type.attr("extension"));
    Encoding encoded=encoding(g,args[0],args);
    if(encoded.pillow)return pillow_bytes(g,encoded,extension);
    O buffer=encode(g,extension,encoded.image,encoded.params);
    return PyBytes_CheckExact(buffer.ptr())?buffer:O(buffer.attr("tobytes")());
}
// LargeImageOutput.get_broadcast_data's previews of an enforced image (cn::image_preview, without the GIL): a list
// of {"width", "height", "url"}, or None when the native path declines and today's Python loop runs.
O previews(O value) {
    if(!cn::image_decode::enabled() || !py::isinstance<py::array>(value))return py::none();
    py::array a=py::reinterpret_borrow<py::array>(value);
    if(a.dtype().kind()!='f' || a.itemsize()!=4 || (a.ndim()!=2 && a.ndim()!=3) || !(a.flags()&py::array::c_style))return py::none();
    const py::ssize_t channels=a.ndim()==2?1:a.shape(2);
    if(channels<1 || channels>4 || a.shape(0)<=0 || a.shape(1)<=0 || a.shape(0)>INT32_MAX || a.shape(1)>INT32_MAX)return py::none();
    std::vector<cn::image_preview::Preview> out;
    bool ok;
    {
        py::gil_scoped_release release;
        ok=cn::image_preview::previews(static_cast<const float*>(a.data()),uint32_t(a.shape(0)),uint32_t(a.shape(1)),
            uint32_t(channels),out);
    }
    if(!ok)return py::none();
    py::list list;
    for(const auto& preview:out) {
        py::dict entry;
        entry["width"]=preview.width;entry["height"]=preview.height;entry["url"]=py::str(preview.url);
        list.append(entry);
    }
    return list;
}
void view_image(py::dict g,O image) {
    O temporary=name(g,"mkdtemp")(py::arg("prefix")="chaiNNer-");
    name(g,"logger").attr("debug")(format("Writing image to temp path: {}",temporary));
    O filename=format("{}.png",name(g,"time").attr("time")());
    O path=name(g,"os").attr("path").attr("join")(temporary,filename);
    O pixels=name(g,"to_uint8")(image,py::arg("normalized")=true);
    O buffer=encode_fpng(pixels);
    bool saved=false;
    if(buffer.is_none())saved=truth(name(g,"cv2").attr("imwrite")(path,pixels));
    else {
        try {
            with_object(graphpy::builtin("open")(path,"wb"),[&](O file){file.attr("write")(buffer);});
            saved=true;
        } catch(const py::error_already_set& error) {
            // imwrite returns false on a filesystem write failure. Preserve the
            // viewer's no-launch policy; encoder/allocation errors still escape.
            if(!error.matches(PyExc_OSError))throw;
        }
    }
    if(saved) {
        if(equal(name(g,"platform").attr("system")(),py::str("Darwin")))name(g,"subprocess").attr("call")(py::make_tuple("open",path));
        else if(equal(name(g,"platform").attr("system")(),py::str("Windows")))name(g,"os").attr("startfile")(path);
        else name(g,"subprocess").attr("call")(py::make_tuple("xdg-open",path));
    }
}
void cv_save(py::dict g,O path,O image,O params) {
    write_file(path,encode(g,index(name(g,"split_file_path")(path),2),image,params));
}
O decode_bytes(O bytes) {
    try{return bytes.attr("decode")(py::arg("encoding")="iso8859-1");}
    catch(py::error_already_set& first){if(!first.matches(PyExc_Exception))throw;GraphHandledException first_context(first);
        try{return bytes.attr("decode")(py::arg("encoding")="utf-8");}
        catch(py::error_already_set& second){if(!second.matches(PyExc_Exception))throw;GraphHandledException second_context(second);return py::str(bytes);}
    }
}
void run_texconv(py::dict g,O args,O error_message) {
    if(!equal(name(g,"platform").attr("system")(),py::str("Windows")))raise(PyExc_ValueError,py::str("Texconv is only supported on Windows. Reading and writing DDS files is only partially supported on other systems."));
    py::list command;command.append(name(g,"__TEXCONV_EXE"));command.append("-nologo");command.attr("extend")(args);
    O result=name(g,"subprocess").attr("run")(command,py::arg("check")=false,py::arg("capture_output")=true);
    if(!equal(result.attr("returncode"),py::int_(0))) {
        O output=add(name(g,"__decode")(result.attr("stdout")),name(g,"__decode")(result.attr("stderr"))).attr("replace")("\r","");
        py::list lines;lines.append("Failed to run texconv.");lines.append(format("texconv: {}",name(g,"__TEXCONV_EXE")));lines.append(format("args: {}",args));lines.append(format("exit code: {}",result.attr("returncode")));lines.append(format("output: {}",output));
        name(g,"logger").attr("error")(py::str("\n").attr("join")(lines));
        raise(PyExc_ValueError,format("{}: Code {}: {}",error_message,result.attr("returncode"),output));
    }
}
// DDS files (Microsoft's DDS_HEADER, DDS_PIXELFORMAT and DDS_HEADER_DXT10). texconv
// does every format conversion; chaiNNer writes and reads only uncompressed
// intermediates: R8G8B8A8_UNORM for formats of at most 8 bits per channel,
// R16G16B16A16_UNORM for 16-bit UNORM and SNORM ones, R32G32B32A32_FLOAT for every
// other. texconv maps UNORM 0..1 to SNORM -1..1 and back, but takes float as is.
uint32_t le32(const uint8_t* p) {return uint32_t(p[0])|uint32_t(p[1])<<8|uint32_t(p[2])<<16|uint32_t(p[3])<<24;}
constexpr uint32_t fourcc(const char (&c)[5]) {return uint32_t(uint8_t(c[0]))|uint32_t(uint8_t(c[1]))<<8|uint32_t(uint8_t(c[2]))<<16|uint32_t(uint8_t(c[3]))<<24;}
enum class DdsDepth {unorm8,unorm16,float32};
DdsDepth dds_depth(uint32_t format) {
    switch(format) {
    // R16G16B16A16, R16G16, R16 UNORM and SNORM
    case 11: case 13: case 35: case 37: case 56: case 58: return DdsDepth::unorm16;
    // 0: a legacy format DXGI lacks (R8G8B8, A4L4, palettized, ...), all of 8 bits or less.
    case 0: case 27: case 28: case 29: case 31: case 48: case 49: case 51: case 60: case 61: case 63: case 65:
    case 68: case 69: case 85: case 86: case 87: case 88: case 90: case 91: case 92: case 93: case 97: case 98:
    case 99: case 100: case 103: case 106: case 107: case 110: case 111: case 112: case 113: case 114: case 115:
    case 191: return DdsDepth::unorm8;
    default: return format>=70&&format<=84?DdsDepth::unorm8:DdsDepth::float32; // BC1-BC5
    }
}
// A legacy pixel format's DXGI value as DirectXTex maps it, 0 when DXGI has none.
uint32_t legacy_format(uint32_t flags,uint32_t code,uint32_t bits,const uint32_t (&m)[4]) {
    const auto is=[&](uint32_t r,uint32_t g,uint32_t b,uint32_t a){return m[0]==r&&m[1]==g&&m[2]==b&&m[3]==a;};
    if(flags&0x4) { // DDPF_FOURCC, a D3DFORMAT number or four characters
        static const std::pair<uint32_t,uint32_t> codes[]={{36,11},{110,13},{111,54},{112,34},{113,10},{114,41},
            {115,16},{116,2},{fourcc("DXT1"),71},{fourcc("DXT2"),74},{fourcc("DXT3"),74},{fourcc("DXT4"),77},
            {fourcc("DXT5"),77},{fourcc("ATI1"),80},{fourcc("BC4U"),80},{fourcc("BC4S"),81},{fourcc("ATI2"),83},
            {fourcc("BC5U"),83},{fourcc("BC5S"),84},{fourcc("RGBG"),68},{fourcc("GRGB"),69},{fourcc("YUY2"),107}};
        for(const auto& [key,format]:codes)if(key==code)return format;
    } else if(flags&0x40) { // DDPF_RGB
        if(bits==32) {
            if(is(0xff,0xff00,0xff0000,0xff000000)||is(0xff,0xff00,0xff0000,0))return 28;
            if(is(0xff0000,0xff00,0xff,0xff000000))return 87;
            if(is(0xff0000,0xff00,0xff,0))return 88;
            if(is(0x3ff,0xffc00,0x3ff00000,0xc0000000)||is(0x3ff00000,0xffc00,0x3ff,0xc0000000))return 24;
            if(is(0xffff,0xffff0000,0,0))return 35;
            if(is(0xffffffff,0,0,0))return 41;
        } else if(bits==16) {
            if(is(0x7c00,0x3e0,0x1f,0x8000)||is(0x7c00,0x3e0,0x1f,0))return 86;
            if(is(0xf800,0x7e0,0x1f,0))return 85;
            if(is(0xf00,0xf0,0xf,0xf000)||is(0xf00,0xf0,0xf,0))return 115;
        }
    } else if(flags&0x20000) { // DDPF_LUMINANCE
        if(bits==8&&is(0xff,0,0,0))return 61;
        if(bits==16&&is(0xffff,0,0,0))return 56;
        if(bits==16&&is(0xff,0,0,0xff00))return 49;
    } else if(flags&0x80000) { // DDPF_BUMPDUDV
        if(bits==16&&is(0xff,0xff00,0,0))return 51;
        if(bits==32&&is(0xff,0xff00,0xff0000,0xff000000))return 31;
        if(bits==32&&is(0xffff,0xffff0000,0,0))return 37;
    } else if((flags&0x2)&&bits==8)return 65; // DDPF_ALPHA
    return 0;
}
// A legacy mask format's name in D3DFORMAT's style, its channels from the highest bit
// down with their widths ("R8G8B8 (legacy)", "A4L4 (legacy)", "P8 (legacy)"); empty
// without masks.
std::string mask_name(uint32_t flags,const uint32_t (&m)[4]) {
    if(flags&0x20)return flags&0x1?"A8P8 (legacy)":"P8 (legacy)"; // DDPF_PALETTEINDEXED8, ALPHAPIXELS
    // DDPF_BUMPDUDV, DDPF_BUMPLUMINANCE, DDPF_LUMINANCE, else RGB or DDPF_ALPHA
    const char* letters=flags&0x80000?"UVWA":flags&0x40000?"UVLA":flags&0x20000?"L--A":"RGBA";
    std::vector<std::pair<uint32_t,char>> channels;
    for(int i=0;i<4;++i)if(m[i])channels.emplace_back(m[i],letters[i]);
    std::sort(channels.begin(),channels.end(),std::greater<>());
    std::string name;
    for(const auto& [mask,letter]:channels)name+=letter+std::to_string(std::popcount(mask));
    return name.empty()?name:name+" (legacy)";
}
struct DdsHeader {
    uint32_t format=0,width=0,height=0,depth=1,mips=1,array=1;
    bool cubemap=false,volume=false;
    std::string fourcc; // a legacy header's four characters, empty otherwise
    std::string masks;  // a legacy mask format's name when DXGI lacks it, empty otherwise
    size_t data=0;      // offset of the first image
    uint32_t images() const {return cubemap?array*6:array;} // a cubemap's faces each count
};
DdsHeader parse_dds(const O& file) {
    const auto* p=reinterpret_cast<const uint8_t*>(PyBytes_AS_STRING(file.ptr()));
    const size_t n=static_cast<size_t>(PyBytes_GET_SIZE(file.ptr()));
    if(n<128||std::memcmp(p,"DDS ",4)!=0||le32(p+4)!=124||le32(p+76)!=32)raise(PyExc_ValueError,py::str("Not a DDS file"));
    DdsHeader h;
    h.height=le32(p+12);h.width=le32(p+16);h.data=128;
    if(le32(p+8)&0x20000)h.mips=std::max(1u,le32(p+28)); // DDSD_MIPMAPCOUNT
    const uint32_t flags=le32(p+80),code=le32(p+84);
    h.volume=(le32(p+112)&0x200000)!=0; // DDSCAPS2_VOLUME
    h.cubemap=(le32(p+112)&0x200)!=0;   // DDSCAPS2_CUBEMAP (texconv refuses partial ones)
    if((flags&0x4)&&code==fourcc("DX10")) {
        if(n<148)raise(PyExc_ValueError,py::str("Truncated DDS header"));
        h.format=le32(p+128);h.array=le32(p+140);h.data=148;
        h.volume=le32(p+132)==4;          // D3D10_RESOURCE_DIMENSION_TEXTURE3D
        h.cubemap=(le32(p+136)&0x4)!=0;   // D3D11_RESOURCE_MISC_TEXTURECUBE
    } else {
        const uint32_t masks[4]={le32(p+92),le32(p+96),le32(p+100),le32(p+104)};
        h.format=legacy_format(flags,code,le32(p+88),masks);
        if(flags&0x4)h.fourcc.assign(reinterpret_cast<const char*>(p+84),4);
        else if(!h.format)h.masks=mask_name(flags,masks);
    }
    if(h.volume)h.depth=std::max(1u,le32(p+24));
    if(!h.width||!h.height||!h.array)raise(PyExc_ValueError,py::str("Invalid DDS dimensions"));
    return h;
}
// A mip level's width, height or depth.
uint32_t level(uint32_t size,uint32_t mip) {return std::max(1u,mip<32?size>>mip:0u);}
// One image of an uncompressed intermediate (image `item`, mip level `mip`, depth
// slice `slice`, laid out as DirectXTex writes them: an array's images each with
// their mips, a volume's mips each with their slices) as BGRA: uint8, or float32 for
// R16G16B16A16_UNORM and R16_UNORM (n / 65535, as Load Image's output normalizes
// uint16; R16 grey, alpha 1) and R32G32B32A32_FLOAT.
O dds_pixels(const O& file,const DdsHeader& h,uint32_t item,uint32_t mip,uint32_t slice) {
    const size_t element=h.format==28?1:h.format==11||h.format==56?2:h.format==2?4:0;
    if(!element)raise(PyExc_ValueError,format("Unexpected DDS intermediate format {}",h.format));
    const size_t samples=h.format==56?1:4,n=static_cast<size_t>(PyBytes_GET_SIZE(file.ptr()));
    if(item>=h.images()||mip>=h.mips||slice>=level(h.depth,mip))raise(PyExc_ValueError,py::str("DDS intermediate lacks the image"));
    if(uint64_t(h.width)*h.height>n||h.depth>n)raise(PyExc_ValueError,py::str("Truncated DDS image"));
    const auto bytes=[&](uint32_t m){return uint64_t(level(h.width,m))*level(h.height,m)*level(h.depth,m)*samples*element;};
    uint64_t offset=h.data,chain=0;
    for(uint32_t m=0;m<h.mips;++m){chain+=bytes(m);if(m<mip)offset+=bytes(m);}
    const uint64_t width=level(h.width,mip),height=level(h.height,mip),pixels=width*height;
    offset+=item*chain+uint64_t(slice)*pixels*samples*element;
    if(offset+pixels*samples*element>n)raise(PyExc_ValueError,py::str("Truncated DDS image"));
    const auto* data=reinterpret_cast<const uint8_t*>(PyBytes_AS_STRING(file.ptr()))+offset;
    py::array image(py::dtype(py::str(element==1?"uint8":"float32")),{py::ssize_t(height),py::ssize_t(width),py::ssize_t(4)});
    auto* out=static_cast<uint8_t*>(image.mutable_data());
    {
        py::gil_scoped_release release;
        for(uint64_t i=0;i<pixels;++i) for(size_t c=0;c<4;++c) {
            const uint8_t* sample=data+(i*samples+(samples==1?0:c<3?2-c:3))*element;
            if(element==2) {
                const float value=samples==1&&c==3?1.0f:float(uint32_t(sample[0])|uint32_t(sample[1])<<8)/65535.0f;
                std::memcpy(out,&value,4);out+=4;
            } else {std::memcpy(out,sample,element);out+=element;}
        }
    }
    return image;
}
// A single-channel DXGI format (R32, R16, R8, A8 and BC4 variants).
bool single_channel(uint32_t format) {
    return (format>=41&&format<=43)||(format>=54&&format<=59&&format!=55)||(format>=61&&format<=65)||format==80||format==81;
}
// An uncompressed DDS (DX10 header) of grey, BGR or BGRA uint8, uint16 or float32
// arrays of one size and type: of `kind` "array" (each image an element; one image is
// a plain 2D texture), "cubemap" (6 faces per cube: +X, -X, +Y, -Y, +Z, -Z) or
// "volume" (each image a depth slice). Grey is one channel (R8, R16, R32_FLOAT) when
// `one_channel` and every image is grey, else RGB: texconv converts RGB to one channel
// by luminance and one channel to RGBA at 16 bits through a path that is off by a step
// at times, so grey is RGB only for targets of several channels, and R8 at 8 bits, as
// WIC loaded a grey PNG. Without alpha the alpha is opaque, and so is the alpha mode:
// texconv keeps a known mode when compressed alpha decodes below opaque, as it did for
// a PNG without alpha.
O raw_dds(const std::vector<py::array>& images,const std::string& kind,bool one_channel) {
    const bool cubemap=kind=="cubemap",volume=kind=="volume";
    if(!cubemap&&!volume&&kind!="array")raise(PyExc_ValueError,format("Unknown DDS texture kind {!r}",kind));
    if(images.empty()||(cubemap&&images.size()%6))raise(PyExc_ValueError,py::str(cubemap?"A cubemap needs 6 faces per cube":"A DDS texture needs an image"));
    const py::array& first=images[0];
    if(first.ndim()!=2&&first.ndim()!=3)throw py::value_error("Expected an image array");
    const py::ssize_t height=first.shape(0),width=first.shape(1);
    const char type=first.dtype().kind();
    const size_t element=static_cast<size_t>(first.itemsize());
    if(!((type=='u'&&(element==1||element==2))||(type=='f'&&element==4)))throw py::value_error("Expected a uint8, uint16 or float32 image");
    bool grey=one_channel,alpha=false;
    for(const py::array& a:images) {
        const py::ssize_t channels=a.ndim()==2?1:a.ndim()==3?a.shape(2):0;
        if(channels!=1&&channels!=3&&channels!=4)raise(PyExc_ValueError,format("Saving DDS is only supported for grayscale, RGB, and RGBA images, not {} channels.",channels));
        if(a.shape(0)!=height||a.shape(1)!=width||a.dtype().kind()!=type||static_cast<size_t>(a.itemsize())!=element)
            raise(PyExc_ValueError,py::str("The images of a DDS texture must have one size"));
        grey=grey&&channels==1;alpha=alpha||channels==4;
    }
    const size_t count=images.size(),pixel=(grey?1:4)*element,plane=size_t(height)*size_t(width)*pixel;
    O result=own(PyBytes_FromStringAndSize(nullptr,static_cast<Py_ssize_t>(148+count*plane)));
    auto* out=reinterpret_cast<uint8_t*>(PyBytes_AS_STRING(result.ptr()));
    std::memset(out,0,148);
    const auto put=[&](size_t at,uint32_t value){for(int i=0;i<4;++i)out[at+i]=uint8_t(value>>(8*i));};
    std::memcpy(out,"DDS ",4);
    // CAPS|HEIGHT|WIDTH|PITCH|PIXELFORMAT|MIPMAPCOUNT, and DEPTH for a volume
    put(4,124);put(8,0x2100F|(volume?0x800000:0));put(12,uint32_t(height));put(16,uint32_t(width));
    put(20,uint32_t(size_t(width)*pixel));put(24,volume?uint32_t(count):1);put(28,1);
    put(76,32);put(80,0x4);put(84,fourcc("DX10"));
    put(108,0x1000|(cubemap||volume?0x8:0)); // DDSCAPS_TEXTURE, COMPLEX
    put(112,cubemap?0xFE00:volume?0x200000:0); // DDSCAPS2_CUBEMAP and its 6 faces, DDSCAPS2_VOLUME
    put(128,element==1?(grey?61:28):element==2?(grey?56:11):(grey?41:2));
    put(132,volume?4:3);put(136,cubemap?0x4:0); // TEXTURE3D or TEXTURE2D, TEXTURECUBE
    put(140,volume?1:uint32_t(cubemap?count/6:count));
    if(!alpha)put(144,3); // DDS_ALPHA_MODE_OPAQUE, as WIC reports a PNG without alpha
    for(size_t i=0;i<count;++i) {
        const py::array& a=images[i];
        const py::ssize_t channels=a.ndim()==2?1:a.shape(2);
        const char* data=static_cast<const char*>(a.data());
        const py::ssize_t row_stride=a.strides(0),pixel_stride=a.strides(1),channel_stride=a.ndim()==3?a.strides(2):0;
        py::gil_scoped_release release;
        static constexpr uint8_t one8=0xff,one16[2]={0xff,0xff},one32[4]={0,0,0x80,0x3f};
        const uint8_t* one=element==1?&one8:element==2?one16:one32;
        uint8_t* at=out+148+i*plane;
        for(py::ssize_t y=0;y<height;++y) for(py::ssize_t x=0;x<width;++x) {
            const char* source=data+y*row_stride+x*pixel_stride;
            if(grey){std::memcpy(at,source,element);at+=element;continue;}
            for(py::ssize_t c=0;c<4;++c) {
                if(c==3&&channels!=4)std::memcpy(at,one,element);
                else std::memcpy(at,source+(channels==1?0:c<3?2-c:3)*channel_stride,element);
                at+=element;
            }
        }
    }
    return result;
}
// A file's bytes: all of them, or its first `size`.
O read_bytes(O path,py::ssize_t size=-1) {
    O data;
    with_object(graphpy::builtin("open")(path,"rb"),[&](O file){data=file.attr("read")(size);});
    return data;
}
// A DDS file's header with its DX10 extension (148 bytes), not its images.
DdsHeader read_dds_header(O path) {return parse_dds(read_bytes(path,148));}
// The intermediate a source format is read through. R16_UNORM and R16_SNORM read as
// R16_UNORM: texconv's R16 -> R16G16B16A16 conversion is off by a step at times.
const char* intermediate_name(uint32_t source) {
    if(source==56||source==58)return "R16_UNORM";
    const DdsDepth depth=dds_depth(source);
    return depth==DdsDepth::unorm8?"R8G8B8A8_UNORM":depth==DdsDepth::unorm16?"R16G16B16A16_UNORM":"R32G32B32A32_FLOAT";
}
// texconv converts the file to the intermediate of its format's depth, the top mip
// only or (`every_level`) every mip; values keep their encoding (an sRGB format's with
// -srgb, in and out alike), premultiplied alpha becomes straight (for sRGB in linear
// light, as texconv premultiplies an sRGB format). `use` reads the converted file.
void dds_convert(py::dict g,O path,const DdsHeader& source,bool every_level,const std::function<void(const O&,const DdsHeader&)>& use) {
    O temporary=name(g,"mkdtemp")(py::arg("prefix")="chaiNNer-");
    finally_do([&] {
        // -m 1 keeps the top level only; the file's own count keeps every level (a
        // smaller count would have texconv generate the levels anew).
        py::list args;for(const char* v:{"-ft","dds","-f",intermediate_name(source.format),"-dx10","-m"})args.append(v);
        args.append(py::str(std::to_string(every_level?source.mips:1)));
        for(const uint32_t srgb:{29u,72u,75u,78u,91u,93u,99u})if(source.format==srgb)args.append("-srgb");
        args.append("-alpha");args.append("-o");
        args.append(temporary);args.append(py::str(path));
        name(g,"__run_texconv")(args,"Unable to convert DDS");
        // texconv names its output after the input's stem, dots and all.
        O output=path_join(name(g,"Path")(temporary),format("{}.dds",index(name(g,"split_file_path")(path),1)));
        O file=read_bytes(output);
        use(file,parse_dds(file));
    },[&]{name(g,"shutil").attr("rmtree")(temporary);});
}
void check_mip(const DdsHeader& source,uint32_t mip) {
    if(mip>=source.mips)raise(PyExc_ValueError,format("Mip level {} does not exist: the file has {}",mip,source.mips));
}
O dds_read(py::dict g,O path,uint32_t mip,uint32_t layer,uint32_t slice) {
    const DdsHeader source=read_dds_header(path);
    check_mip(source,mip);
    if(layer>=source.images())raise(PyExc_ValueError,format("Layer {} does not exist: the file has {}{}",layer,source.images(),source.cubemap?" (cubemap faces)":""));
    if(slice>=level(source.depth,mip))raise(PyExc_ValueError,format("Slice {} does not exist: mip level {} has {}",slice,mip,level(source.depth,mip)));
    O image;
    dds_convert(g,path,source,mip!=0,[&](const O& file,const DdsHeader& h){image=dds_pixels(file,h,layer,mip,slice);});
    return image;
}
// Every image of a mip level: each layer (cubemap faces) or, for a volume, each slice.
py::list dds_read_all(py::dict g,O path,uint32_t mip) {
    const DdsHeader source=read_dds_header(path);
    check_mip(source,mip);
    py::list images;
    dds_convert(g,path,source,mip!=0,[&](const O& file,const DdsHeader& h) {
        const uint32_t count=source.volume?level(source.depth,mip):source.images();
        for(uint32_t i=0;i<count;++i)images.append(dds_pixels(file,h,source.volume?0:i,mip,source.volume?i:0));
    });
    return images;
}
// A DDS file's header: its DXGI format value (0 for a legacy format DXGI lacks), the
// legacy FourCC (empty for DX10 and masks), the name of a mask format DXGI lacks (empty
// otherwise), size, depth, mip levels, array size (cubemaps: cubes) and whether it is a
// cubemap.
py::tuple dds_info(O path) {
    const DdsHeader h=read_dds_header(path);
    return py::make_tuple(h.format,h.fourcc,h.masks,h.width,h.height,h.depth,h.mips,h.array,h.cubemap);
}
// IEC 61966-2-1's sRGB curve, either way, on B, G and R (or grey) of a C-contiguous
// float32 image in place; odd below zero, so negative values mirror positive ones.
void srgb_curve(py::array image,bool to_linear) {
    const py::ssize_t channels=image.ndim()==2?1:image.shape(2),colour=channels==4?3:channels;
    const py::ssize_t pixels=image.shape(0)*image.shape(1);
    float* data=static_cast<float*>(image.mutable_data());
    py::gil_scoped_release release;
    for(py::ssize_t i=0;i<pixels;++i) for(py::ssize_t c=0;c<colour;++c) {
        float& value=data[i*channels+c];
        const double a=std::fabs(double(value));
        const double curve=to_linear?(a<=0.04045?a/12.92:std::pow((a+0.055)/1.055,2.4)):(a<=0.0031308?a*12.92:1.055*std::pow(a,1/2.4)-0.055);
        value=float(std::copysign(curve,double(value)));
    }
}
// The image as the intermediate of the target format's depth: uint8 and uint16 by
// to_uint8/to_uint16 (8-bit targets keep the PNG intermediate's quantization), float32
// otherwise, unclipped. A colour conversion runs before, on a float32 copy.
O dds_intermediate(py::dict g,O image,uint32_t target,const O& conversion) {
    O np=name(g,"np");
    const bool keep=equal(conversion,py::str("keep"));
    if(!keep||dds_depth(target)==DdsDepth::float32) {
        const bool floating=equal(np.attr("dtype")(image.attr("dtype")).attr("kind"),py::str("f"));
        image=keep?(floating?np.attr("asarray")(image,np.attr("float32")):name(g,"normalize")(image)):
            np.attr("array")(floating?image:name(g,"normalize")(image),py::arg("dtype")=np.attr("float32"),py::arg("order")="C");
        if(!keep) {
            if(!equal(conversion,py::str("srgb_to_linear"))&&!equal(conversion,py::str("linear_to_srgb")))raise(PyExc_ValueError,format("Unknown colour conversion {!r}",conversion));
            srgb_curve(py::reinterpret_borrow<py::array>(image),equal(conversion,py::str("srgb_to_linear")));
        }
    }
    switch(dds_depth(target)) {
    case DdsDepth::unorm8: return name(g,"to_uint8")(image,py::arg("normalized")=true);
    case DdsDepth::unorm16: return name(g,"to_uint16")(image,py::arg("normalized")=true);
    default: return image;
    }
}
void save_dds(py::dict g,py::tuple values) {
    if(values.size()!=15)throw py::type_error("DDS save expects 15 input values");
    O path=values[0],image=values[1],format_type=values[2];
    O parts=name(g,"split_file_path")(path),filename=index(parts,1);
    if(!equal(index(parts,2),py::str(".dds")))raise(PyExc_AssertionError,py::str("The file to save must end with '.dds'"));
    const uint32_t target=py::int_(name(g,"DXGI_CODES")[format_type]).cast<uint32_t>();
    std::vector<py::array> pixels;
    for(py::handle item:PyList_Check(image.ptr())?py::reinterpret_borrow<py::list>(image):py::list(py::make_tuple(image)))
        pixels.push_back(py::array::ensure(dds_intermediate(g,py::reinterpret_borrow<O>(item),target,values[10])));
    O temporary=name(g,"mkdtemp")(py::arg("prefix")="chaiNNer-");
    finally_do([&] {
        O raw=name(g,"os").attr("path").attr("join")(temporary,format("{}.dds",filename));
        O data=raw_dds(pixels,py::str(values[14]).cast<std::string>(),dds_depth(target)==DdsDepth::unorm8||single_channel(target));
        with_object(graphpy::builtin("open")(raw,"wb"),[&](O file){file.attr("write")(data);});
        // texconv writes into the temporary folder, over the intermediate (-y; it reads the whole file first).
        py::list args;args.append("-y");args.append("-f");args.append(format_type);args.append(truth(values[8])?"-dx9":"-dx10");args.append("-m");args.append(py::str(values[3]));args.append("-o");args.append(temporary);
        // "default" leaves texconv its own choice (Fant; for sRGB, filters of its own).
        if(!equal(values[13],py::str("default"))){args.append("-if");args.append(values[13]);}
        std::string bc;for(const auto& pair:{std::pair<size_t,char>{4,'u'},{5,'d'},{6,'q'},{7,'x'}})if(truth(values[pair.first]))bc+=pair.second;
        if(!bc.empty()){args.append("-bc");args.append("-"+bc);}
        if(contains(name(g,"SRGB_FORMATS"),format_type))args.append("-srgbi");
        if(truth(values[9]))args.append("-sepalpha");
        // Under a DX9 header texconv writes premultiplied BC2 and BC3 as DXT2 and DXT4.
        if(truth(values[11]))args.append("-pmalpha");
        if(!truth(values[12]))args.append("-nogpu");
        args.append(raw);name(g,"__run_texconv")(args,"Unable to write DDS");
        // The finished file goes to the target folder's temporary file, then onto the final name; a final file with
        // the same bytes is left untouched.
        const std::filesystem::path written(py::str(raw).cast<std::wstring>()),final(py::str(path).cast<std::wstring>());
        py::gil_scoped_release release;
        cn::files::move_atomic(written,final);
    },[&]{name(g,"shutil").attr("rmtree")(temporary);});
}
} // namespace
void cn_bind_image_io(py::module_& module) {
    module.def("image_io_ext",&get_ext);module.def("image_io_alpha",&remove_alpha);
    module.def("image_io_read_cv",&read_cv);module.def("image_io_read_pil",&read_pil);module.def("image_io_read_dds",&read_dds);
    module.def("image_io_read_tiff_alpha",&read_tiff_alpha);
    module.def("image_io_for_ext",&for_ext);module.def("image_io_load",&load_image);
    module.def("image_io_full_path",&full_path);module.def("image_io_save",&save_image);module.def("image_io_view",&view_image);
    module.def("image_io_save_prepare",&save_prepare);module.def("image_io_orient",&orient_tag);
    module.def("image_io_cv_save",&cv_save);module.def("image_io_previews",&previews);
    module.def("image_io_decode_bytes",&decode_bytes);module.def("image_io_run_texconv",&run_texconv);
    module.def("image_io_dds_read",&dds_read);module.def("image_io_save_dds",&save_dds);module.def("image_io_dds_info",&dds_info);
    module.def("image_io_dds_read_all",&dds_read_all);module.def("image_io_save_dds_inputs",&save_dds_inputs);
}
