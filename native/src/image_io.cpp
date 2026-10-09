/* chaiNNer image I/O application algorithms (GPL-3.0).
 * Codec engines are fpng/OpenCV/Pillow/DirectXTex. C++ owns decoder ordering,
 * buffer/channel preparation, save policy/options, and owned job lifecycles.
 * CPython pathlib and OS APIs preserve the installed host semantics. */
#include "graph_python.hpp"
#include "graph_exception.hpp"
#include "fpng.h"
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace py = pybind11;
namespace {
using graphpy::O;
using graphpy::truth;
using graphpy::contains;
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

// fpng accepts packed RGB(A) bytes. All codec dispatch decisions happen before
// compression; a failure in the selected encoder is reported, never retried
// through another codec. Other dtypes/channels/options retain OpenCV semantics.
O encode_fpng(const O& image) {
    if (!py::isinstance<py::array>(image)) return py::none();
    py::array input=py::reinterpret_borrow<py::array>(image);
    if (input.ndim()!=3 || input.itemsize()!=1 || input.dtype().kind()!='u' ||
        (input.shape(2)!=3 && input.shape(2)!=4)) return py::none();
    const py::ssize_t height=input.shape(0),width=input.shape(1),channels=input.shape(2);
    // Vendor 1.0.6 writes only two dimension bytes into IHDR despite accepting
    // larger values. It also uses uint32 offsets and signed row sizes. Keep
    // valid dimensions and worst-case uncompressed framing below those limits.
    constexpr uint64_t maximum_dimension=65535;
    constexpr uint64_t maximum_filtered=INT32_MAX-UINT64_C(1048576);
    if (height<=0 || width<=0 || static_cast<uint64_t>(height)>maximum_dimension ||
        static_cast<uint64_t>(width)>maximum_dimension ||
        (static_cast<uint64_t>(width)*static_cast<uint64_t>(channels)+1)*static_cast<uint64_t>(height)>maximum_filtered)
        return py::none();
    const py::ssize_t row_stride=input.strides(0),pixel_stride=input.strides(1),channel_stride=input.strides(2);
    const auto* source=static_cast<const uint8_t*>(input.data());
    std::vector<uint8_t> pixels,encoded;
    bool success;
    {
        // Keep input alive and snapshot every Python-owned field before release.
        // Copy also handles readonly, reversed and broadcast array views.
        py::gil_scoped_release release;
        static std::once_flag initialized;
        std::call_once(initialized,fpng::fpng_init);
        pixels.resize(static_cast<size_t>(height)*static_cast<size_t>(width)*static_cast<size_t>(channels));
        auto* destination=pixels.data();
        for(py::ssize_t y=0;y<height;++y) for(py::ssize_t x=0;x<width;++x) {
            const uint8_t* pixel=source+y*row_stride+x*pixel_stride;
            *destination++=pixel[2*channel_stride];
            *destination++=pixel[channel_stride];
            *destination++=pixel[0];
            if(channels==4)*destination++=pixel[3*channel_stride];
        }
        success=fpng::fpng_encode_image_to_memory(pixels.data(),static_cast<uint32_t>(width),
            static_cast<uint32_t>(height),static_cast<uint32_t>(channels),encoded,0);
    }
    if(!success)throw std::runtime_error("fpng failed to encode the PNG image");
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
// A classic TIFF structure ("II" little-endian or "MM" big-endian, magic 42): valid
// when the header and its first IFD's entry count lie in the n bytes; the IFD's
// entries are 12 bytes each from ifd+2.
struct ClassicTiff {
    const uint8_t* p; size_t n; bool little=false,valid=false; size_t ifd=0;
    ClassicTiff(const uint8_t* bytes,size_t size):p(bytes),n(size) {
        if(n<8 || !((p[0]=='I'&&p[1]=='I')||(p[0]=='M'&&p[1]=='M')))return;
        little=p[0]=='I';ifd=u32(4);
        valid=u16(2)==42 && ifd<=n-2;
    }
    uint32_t u16(size_t at) const {return little?uint32_t(p[at]|p[at+1]<<8):uint32_t(p[at]<<8|p[at+1]);}
    uint32_t u32(size_t at) const {return little?u16(at)|u16(at+2)<<16:u16(at)<<16|u16(at+2);}
};
// EXIF Orientation (tag 274) of an EXIF block, the TIFF header and IFD0 after an
// optional "Exif\0\0": 2-8, or 1 when it is absent, malformed or out of range.
int exif_orientation(const uint8_t* p,size_t n) {
    if(n>=6 && std::memcmp(p,"Exif\0\0",6)==0){p+=6;n-=6;}
    const ClassicTiff tiff(p,n);
    if(!tiff.valid)return 1;
    for(size_t entry=tiff.ifd+2,end=entry+size_t(tiff.u16(tiff.ifd))*12;entry<end && entry+12<=n;entry+=12) {
        if(tiff.u16(entry)!=274)continue;
        const uint32_t type=tiff.u16(entry+2);
        const uint32_t value=tiff.u32(entry+4)!=1?0:type==3?tiff.u16(entry+8):type==4?tiff.u32(entry+8):0;
        return value>=2&&value<=8?int(value):1;
    }
    return 1;
}
// The EXIF block of an encoded JPEG (the first Exif APP1 before the scan), PNG
// (eXIf, before or after IDAT) or WebP (EXIF chunk) file, found by walking its
// segments or chunks without decoding; empty for every other format (TIFF's
// decoders orient it themselves).
std::string_view exif_block(const uint8_t* p,size_t n) {
    const auto be32=[&](size_t at){return size_t(p[at])<<24|size_t(p[at+1])<<16|size_t(p[at+2])<<8|p[at+3];};
    const auto view=[&](size_t at,size_t length){return std::string_view(reinterpret_cast<const char*>(p+at),length);};
    if(n>=4 && p[0]==0xFF && p[1]==0xD8) {
        for(size_t at=2;at+4<=n && p[at]==0xFF;) {
            const uint8_t marker=p[at+1];
            if(marker==0xFF){++at;continue;} // fill byte
            if(marker==0xDA || marker==0xD9)break;
            if(marker==0x01 || (marker>=0xD0 && marker<=0xD7)){at+=2;continue;}
            const size_t length=size_t(p[at+2])<<8|p[at+3];
            if(length<2 || length>n-at-2)break;
            if(marker==0xE1 && length>=8 && std::memcmp(p+at+4,"Exif\0\0",6)==0)return view(at+4,length-2);
            at+=2+length;
        }
    } else if(n>=8 && std::memcmp(p,"\x89PNG\r\n\x1a\n",8)==0) {
        for(size_t at=8;at+12<=n;) {
            const size_t length=be32(at);
            if(length>n-at-12 || std::memcmp(p+at+4,"IEND",4)==0)break;
            if(std::memcmp(p+at+4,"eXIf",4)==0)return view(at+8,length);
            at+=12+length;
        }
    } else if(n>=12 && std::memcmp(p,"RIFF",4)==0 && std::memcmp(p+8,"WEBP",4)==0) {
        for(size_t at=12;at+8<=n;) {
            const size_t length=size_t(p[at+4])|size_t(p[at+5])<<8|size_t(p[at+6])<<16|size_t(p[at+7])<<24;
            if(length>n-at-8)break;
            if(std::memcmp(p+at,"EXIF",4)==0)return view(at+8,length);
            at+=8+length+(length&1);
        }
    }
    return {};
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
O read_cv(py::dict g,O path) {
    O ext=name(g,"get_ext")(path);
    if(!contains(name(g,"get_opencv_formats")(),ext))return py::none();
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
    if(!contains(name(g,"get_pil_formats")(),name(g,"get_ext")(path)))return py::none();
    O im=name(g,"Image").attr("open")(path);
    if(equal(im.attr("mode"),py::str("P")))im=im.attr("convert")(im.attr("palette").attr("mode"));
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
O read_dds(py::dict g,O path) {
    if(!equal(name(g,"get_ext")(path),py::str(".dds")))return py::none();
    if(!equal(name(g,"platform").attr("system")(),py::str("Windows")))return py::none();
    O png=name(g,"dds_to_png_texconv")(path),image;
    finally_do([&] {image=name(g,"_read_cv")(png);if(!image.is_none())image=name(g,"remove_unnecessary_alpha")(image);},[&]{name(g,"os").attr("remove")(png);});
    return image;
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
// formats (GIF, TGA, AVIF) get a uint8 image in RGB(A) order and their save
// options; OpenCV formats get their parameters and the precision conversion.
struct Encoding {O image; bool pillow=false; py::dict options; py::list params;};
bool pillow_format(const py::dict& g,const O& format_type) {
    return option(g,format_type,"ImageFormat","GIF")||option(g,format_type,"ImageFormat","TGA")||option(g,format_type,"ImageFormat","AVIF");
}
Encoding encoding(py::dict g,O image,py::tuple args) {
    O format_type=args[4];
    Encoding result;
    if(pillow_format(g,format_type)) {
        image=name(g,"to_uint8")(image,py::arg("normalized")=true);
        if(option(g,format_type,"ImageFormat","AVIF")){result.options["quality"]=args[7];result.options["subsampling"]=O(args[18]).attr("value");}
        O channels=index(name(g,"get_h_w_c")(image),2);
        if(equal(channels,py::int_(3)))image=swap_channels(g,image,"COLOR_BGR2RGB");
        else if(equal(channels,py::int_(4)))image=swap_channels(g,image,"COLOR_BGRA2RGBA");
        else if(!equal(channels,py::int_(1)))raise(PyExc_RuntimeError,format("Unsupported number of channels. Saving .{} images is only supported for grayscale, RGB, and RGBA images.",format_type.attr("extension")));
        result.image=image;result.pillow=true;
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
// OpenCV writes 4-sample RGB TIFFs without ExtraSamples (tag 338), which TIFF 6.0
// requires, so readers may take the fourth sample for an unknown one (upstream
// chaiNNer #2950). Adds 338 = SHORT 2 (unassociated alpha: chaiNNer's straight
// alpha) to a classic TIFF's first IFD: a copy of the IFD with the entry in tag
// order goes to the end of the file, on a word boundary, and the header points to
// it; every other byte stays where it was. Other buffers are returned as they are.
O add_extra_samples(O buffer) {
    py::array array=py::reinterpret_borrow<py::array>(buffer);
    const auto* p=static_cast<const uint8_t*>(array.data());
    const size_t n=static_cast<size_t>(array.nbytes());
    const ClassicTiff tiff(p,n);
    if(!tiff.valid)return buffer;
    const size_t entries=tiff.ifd+2,count=tiff.u16(tiff.ifd);
    if(count*12+4>n-entries)return buffer;
    uint32_t samples=0,photometric=0;
    size_t insert=count;
    for(size_t i=0;i<count;++i) {
        const size_t entry=entries+i*12;
        const uint32_t tag=tiff.u16(entry);
        if(tag==338)return buffer;
        if(tag==277)samples=tiff.u16(entry+8);
        if(tag==262)photometric=tiff.u16(entry+8);
        if(tag>338 && insert==count)insert=i;
    }
    const size_t moved=n+(n&1),size=moved+2+(count+1)*12+4;
    if(samples!=4 || photometric!=2 || count==0xFFFF || moved>UINT32_MAX)return buffer;
    O result=own(PyBytes_FromStringAndSize(nullptr,static_cast<Py_ssize_t>(size)));
    auto* out=reinterpret_cast<uint8_t*>(PyBytes_AS_STRING(result.ptr()));
    uint8_t* at=out+n;
    const auto put=[&](uint32_t value,int bytes){for(int i=0;i<bytes;++i)*at++=uint8_t(value>>(8*(tiff.little?i:bytes-1-i)));};
    std::memcpy(out,p,n);
    if(n&1)*at++=0;
    put(uint32_t(count+1),2);
    std::memcpy(at,p+entries,insert*12);at+=insert*12;
    put(338,2);put(3,2);put(1,4);put(2,2);put(0,2);
    std::memcpy(at,p+entries+insert*12,(count-insert)*12+4); // and the next IFD's offset
    at=out+4;put(uint32_t(moved),4);
    return result;
}
// The codec of cv_save for a file extension: fpng for a default PNG, else
// OpenCV's imencode. Returns bytes, or imencode's buffer array.
O encode(py::dict g,O extension,O image,O params) {
    O buffer=py::none();
    // Explicit codec parameters are a distinct OpenCV contract (compression,
    // bilevel, strategy). Only default PNG encoding is fpng-compatible.
    if(equal(extension.attr("lower")(),py::str(".png")) &&
        (PyList_CheckExact(params.ptr()) || PyTuple_CheckExact(params.ptr())) && py::len(params)==0)
        buffer=encode_fpng(image);
    if(buffer.is_none()) {
        O encoded=name(g,"cv2").attr("imencode")(format(".{}",extension),image,params);
        buffer=index(encoded,1);
        O lower=extension.attr("lower")();
        if(equal(lower,py::str(".tiff")) || equal(lower,py::str(".tif")))buffer=add_extra_samples(buffer);
    }
    return buffer;
}
// Image.save(path)'s file handling (Pillow 9.2), for prepared Pillow bytes: open
// "w+b"; when the write fails, close the file, remove it if this save created it
// (ignoring a PermissionError), and re-raise; close after a write.
void write_like_pillow(O path,O data) {
    O os=py::module_::import("os"),filename=py::str(path);
    const bool created=!truth(os.attr("path").attr("exists")(filename));
    O file=graphpy::builtin("open")(filename,"w+b");
    try {file.attr("write")(data);}
    catch(py::error_already_set& error) {
        if(!error.matches(PyExc_Exception))throw;
        GraphHandledException handled(error);
        file.attr("close")();
        if(created) {
            try {os.attr("remove")(filename);}
            catch(py::error_already_set& removal) {if(!removal.matches(PyExc_PermissionError))throw;}
        }
        throw;
    }
    file.attr("close")();
}
void save_image(py::dict g,py::tuple args,O prepared) {
    if(args.size()!=20)throw py::type_error("Image save expects 20 input values");
    O lazy=args[0],base=args[1],relative=args[2],filename=args[3],format_type=args[4];
    O path=name(g,"get_full_path")(base,relative,filename,format_type);
    if(truth(path.attr("exists")())) {
        if(truth(args[19])) {name(g,"logger").attr("debug")(format("Skipping existing file: {}",path));return;}
    } else path.attr("parent").attr("mkdir")(py::arg("parents")=true,py::arg("exist_ok")=true);
    name(g,"logger").attr("debug")(format("Writing image to path: {}",path));
    O image=lazy.attr("value");
    // A prepared value (SP3-P9) holds the bytes the branches below would write, as
    // encoded for the format's own extension, written as each branch writes its file.
    // A path that resolved to another spelling (an existing file or a link) encodes
    // as today and leaves the value unused: a node-level exception to the seam's one
    // decision point, since only the commit knows the resolved path.
    if(!prepared.is_none() && equal(index(py::module_::import("os").attr("path").attr("splitext")(path),1),add(py::str("."),format_type.attr("extension")))) {
        O data=prepared.attr("result")();
        if(pillow_format(g,format_type))write_like_pillow(path,data);
        else with_object(graphpy::builtin("open")(path,"wb"),[&](O file){file.attr("write")(data);}); // as cv_save
        return;
    }
    if(option(g,format_type,"ImageFormat","DDS")) {
        image=name(g,"to_uint8")(image,py::arg("normalized")=true);
        O dds=args[12]; bool legacy=contains(name(g,"LEGACY_TO_DXGI"),dds)||contains(name(g,"PREFER_DX9"),dds);
        name(g,"save_as_dds")(path,image,name(g,"to_dxgi")(dds),
            py::arg("mipmap_levels")=args[16],py::arg("dithering")=args[15],
            py::arg("uniform_weighting")=option(g,args[14],"DDSErrorMetric","UNIFORM"),
            py::arg("minimal_compression")=option(g,args[13],"BC7Compression","BEST_SPEED"),
            py::arg("maximum_compression")=option(g,args[13],"BC7Compression","BEST_QUALITY"),
            py::arg("dx9")=legacy,py::arg("separate_alpha")=args[17]);
        return;
    }
    Encoding encoded=encoding(g,image,args);
    if(encoded.pillow) {
        with_object(name(g,"Image").attr("fromarray")(encoded.image),[&](O im){im.attr("save")(path,**encoded.options);});
        return;
    }
    name(g,"cv_save_image")(path,encoded.image,encoded.params);
}
// Save Image's prepare phase (SP3-P9): the bytes save_image writes for these
// inputs (input 0 the image itself) to a path with the format's own extension,
// encoded in memory; None for DDS, which texconv writes itself.
O save_prepare(py::dict g,py::tuple args) {
    if(args.size()!=20)throw py::type_error("Image save expects 20 input values");
    O format_type=args[4];
    if(option(g,format_type,"ImageFormat","DDS"))return py::none();
    O extension=add(py::str("."),format_type.attr("extension"));
    Encoding encoded=encoding(g,args[0],args);
    if(encoded.pillow) {
        // A buffer has no extension, so the plugin is named, picked as Image.save(path)
        // picks it (Pillow 9.2): preinit(), init() when the extension is unknown, then
        // EXTENSION[extension]; the registry may not be initialized yet.
        O image_module=name(g,"Image"),buffer=py::module_::import("io").attr("BytesIO")();
        image_module.attr("preinit")();
        if(!contains(image_module.attr("EXTENSION"),extension))image_module.attr("init")();
        O pillow=image_module.attr("EXTENSION")[extension];
        with_object(image_module.attr("fromarray")(encoded.image),[&](O im){im.attr("save")(buffer,py::arg("format")=pillow,**encoded.options);});
        return buffer.attr("getvalue")();
    }
    O buffer=encode(g,extension,encoded.image,encoded.params);
    return PyBytes_CheckExact(buffer.ptr())?buffer:O(buffer.attr("tobytes")());
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
    O buffer=encode(g,index(name(g,"split_file_path")(path),2),image,params);
    with_object(graphpy::builtin("open")(path,"wb"),[&](O file){file.attr("write")(buffer);});
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
O dds_png(py::dict g,O path) {
    O prefix=name(g,"uuid").attr("uuid4")().attr("hex");
    O basename=index(name(g,"split_file_path")(path),1);
    O temporary=name(g,"mkdtemp")(py::arg("prefix")="chaiNNer-");
    py::list args;for(const char* v:{"-f","rgba","-srgb","-ft","png","-px"})args.append(v);
    args.append(prefix);args.append("-o");args.append(temporary);args.append(py::str(path));
    name(g,"__run_texconv")(args,"Unable to convert DDS");
    return path_join(name(g,"Path")(temporary),add(prefix,basename)).attr("with_suffix")(".png");
}
void save_dds(py::dict g,py::tuple values) {
    if(values.size()!=10)throw py::type_error("DDS save expects10 input values");
    O path=values[0],image=values[1],format_type=values[2];
    O parts=name(g,"split_file_path")(path),directory=index(parts,0),filename=index(parts,1);
    if(!equal(index(parts,2),py::str(".dds")))raise(PyExc_AssertionError,py::str("The file to save must end with '.dds'"));
    O temporary=name(g,"mkdtemp")(py::arg("prefix")="chaiNNer-");
    finally_do([&] {
        O png=name(g,"os").attr("path").attr("join")(temporary,format("{}.png",filename));
        name(g,"cv_save_image")(png,image,py::list());
        py::list args;args.append("-y");args.append("-f");args.append(format_type);args.append(truth(values[8])?"-dx9":"-dx10");args.append("-m");args.append(py::str(values[3]));args.append("-o");args.append(py::str(directory));
        std::string bc;for(const auto& pair:{std::pair<size_t,char>{4,'u'},{5,'d'},{6,'q'},{7,'x'}})if(truth(values[pair.first]))bc+=pair.second;
        if(!bc.empty()){args.append("-bc");args.append("-"+bc);}
        if(contains(name(g,"SRGB_FORMATS"),format_type))args.append("-srgbi");
        if(truth(values[9]))args.append("-sepalpha");
        args.append(png);name(g,"__run_texconv")(args,"Unable to write DDS");
    },[&]{name(g,"shutil").attr("rmtree")(temporary);});
}
} // namespace
void cn_bind_image_io(py::module_& module) {
    module.def("image_io_ext",&get_ext);module.def("image_io_alpha",&remove_alpha);
    module.def("image_io_read_cv",&read_cv);module.def("image_io_read_pil",&read_pil);module.def("image_io_read_dds",&read_dds);
    module.def("image_io_for_ext",&for_ext);module.def("image_io_load",&load_image);
    module.def("image_io_full_path",&full_path);module.def("image_io_save",&save_image);module.def("image_io_view",&view_image);
    module.def("image_io_save_prepare",&save_prepare);
    module.def("image_io_cv_save",&cv_save);
    module.def("image_io_decode_bytes",&decode_bytes);module.def("image_io_run_texconv",&run_texconv);
    module.def("image_io_dds_png",&dds_png);module.def("image_io_save_dds",&save_dds);
}
