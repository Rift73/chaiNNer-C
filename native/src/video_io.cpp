/* chaiNNer video application policy/state, adapted from its GPL-3.0 sources.
 * FFmpeg codecs/processes, ffmpeg-python's public graph builder, CPython object
 * and pipe interfaces, and NumPy's native zero-copy buffer view remain explicit
 * dependencies. No original Python video algorithm is executed by this unit.
 */
#include "graph_python.hpp"
#include "graph_exception.hpp"
#include <chrono>
#include <cmath>
#include <numbers>
#include <vector>
#include <condition_variable>
#include <memory>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#endif

namespace {
namespace py = pybind11;
using graphpy::O;
using graphpy::truth;
O own(PyObject *p) { if (!p) throw py::error_already_set(); return py::reinterpret_steal<O>(p); }
O global(const py::dict &g, const char *s) { return g[py::str(s)]; }
O builtin(const char *s) { return graphpy::builtin(s); }
O add(const O&a,const O&b) { return own(PyNumber_Add(a.ptr(),b.ptr())); }
O sub(const O&a,const O&b) { return own(PyNumber_Subtract(a.ptr(),b.ptr())); }
O mul(const O&a,const O&b) { return own(PyNumber_Multiply(a.ptr(),b.ptr())); }
O div(const O&a,const O&b) { return own(PyNumber_TrueDivide(a.ptr(),b.ptr())); }
O mod(const O&a,const O&b) { return own(PyNumber_Remainder(a.ptr(),b.ptr())); }
bool eq(const O&a,const O&b) { return truth(own(PyObject_RichCompare(a.ptr(),b.ptr(),Py_EQ))); }
bool cmp(const O&a,const O&b,int op) { return truth(own(PyObject_RichCompare(a.ptr(),b.ptr(),op))); }
O member(const py::dict&g,const char*t,const char*m) { return global(g,t).attr(m); }
py::dict dict(std::initializer_list<std::pair<const char*,O>> entries) {
    py::dict d; for(const auto&v:entries)d[py::str(v.first)]=v.second; return d;
}
O call(const O&f, const py::tuple&a=py::tuple(), const py::dict&k=py::dict()) {
    return own(PyObject_Call(f.ptr(),a.ptr(),k.ptr()));
}
O fmt(const char*p,const O&v) { return py::str(p).attr("format")(v); }
[[noreturn]] void fail(PyObject *type,const char*message) { graphpy::raise(type,py::str(message)); }
O function_label(const O&f) {return py::str("{}.{}()").attr("format")(f.attr("__module__"),f.attr("__qualname__"));}
py::dict keyword_mapping(const O&f,const O&mapping) {
    // Python's ** expansion requires a mapping, rather than dict(iterable).
    // Preserve its dict fast path, duplicate-key rejection, and diagnostics.
    if(PyDict_Check(mapping.ptr()) && Py_TYPE(mapping.ptr())->tp_iter==PyDict_Type.tp_iter)
        return py::reinterpret_steal<py::dict>(own(PyDict_Copy(mapping.ptr())).release());
    O keys;
    try {keys=mapping.attr("keys")();}
    catch(const py::error_already_set&e) {
        if(!e.matches(PyExc_AttributeError))throw;
        graphpy::raise(PyExc_TypeError,py::str("{} argument after ** must be a mapping, not {}").attr("format")(
            function_label(f),py::str(Py_TYPE(mapping.ptr())->tp_name)));
    }
    py::dict result;
    for(py::handle kh:keys) {
        O key=py::reinterpret_borrow<O>(kh);
        if(result.contains(key))graphpy::raise(PyExc_TypeError,py::str("{} got multiple values for keyword argument '{}'").attr("format")(function_label(f),key));
        result[key]=mapping[key];
    }
    return result;
}
py::tuple star_arguments(const O&f,const O&values) {
    if(!Py_TYPE(values.ptr())->tp_iter && !PySequence_Check(values.ptr()))
        graphpy::raise(PyExc_TypeError,py::str("{} argument after * must be an iterable, not {}").attr("format")(
            function_label(f),py::str(Py_TYPE(values.ptr())->tp_name)));
    return py::tuple(values);
}

// True when the CLI reader's autorotation transposes the frames, so they are height x
// width: a display matrix (ffprobe's text form, 9 integers in 16.16) whose rotation, as
// fftools get_rotation derives it from av_display_rotation_get, is a quarter turn.
// The PyAV reader (video.py _upright_filters) inserts the same filters.
bool quarter_turn(const O&stream) {
    for(py::handle sh:stream.attr("get")("side_data_list",py::list())) {
        O text=py::reinterpret_borrow<O>(sh).attr("get")("displaymatrix",py::none());
        if(text.is_none())continue;
        std::vector<double> m;
        for(py::handle line:text.attr("splitlines")()) {
            O parts=py::reinterpret_borrow<O>(line).attr("split")(":");
            if(py::len(parts)!=2)continue;
            for(py::handle v:parts[py::int_(1)].attr("split")())
                m.push_back(builtin("float")(builtin("int")(v)).cast<double>()/65536.0);
        }
        if(m.size()!=9)return false;
        double sx=std::hypot(m[0],m[3]),sy=std::hypot(m[1],m[4]);
        if(sx==0.0||sy==0.0)return false;
        double rotation=-(std::atan2(m[1]/sy,m[0]/sx)*180/std::numbers::pi);
        double theta=-std::round(rotation);
        theta-=360*std::floor(theta/360+0.9/360);
        return std::fabs(theta-90)<1.0||std::fabs(theta-270)<1.0;
    }
    return false;
}

O metadata(const py::dict&g,const O&path,const O&env) {
    O probe = global(g,"ffmpeg").attr("probe")(path, py::arg("cmd")=env.attr("ffprobe"));
    O format=probe.attr("get")("format",py::none());
    if(format.is_none())fail(PyExc_RuntimeError,"Failed to get video format. Please report.");
    O video=py::none();
    for(py::handle sh:probe[py::str("streams")]) {
        O stream=py::reinterpret_borrow<O>(sh);
        if(eq(stream[py::str("codec_type")],py::str("video"))) {video=stream;break;}
    }
    if(video.is_none())fail(PyExc_RuntimeError,"No video stream found in file");
    O width=video.attr("get")("width",py::none());
    if(width.is_none())fail(PyExc_RuntimeError,"No width found in video stream");
    width=builtin("int")(width);
    O height=video.attr("get")("height",py::none());
    if(height.is_none())fail(PyExc_RuntimeError,"No height found in video stream");
    height=builtin("int")(height);
    if(quarter_turn(video))std::swap(width,height);
    O rate=video.attr("get")("r_frame_rate",py::none());
    if(rate.is_none())fail(PyExc_RuntimeError,"No fps found in video stream");
    O numerator=builtin("int")(rate.attr("split")("/")[py::int_(0)]);
    O denominator=builtin("int")(rate.attr("split")("/")[py::int_(1)]);
    O fps=div(numerator,denominator);
    O count=video.attr("get")("nb_frames",py::none());
    if(count.is_none()) {
        O duration=video.attr("get")("duration",py::none());
        if(duration.is_none())duration=format.attr("get")("duration",py::none());
        if(duration.is_none())fail(PyExc_RuntimeError,"No frame count or duration found in video stream. Unable to determine video length. Please report.");
        count=mul(builtin("float")(duration),fps);
    }
    count=builtin("int")(count);
    return global(g,"VideoMetadata")(py::arg("width")=width,py::arg("height")=height,
        py::arg("fps")=fps,py::arg("frame_count")=count);
}
void loader_init(const py::dict&g,const O&self,const O&path,const O&env) {
    self.attr("path")=path; self.attr("ffmpeg_env")=env;
    self.attr("metadata")=global(g,"VideoMetadata").attr("from_file")(path,env);
}
O audio_stream(const py::dict&g,const O&self) {return global(g,"ffmpeg").attr("input")(self.attr("path")).attr("audio");}

O container_encoders(const py::dict&g,const O&self) {
    for(const auto &entry: {"MKV","MP4","MOV","WEBM","AVI","GIF"}) {
        if(!eq(self,member(g,"VideoFormat",entry)))continue;
        if(std::string(entry)=="MKV")return py::make_tuple(member(g,"VideoEncoder","H264"),member(g,"VideoEncoder","H265"),member(g,"VideoEncoder","VP9"),member(g,"VideoEncoder","FFV1"));
        if(std::string(entry)=="MP4")return py::make_tuple(member(g,"VideoEncoder","H264"),member(g,"VideoEncoder","H265"),member(g,"VideoEncoder","VP9"));
        if(std::string(entry)=="MOV")return py::make_tuple(member(g,"VideoEncoder","H264"),member(g,"VideoEncoder","H265"));
        if(std::string(entry)=="WEBM")return py::make_tuple(member(g,"VideoEncoder","VP9"));
        if(std::string(entry)=="AVI")return py::make_tuple(member(g,"VideoEncoder","H264"));
        return py::tuple();
    }
    graphpy::raise(PyExc_ValueError,fmt("Unknown container: {}",self));
}
O encoder_formats(const py::dict&g,const O&self) {
    py::list result;
    for(py::handle fh:global(g,"VideoFormat")) {
        O format=py::reinterpret_borrow<O>(fh);
        if(graphpy::contains(format.attr("encoders"),self))result.append(format);
    }
    return py::tuple(result);
}
O simple_format(const py::dict&g,const O&format,const O&quality) {
    py::dict containers,encoders;
    for(const auto&name: {"MP4_H264","MP4_H265","WEBM","GIF"}) {
        O key=member(g,"SimpleVideoFormat",name);
        const std::string n(name);
        containers[key]=member(g,"VideoFormat",n=="WEBM"?"WEBM":n=="GIF"?"GIF":"MP4");
        encoders[key]=member(g,"VideoEncoder",n=="WEBM"?"VP9":n=="MP4_H265"?"H265":"H264");
    }
    O container=containers[format],encoder=encoders[format];
    O crf=builtin("int")(mul(div(sub(py::int_(100),quality),py::int_(100)),py::int_(51)));
    const char*preset="ULTRA_FAST";
    if(cmp(quality,py::int_(95),Py_GT))preset="VERY_SLOW";
    else if(cmp(quality,py::int_(80),Py_GT))preset="SLOWER";
    else if(cmp(quality,py::int_(60),Py_GT))preset="SLOW";
    else if(cmp(quality,py::int_(50),Py_GE))preset="MEDIUM";
    else if(cmp(quality,py::int_(35),Py_GT))preset="FAST";
    else if(cmp(quality,py::int_(20),Py_GT))preset="VERY_FAST";
    return py::make_tuple(container,encoder,member(g,"VideoPreset",preset),crf);
}

// Once termination starts, attempt every owned-handle cleanup even when a pipe
// close fails (notably BrokenPipeError). The first error remains observable.
void abort_process(const O&p) {
    std::exception_ptr error;
    auto attempt=[&](auto action) {try {action();}catch(...) {if(!error)error=std::current_exception();}};
    attempt([&](){if(p.attr("poll")().is_none())p.attr("terminate")();});
    for(const char*name: {"stdin","stdout","stderr"})attempt([&](){
        O pipe=p.attr(name);if(!pipe.is_none())pipe.attr("close")();
    });
    attempt([&](){
        try {p.attr("wait")(py::arg("timeout")=5);}
        catch(const py::error_already_set&e) {
            O timeout=py::module_::import("subprocess").attr("TimeoutExpired");
            if(!e.matches(timeout.ptr()))throw;
            p.attr("kill")();p.attr("wait")();
        }
    });
    if(error)std::rethrow_exception(error);
}

struct PipeCloser {
    O pipe;
    std::mutex mutex;
    std::condition_variable condition;
    bool done=false;
    std::exception_ptr error;
    explicit PipeCloser(O p):pipe(std::move(p)) {}
};

// A failed graph must leave the frames already handed to FFmpeg just as the
// original writer's buffered-pipe EOF did. This is not successful completion:
// it never runs the audio mux. Only a stalled encoder is terminated. Closing a
// BufferedWriter can itself block, so native ownership includes its close
// worker and an unconditional GIL-free join; a worker is never detached.
void abort_writer(const O&p) {
    if(truth(py::module_::import("sys").attr("is_finalizing")())) {
        // A new GIL-acquiring thread cannot start safely during finalization.
        abort_process(p);return;
    }
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    auto state=std::make_shared<PipeCloser>(p.attr("stdin"));
    std::exception_ptr error;
    auto remember=[&](){if(!error)error=std::current_exception();};
    std::thread closer;
    try {
        closer=std::thread([state](){
            // The caller retains state until after join, so no Python field
            // of PipeCloser can be decref'd by the worker after releasing GIL.
            py::gil_scoped_acquire gil;
            try {if(!state->pipe.is_none())state->pipe.attr("close")();}
            catch(...) {state->error=std::current_exception();}
            {std::lock_guard<std::mutex> lock(state->mutex);state->done=true;}
            state->condition.notify_one();
        });
    } catch(...) {
        error=std::current_exception();
        try {abort_process(p);}catch(...) {}
        std::rethrow_exception(error);
    }
    struct Join {
        std::thread&thread;
        ~Join() {if(thread.joinable()){py::gil_scoped_release release;thread.join();}}
    } join{closer};
    bool closed=false;
    {
        py::gil_scoped_release release;
        std::unique_lock<std::mutex> lock(state->mutex);
        closed=state->condition.wait_until(lock,deadline,[&](){return state->done;});
    }
    const O timeout=py::module_::import("subprocess").attr("TimeoutExpired");
    auto wait=[&](double seconds) {
        try {p.attr("wait")(py::arg("timeout")=seconds);return true;}
        catch(const py::error_already_set&e) {if(!e.matches(timeout.ptr()))remember();return false;}
    };
    bool reaped=false;
    if(closed) {
        if(state->error)error=state->error;
        const double seconds=std::chrono::duration<double>(deadline-std::chrono::steady_clock::now()).count();
        reaped=wait(seconds>0?seconds:0);
    }
    if(!reaped) {
        bool terminated=true;
        try {if(p.attr("poll")().is_none())p.attr("terminate")();}
        catch(...) {remember();terminated=false;}
        if(terminated)reaped=wait(5);
        if(!reaped) {
            try {p.attr("kill")();}catch(...) {remember();}
            reaped=wait(5);
        }
    }
    bool pending;
    {std::lock_guard<std::mutex> lock(state->mutex);pending=!state->done;}
#ifdef _WIN32
    if(pending) {
        // Cancel only synchronous I/O issued by this owned close thread. This
        // also unblocks it if process termination failed. ERROR_NOT_FOUND can
        // occur just before the worker enters WriteFile, so retry while waiting
        // for completion. Polling is bounded; the mandatory join is not a claim
        // of a universal shutdown deadline if the OS denies cancellation or a
        // foreign Python close callback never returns. Never detach that worker.
        const auto cancellation_deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
        while(pending) {
            if(!CancelSynchronousIo(closer.native_handle())) {
                const DWORD status=GetLastError();
                if(status!=ERROR_NOT_FOUND) {
                    try {PyErr_SetFromWindowsErr(static_cast<int>(status));throw py::error_already_set();}
                    catch(...) {remember();}
                }
            }
            {
                py::gil_scoped_release release;
                std::unique_lock<std::mutex> lock(state->mutex);
                pending=!state->condition.wait_for(lock,std::chrono::milliseconds(10),[&](){return state->done;});
            }
            if(std::chrono::steady_clock::now()>=cancellation_deadline)break;
        }
    }
#else
    (void)pending;
#endif
    {py::gil_scoped_release release;closer.join();}
    if(!error && state->error)error=state->error;
    // After cancellation the raw pipe finally closes, which can let an encoder
    // whose first termination attempt failed consume EOF and exit naturally.
    if(!reaped) {
        try {p.attr("wait")(py::arg("timeout")=5);}
        catch(...) {remember();}
    }
    if(error)std::rethrow_exception(error);
}

// Own only processes created by this writer; never a process merely assigned
// by an external caller. Abandonment stops the owned encoder but never muxes.
class WriterResource {
public:
    O process=py::none();
    bool completed=false;
    bool aborted=false;
    void abort() {
        if(completed || aborted)return;
        aborted=true;
        if(process.is_none())return;
        O p=process; process=py::none();
        abort_writer(p);
    }
    ~WriterResource() {
        if(!Py_IsInitialized())return;
        try {abort();}
        catch(py::error_already_set &e) {e.discard_as_unraisable("video writer owned-process cleanup");}
        catch(...) {}
    }
};
std::shared_ptr<WriterResource> resource(const O&self) {
    if(py::hasattr(self,"_native_video_resource"))return self.attr("_native_video_resource").cast<std::shared_ptr<WriterResource>>();
    auto owned=std::make_shared<WriterResource>();
    self.attr("_native_video_resource")=py::cast(owned); return owned;
}
void writer_start(const py::dict&g,const O&self,const O&width,const O&height) {
    if(!self.attr("out").is_none())return;
    if(resource(self)->aborted)fail(PyExc_RuntimeError,"Video writer was aborted");
    if(graphpy::contains(py::make_tuple(member(g,"VideoEncoder","H264"),member(g,"VideoEncoder","H265")),self.attr("encoder")))
        graphpy::require(eq(mod(height,py::int_(2)),py::int_(0)) && eq(mod(width,py::int_(2)),py::int_(0)),
            fmt("The \"{}\" encoder requires an even-number frame resolution.",self.attr("encoder").attr("value")));
    try {
        O stream=global(g,"ffmpeg").attr("input")("pipe:",py::arg("format")="rawvideo",py::arg("pix_fmt")="bgr24",
            py::arg("s")=py::str("{}x{}").attr("format")(width,height),py::arg("r")=self.attr("fps"),py::arg("loglevel")="error");
        O output=stream.attr("output");
        py::dict params=keyword_mapping(output,self.attr("output_params"));
        if(params.contains("loglevel")) {
            O label=py::str("{}.{}() got multiple values for keyword argument 'loglevel'").attr("format")(output.attr("__module__"),output.attr("__qualname__"));
            graphpy::raise(PyExc_TypeError,label);
        }
        params["loglevel"]="error";
        stream=call(output,py::tuple(),params).attr("overwrite_output")();
        O global_args=stream.attr("global_args");
        stream=call(global_args,star_arguments(global_args,self.attr("global_params")));
        O process=stream.attr("run_async")(py::arg("pipe_stdin")=true,py::arg("pipe_stdout")=false,py::arg("cmd")=self.attr("ffmpeg_env").attr("ffmpeg"));
        auto state=resource(self);state->process=process;state->completed=false;
        self.attr("out")=process;
    } catch(const py::error_already_set&e) {
        if(!e.matches(PyExc_Exception))throw;
        GraphHandledException handled(e);
        global(g,"logger").attr("warning")("Failed to open video writer",py::arg("exc_info")=e.value());
    }
}
O frame_payload(const py::dict&g,const O&frame) {
    // Keep ndarray subclasses and custom quantizers on their original tobytes
    // protocol. Ordinary contiguous frames can expose exactly those C-order
    // bytes without allocating another full frame. The memoryview owns its
    // exporter through every blocking/partial write and denies sink mutation.
    O array_type=global(g,"np").attr("ndarray");
    if(reinterpret_cast<PyObject*>(Py_TYPE(frame.ptr()))==array_type.ptr()) {
        O view=own(PyMemoryView_FromObject(frame.ptr()));
        const Py_buffer*buffer=PyMemoryView_GET_BUFFER(view.ptr());
        if(buffer->len>0 && PyBuffer_IsContiguous(buffer,'C'))
            return view.attr("cast")("B").attr("toreadonly")();
    }
    return frame.attr("tobytes")();
}
void write_frame_payload(const O&write,const O&payload) {
    // A blocking BufferedWriter normally consumes the full buffer. Honor a
    // short write without dropping the tail, and reject nonblocking/no-progress
    // results instead of spinning or silently emitting a truncated frame.
    if(!PyBytes_CheckExact(payload.ptr()) && !PyMemoryView_Check(payload.ptr())) {
        write(payload); // Preserve the public custom tobytes/write protocol.
        return;
    }
    const Py_ssize_t total=PyObject_Length(payload.ptr());
    if(total<0)throw py::error_already_set();
    Py_ssize_t offset=0;
    O remaining=payload,view=py::none();
    for(;;) {
        O written=write(remaining);
        if(written.is_none())fail(PyExc_BlockingIOError,"Video pipe write made no progress");
        const Py_ssize_t count=PyNumber_AsSsize_t(written.ptr(),PyExc_OverflowError);
        if(count==-1 && PyErr_Occurred())throw py::error_already_set();
        if(count<0 || count>total-offset)fail(PyExc_OSError,"Invalid video pipe write count");
        if(count==0 && offset<total)fail(PyExc_BlockingIOError,"Video pipe write made no progress");
        offset+=count;
        if(offset==total)return;
        if(view.is_none())view=PyMemoryView_Check(payload.ptr())?payload:own(PyMemoryView_FromObject(payload.ptr()));
        remaining=view[py::slice(py::int_(offset),py::int_(total),py::int_(1))];
    }
}
void writer_frame(const py::dict&g,const O&self,const O&image,const O&prepared) {
    if(self.attr("out").is_none()) {
        O shape=global(g,"get_h_w_c")(image);
        self.attr("start")(shape[py::int_(1)],shape[py::int_(0)]);
    }
    // A prepared frame (SP3-P9) is to_uint8's result computed ahead: only the payload.
    O frame=prepared.is_none()?O(global(g,"to_uint8")(image,py::arg("normalized")=true)):O(prepared.attr("result")());
    if(!self.attr("out").is_none() && !self.attr("out").attr("stdin").is_none()) {
        O write=self.attr("out").attr("stdin").attr("write");
        write_frame_payload(write,frame_payload(g,frame));
    }
    else fail(PyExc_RuntimeError,"Failed to open video writer");
}

// Load Video's audio is ffmpeg.input(path).audio: the file the mux reads it from.
O audio_source(const O&self) {return self.attr("audio").attr("node").attr("kwargs")[py::str("filename")];}
py::list audio_streams(const py::dict&g,const O&self) {
    // The audio streams of the audio's source file, as ffprobe reports them.
    O probe=global(g,"ffmpeg").attr("probe")(audio_source(self),py::arg("cmd")=self.attr("ffmpeg_env").attr("ffprobe"));
    py::list streams;
    for(py::handle sh:probe[py::str("streams")]) {
        O stream=py::reinterpret_borrow<O>(sh);
        if(eq(stream.attr("get")("codec_type"),py::str("audio")))streams.append(stream);
    }
    return streams;
}
py::dict audio_options(const py::dict&g,const O&self,const O&settings) {
    py::dict p=dict({{"vcodec",py::str("copy")},{"acodec",py::str("copy")}});
    if(eq(self.attr("container"),member(g,"VideoFormat","WEBM"))) {
        if(graphpy::contains(py::make_tuple(member(g,"AudioSettings","TRANSCODE"),member(g,"AudioSettings","AUTO")),settings)) {
            // libopus takes at most 256 kb/s per channel, so min(320k, 256k x channels):
            // mono gets 256k, where 320k failed; two or more channels keep 320k.
            p["acodec"]="libopus";p["b:a"]="320k";
            for(py::handle sh:audio_streams(g,self))
                if(eq(py::reinterpret_borrow<O>(sh).attr("get")("channels"),py::int_(1)))p["b:a"]="256k";
        } else graphpy::raise(PyExc_ValueError,fmt("WebM does not support {}",settings));
    } else if(eq(settings,member(g,"AudioSettings","TRANSCODE"))) {
        p["acodec"]="aac";p["b:a"]="320k";
    }
    p["loglevel"]="error"; // FFmpeg's stderr is then its error alone, which a failed mux reports.
    return p;
}
void mux_audio(const py::dict&g,const O&self) {
    // chaiNNer's own FFmpeg muxes the audio after the video (upstream chaiNNer v0.25.1).
    // Auto copies the audio and, when the container cannot hold the copy, transcodes it
    // as Transcode does. Audio the mux cannot carry fails the run instead of leaving a
    // silent video (upstream chaiNNer #3331); a source without audio has none to carry.
    if(self.attr("audio").is_none())return;
    O container=self.attr("container");
    py::dict params=audio_options(g,self,self.attr("audio_settings")); // A WebM Copy raises before the mux.
    bool transcode_on_failure=eq(self.attr("audio_settings"),member(g,"AudioSettings","AUTO")) && eq(params["acodec"],py::str("copy"));
    O path=self.attr("save_path"),os=global(g,"os"),ffmpeg=global(g,"ffmpeg"),logger=global(g,"logger");
    if(!truth(os.attr("path").attr("exists")(path))) {logger.attr("error")(fmt("Video file not found at {}",path));return;}
    O parts=os.attr("path").attr("splitext")(path);
    O av=py::str("{}_av{}").attr("format")(parts[py::int_(0)],parts[py::int_(1)]);
    auto discard=[&]() {
        // A failed mux can leave FFmpeg's empty or partial output behind.
        if(!truth(os.attr("path").attr("exists")(av)))return;
        try {os.attr("remove")(av);}
        catch(const py::error_already_set&cleanup) {
            if(!cleanup.matches(PyExc_Exception))throw;
            GraphHandledException cleanup_handled(cleanup);
            logger.attr("warning")(fmt("Failed to cleanup temporary file: {}",builtin("str")(cleanup.value())));
        }
    };
    try {
        logger.attr("debug")(fmt("Attempting to process audio from: {}",self.attr("audio")));
        O video=ffmpeg.attr("input")(builtin("str")(path));
        bool muxed=false;
        for(;;) {
            O output=call(ffmpeg.attr("output"),py::make_tuple(video,self.attr("audio"),builtin("str")(av)),params).attr("overwrite_output")();
            try {ffmpeg.attr("run")(output,py::arg("cmd")=self.attr("ffmpeg_env").attr("ffmpeg"),py::arg("capture_stdout")=true,py::arg("capture_stderr")=true);muxed=true;break;}
            catch(const py::error_already_set&e) {
                if(!e.matches(ffmpeg.attr("Error").ptr()))throw;
                GraphHandledException handled(e);
                O stderr_bytes=e.value().attr("stderr");
                O message=stderr_bytes.is_none()?O(builtin("str")(e.value())):O(stderr_bytes.attr("decode")("utf-8","replace").attr("strip")());
                py::list streams=audio_streams(g,self);
                if(streams.empty()) {
                    logger.attr("warning")(fmt("The audio source has no audio stream, so the video is saved without audio: {}",audio_source(self)));
                    break;
                }
                O codec=streams[0].attr("get")("codec_name","unknown");
                O ext=container.attr("value");
                if(transcode_on_failure) {
                    logger.attr("info")(py::str("Auto transcodes the {} audio, which the .{} file cannot hold as a copy: {}").attr("format")(codec,ext,message));
                } else if(eq(params["acodec"],py::str("copy"))) {
                    graphpy::raise(PyExc_RuntimeError,py::str("Save Video could not copy the {} audio into the .{} file. Set Audio to Auto or Transcode to re-encode it. FFmpeg: {}").attr("format")(codec,ext,message));
                } else {
                    graphpy::raise(PyExc_RuntimeError,py::str("Save Video could not transcode the {} audio to {} for the .{} file. FFmpeg: {}").attr("format")(codec,params["acodec"],ext,message));
                }
            }
            params=audio_options(g,self,member(g,"AudioSettings","TRANSCODE"));transcode_on_failure=false;
        }
        if(!muxed)discard();
        else if(truth(os.attr("path").attr("exists")(av)))os.attr("replace")(av,path); // Atomic: the video survives a failure.
        else graphpy::raise(PyExc_RuntimeError,fmt("Expected output file not created: {}",av));
    } catch(const py::error_already_set&e) {
        if(!e.matches(PyExc_Exception))throw;
        GraphHandledException handled(e);
        discard();
        throw;
    }
}
void writer_close(const py::dict&g,const O&self) {
    auto state=resource(self);
    if(state->completed || state->aborted)return; // Completion/mux occurs once; abort never commits.
    if(!self.attr("out").is_none()) {
        if(!self.attr("out").attr("stdin").is_none())self.attr("out").attr("stdin").attr("close")();
        self.attr("out").attr("wait")();
    }
    mux_audio(g,self);
    state->completed=true;
}

O save_video(const py::dict&g,const O&context,O unused,const O&directory,const O&name,const O&simplicity,
    O container,O encoder,O preset,O crf,O additional,const O&simple,const O&quality,const O&fps,O audio,const O&audio_settings) {
    (void)unused;
    if(eq(simplicity,member(g,"Simplicity","SIMPLE"))) {
        O settings=global(g,"get_simple_format")(simple,quality);
        container=settings[py::int_(0)];encoder=settings[py::int_(1)];preset=settings[py::int_(2)];crf=settings[py::int_(3)];
    }
    O filename=py::str("{}.{}").attr("format")(name,container.attr("ext"));
    O path=div(directory,filename).attr("resolve")();
    path.attr("parent").attr("mkdir")(py::arg("parents")=true,py::arg("exist_ok")=true);
    py::dict params=dict({{"filename",builtin("str")(path)},{"pix_fmt",py::str("yuv420p")},{"r",fps},{"movflags",py::str("faststart")}});
    if(graphpy::contains(container.attr("encoders"),encoder)) {
        params["vcodec"]=encoder.attr("value");
        O allowed=global(g,"PARAMETERS")[encoder];
        if(graphpy::contains(allowed,py::str("preset")))params["preset"]=preset.attr("value");
        if(graphpy::contains(allowed,py::str("crf")))params["crf"]=crf;
    }
    py::list globals;
    if(eq(simplicity,member(g,"Simplicity","ADVANCED")) && !additional.is_none()) {
        additional=add(py::str(" "),py::str(" ").attr("join")(additional.attr("split")()));
        O all=additional.attr("split")(" -");
        O values=all[py::slice(py::int_(1),py::none(),py::none())];
        for(py::handle ph:values) {
            O parameter=py::reinterpret_borrow<O>(ph),key=parameter,value=py::none();
            O pieces=parameter.attr("split")(" ");
            if(py::len(pieces)==2) {key=pieces[py::int_(0)];value=pieces[py::int_(1)];}
            if(!value.is_none()) {
                for(const char*blocked: {"filename","vcodec","crf","preset","c:"}) {
                    if(!truth(key.attr("startswith")(blocked)))params[key]=value;
                    else graphpy::raise(PyExc_ValueError,fmt("Duplicate parameter: -{}",parameter));
                }
            } else globals.append(fmt("-{}",parameter));
        }
    }
    if(eq(container,member(g,"VideoFormat","GIF")))audio=py::none();
    O writer=global(g,"Writer")(py::arg("container")=container,py::arg("encoder")=encoder,py::arg("fps")=fps,
        py::arg("audio")=audio,py::arg("audio_settings")=audio_settings,py::arg("save_path")=builtin("str")(path),
        py::arg("output_params")=params,py::arg("global_params")=globals,
        py::arg("ffmpeg_env")=global(g,"FFMpegEnv").attr("get_integrated")(context.attr("storage_dir")));
    auto state=resource(writer);
    context.attr("add_cleanup")(py::cpp_function([state](){state->abort();}));
    O iterate=py::cpp_function([writer](const O&frame){writer.attr("write_frame")(frame);});
    O complete=py::cpp_function([writer](){writer.attr("close")();});
    // The collector carries its writer: Save Video's commit phase (SP3-P9) writes a
    // prepared frame through it.
    return global(g,"VideoCollector")(py::arg("on_iterate")=iterate,py::arg("on_complete")=complete,py::arg("writer")=writer);
}

class FrameIterator {
    py::dict globals_;
    O loader_,process_=py::none(),width_=py::none(),height_=py::none();
    bool started_=false,entered_=false,closed_=false,running_=false;
    void start() {
        started_=true;
        process_=global(globals_,"ffmpeg").attr("input")(loader_.attr("path"))
            .attr("output")("pipe:",py::arg("format")="rawvideo",py::arg("pix_fmt")="bgr24",
                py::arg("sws_flags")="lanczos+accurate_rnd+full_chroma_int+full_chroma_inp+bitexact",
                py::arg("loglevel")="error")
            .attr("run_async")(py::arg("pipe_stdout")=true,py::arg("pipe_stderr")=false,
                py::arg("cmd")=loader_.attr("ffmpeg_env").attr("ffmpeg"));
        if(!truth(builtin("isinstance")(process_,global(globals_,"subprocess").attr("Popen")))) {PyErr_SetNone(PyExc_AssertionError);throw py::error_already_set();}
        process_.attr("__enter__")();entered_=true;
        if(!truth(builtin("isinstance")(process_.attr("stdout"),global(globals_,"BufferedIOBase")))) {PyErr_SetNone(PyExc_AssertionError);throw py::error_already_set();}
        width_=loader_.attr("metadata").attr("width");
        height_=loader_.attr("metadata").attr("height");
    }
    bool finish(const O&type=py::none(),const O&value=py::none(),const O&trace=py::none()) {
        closed_=true;
        if(!entered_)return false;
        entered_=false;
        try {return truth(process_.attr("__exit__")(type,value,trace));}
        catch(const py::error_already_set&e) {
            // Popen.__exit__ can itself fail while closing a broken pipe. Keep
            // that error, but still reap the child we created before returning.
            GraphHandledException handled(e);
            try {abort_process(process_);}
            catch(const py::error_already_set&cleanup) {
                global(globals_,"logger").attr("warning")("Failed to clean up video reader",py::arg("exc_info")=cleanup.value());
            }
            throw;
        }
    }
public:
    FrameIterator(py::dict g,O loader):globals_(std::move(g)),loader_(std::move(loader)) {}
    O next() {
        if(running_)fail(PyExc_ValueError,"generator already executing");
        if(closed_)throw py::stop_iteration();
        struct Running {bool&v;explicit Running(bool&b):v(b){v=true;}~Running(){v=false;}} guard(running_);
        try {
            if(!started_)start();
            // D8 (SP4c): the frame is read into a fresh NumPy buffer, which the NumPy pool
            // serves. BufferedReader.readinto fills it as read(size) did: it reads until
            // the frame is full or the pipe ends.
            O np=global(globals_,"np");
            O size=mul(mul(width_,height_),py::int_(3));
            O buffer=np.attr("empty")(size,np.attr("uint8"));
            O count=process_.attr("stdout").attr("readinto")(buffer);
            if(!truth(count)) {
                global(globals_,"logger").attr("debug")("Can't receive frame (stream end?). Exiting ...");
                finish();throw py::stop_iteration();
            }
            py::list shape;shape.append(height_);shape.append(width_);shape.append(py::int_(3));
            // The frame stays immutable, as the view of the read's bytes was: the buffer is
            // read-only, so its view is and cannot be made writeable again.
            buffer.attr("setflags")(py::arg("write")=false);
            // A short read reshapes only the bytes read: today's frombuffer(...).reshape error.
            O data=cmp(count,size,Py_LT)?O(buffer[py::slice(py::none(),count,py::none())]):buffer;
            return data.attr("reshape")(shape);
        } catch(const py::error_already_set&e) {
            if(closed_)throw;
            GraphHandledException handled(e);
            if(entered_) {if(finish(e.type(),e.value(),e.trace()?e.trace():O(py::none())))throw py::stop_iteration();}
            else {closed_=true;if(!process_.is_none())abort_process(process_);}
            throw;
        }
    }
    void close() {
        if(running_)fail(PyExc_ValueError,"generator already executing");
        if(closed_)return;
        struct Running {bool&v;explicit Running(bool&b):v(b){v=true;}~Running(){v=false;}} guard(running_);
        closed_=true;entered_=false;
        if(!process_.is_none())abort_process(process_);
    }
    ~FrameIterator() {
        if(!Py_IsInitialized())return;
        try {close();}
        catch(py::error_already_set&e) {e.discard_as_unraisable("video reader owned-process cleanup");}
        catch(...) {}
    }
};

class IndexedFrames {
    O loader_,iterator_=py::none(),use_limit_,limit_,index_=py::int_(0);
    bool yielded_=false,closed_=false,running_=false;
    void finish() {
        if(closed_)return;
        closed_=true;
        if(!iterator_.is_none() && py::hasattr(iterator_,"close"))iterator_.attr("close")();
    }
public:
    IndexedFrames(O loader,O use,O limit):loader_(std::move(loader)),use_limit_(std::move(use)),limit_(std::move(limit)) {}
    void close() {
        if(running_)fail(PyExc_ValueError,"generator already executing");
        // CPython 3.14.8 gen_close: the original supplier is suspended at a yield
        // outside any try block, so close() marks it finished and then clears its
        // frame, releasing the reader, whose finalizer runs its cleanup (errors
        // unraisable). A re-entry from that cleanup finds the supplier finished.
        if(closed_)return;
        closed_=true;
        iterator_=py::none();
    }
    O next() {
        if(running_)fail(PyExc_ValueError,"generator already executing");
        if(closed_)throw py::stop_iteration();
        struct Running {bool&v;explicit Running(bool&b):v(b){v=true;}~Running(){v=false;}} guard(running_);
        try {
            if(iterator_.is_none())iterator_=builtin("iter")(loader_.attr("stream_frames")());
            if(yielded_) {
                // The original limit check is after yield, including limit <= 0.
                if(truth(use_limit_) && cmp(add(index_,py::int_(1)),limit_,Py_GE)) {finish();throw py::stop_iteration();}
                index_=add(index_,py::int_(1));
            }
            O frame=builtin("next")(iterator_);yielded_=true;
            return py::make_tuple(frame,index_);
        } catch(const py::error_already_set&e) {
            GraphHandledException handled(e);finish();throw;
        }
    }
    ~IndexedFrames() {
        if(!Py_IsInitialized())return;
        try {close();}
        catch(py::error_already_set&e) {e.discard_as_unraisable("video indexed iterator cleanup");}
        catch(...) {}
    }
};

O load_video(const py::dict&g,const O&context,const O&path,const O&use_limit,const O&limit) {
    O parts=global(g,"split_file_path")(path);
    O loader=global(g,"VideoLoader")(path,global(g,"FFMpegEnv").attr("get_integrated")(context.attr("storage_dir")));
    O count=loader.attr("metadata").attr("frame_count");
    if(truth(use_limit))count=builtin("min")(count,limit);
    O audio=loader.attr("get_audio_stream")();
    O supplier=py::cpp_function([loader,context,use_limit,limit](){
        auto it=std::make_shared<IndexedFrames>(loader,use_limit,limit);
        context.attr("add_cleanup")(py::cpp_function([it](){it->close();}));
        return py::cast(it);
    });
    O generator=global(g,"Generator").attr("from_iter")(py::arg("supplier")=supplier,py::arg("expected_length")=count)
        .attr("with_metadata")(loader.attr("metadata"));
    generator.attr("source_paths")=py::tuple(); // Frames come from the FFmpeg pipe.
    return py::make_tuple(generator,parts[py::int_(0)],parts[py::int_(1)],loader.attr("metadata").attr("fps"),audio);
}
} // namespace

void cn_bind_video_io(pybind11::module_&m) {
    py::class_<WriterResource,std::shared_ptr<WriterResource>>(m,"_VideoWriterResource").def("abort",&WriterResource::abort);
    py::class_<FrameIterator,std::shared_ptr<FrameIterator>>(m,"_VideoFrames")
        .def("__iter__",[](const std::shared_ptr<FrameIterator>&s){return s;})
        .def("__next__",&FrameIterator::next).def("close",&FrameIterator::close);
    py::class_<IndexedFrames,std::shared_ptr<IndexedFrames>>(m,"_VideoIndexedFrames")
        .def("__iter__",[](const std::shared_ptr<IndexedFrames>&s){return s;})
        .def("__next__",&IndexedFrames::next).def("close",&IndexedFrames::close);
    m.def("video_metadata",&metadata);m.def("video_loader_init",&loader_init);m.def("video_audio_stream",&audio_stream);
    m.def("video_stream_frames",[](const py::dict&g,const O&loader){return std::make_shared<FrameIterator>(g,loader);});
    m.def("video_container_encoders",&container_encoders);m.def("video_encoder_formats",&encoder_formats);
    m.def("video_simple_format",&simple_format);m.def("video_writer_start",&writer_start);m.def("video_writer_frame",&writer_frame);
    m.def("video_writer_close",&writer_close);
    m.def("video_load",&load_video);m.def("video_save",&save_video);
}
