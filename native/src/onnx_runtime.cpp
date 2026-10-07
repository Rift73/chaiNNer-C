/* Native ONNX host resource ownership, metadata, and dense tensor coordination.
 * This is a host adapter, NOT a replacement implementation of the ORT engine.
 * Uses its public API22; the pinned MIT API header/license live in
 * native/third_party/onnxruntime. No private Python binding address is used.
 */
#include "onnx_runtime.h"
#include "../third_party/onnxruntime/onnxruntime_c_api.h"
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
struct Failure : std::runtime_error {
    int code;
    Failure(int value, const std::string &message) : std::runtime_error(message), code(value) {}
};
void ensure(bool condition, const char *message) {
    if (!condition) throw Failure(ORT_INVALID_ARGUMENT, message);
}
bool span(const void *data,size_t bytes,size_t alignment=1) {
    uintptr_t address=reinterpret_cast<uintptr_t>(data);
    return (!bytes || data) && bytes<=static_cast<size_t>(PTRDIFF_MAX) &&
        address<=UINTPTR_MAX-bytes && address%alignment==0;
}
void ort_check(const OrtApi *api, OrtStatus *status) {
    if (!status) return;
    int code = api->GetErrorCode(status);
    std::string message;
    try { message = api->GetErrorMessage(status); }
    catch (...) { api->ReleaseStatus(status); throw; }
    api->ReleaseStatus(status);
    throw Failure(code, message);
}
void error_text(char *dst, size_t capacity, const char *message) noexcept {
    if (!dst || !capacity) return;
    size_t length = std::min(capacity-1, std::strlen(message));
    std::memcpy(dst, message, length); dst[length] = 0;
}
template<class F> int guarded(char *error, size_t capacity, F &&operation) noexcept {
    error_text(error, capacity, "");
    try { operation(); return ORT_OK; }
    catch (const Failure &e) { error_text(error,capacity,e.what()); return e.code; }
    catch (const std::bad_alloc &) { error_text(error,capacity,"Native ONNX allocation failed"); return ORT_FAIL; }
    catch (const std::exception &e) { error_text(error,capacity,e.what()); return ORT_FAIL; }
    catch (...) { error_text(error,capacity,"Unexpected native ONNX exception"); return ORT_FAIL; }
}
template<class T> struct Owned {
    T *value = nullptr;
    void (ORT_API_CALL *release)(T *);
    explicit Owned(void (ORT_API_CALL *fn)(T *)) : release(fn) {}
    ~Owned() { if (value) release(value); }
    Owned(const Owned &) = delete;
    Owned &operator=(const Owned &) = delete;
};
struct Runtime {
    HMODULE module = nullptr;
    const OrtApi *api = nullptr;
    OrtEnv *env = nullptr;
    std::string version;
    ~Runtime() {
        if (env) api->ReleaseEnv(env);
        if (module) FreeLibrary(module);
    }
};
struct Session {
    std::shared_ptr<Runtime> runtime;
    OrtSession *value = nullptr;
    std::string metadata;
    ~Session() { if (value) runtime->api->ReleaseSession(value); }
};
struct InputValue {
    std::shared_ptr<Runtime> runtime;
    std::vector<uint8_t> bytes;
    OrtValue *value = nullptr;
    ~InputValue() { if (value) runtime->api->ReleaseValue(value); }
};
struct Result {
    std::shared_ptr<Session> session;
    std::vector<std::unique_ptr<InputValue>> inputs;
    std::vector<OrtValue *> values;
    ~Result() { for (OrtValue *v : values) if (v) session->runtime->api->ReleaseValue(v); }
};
struct Registry {
    std::mutex mutex;
    uint64_t next = 1;
    std::unordered_map<uint64_t,std::shared_ptr<Session>> sessions;
    std::unordered_map<uint64_t,std::shared_ptr<Result>> results;
    std::mutex runtime_mutex;
    std::unordered_map<std::wstring,std::weak_ptr<Runtime>> runtimes;
};
Registry &registry() {
    // Deliberately no static destructor: ORT prohibits session destruction in
    // DllMain. Python finalizers/explicit close release every normal resource.
    static Registry *value = new Registry;
    return *value;
}
std::shared_ptr<Runtime> runtime(const wchar_t *path) {
    ensure(path && *path, "ONNX Runtime library path is required");
    std::wstring key(path);
    ensure((key.size() >= 3 && key[1] == L':' && (key[2] == L'\\' || key[2] == L'/')) ||
        (key.size() >= 2 && key[0] == L'\\' && key[1] == L'\\'), "ONNX Runtime library path must be absolute");
    auto &r = registry();
    std::lock_guard lock(r.runtime_mutex);
    if (auto previous = r.runtimes[key].lock()) return previous;
    auto result = std::make_shared<Runtime>();
    result->module = LoadLibraryExW(path,nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!result->module) throw Failure(ORT_FAIL,"Cannot load ONNX Runtime library (Windows error " + std::to_string(GetLastError()) + ")");
    using GetBase = const OrtApiBase *(ORT_API_CALL *)(void);
    FARPROC address = GetProcAddress(result->module,"OrtGetApiBase");
    ensure(address != nullptr, "ONNX Runtime does not export OrtGetApiBase");
    GetBase get_base;
    static_assert(sizeof(get_base) == sizeof(address));
    std::memcpy(&get_base,&address,sizeof(get_base));
    const OrtApiBase *base = get_base();
    result->api = base->GetApi(ORT_API_VERSION);
    ensure(result->api != nullptr,"ONNX Runtime does not support public API22");
    result->version = base->GetVersionString();
    ort_check(result->api,result->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING,"Default",&result->env));
    r.runtimes[key] = result;
    return result;
}
template<class T> std::shared_ptr<T> lookup(uint64_t id, const std::unordered_map<uint64_t,std::shared_ptr<T>> &map) {
    auto found = map.find(id);
    ensure(found != map.end(),"Native ONNX handle is closed or invalid");
    return found->second;
}
std::shared_ptr<Session> session_for(uint64_t id) {
    auto &r=registry(); std::lock_guard lock(r.mutex); return lookup(id,r.sessions);
}
std::shared_ptr<Result> result_for(uint64_t id) {
    auto &r=registry(); std::lock_guard lock(r.mutex); return lookup(id,r.results);
}
uint64_t next_handle(Registry &r) {
    ensure(r.next != UINT64_MAX,"Native ONNX handle space exhausted");
    return r.next++;
}
size_t element_bytes(int type) {
    switch (type) {
    case 1: case 6: case 12: return 4;
    case 2: case 3: case 9: return 1;
    case 4: case 5: case 10: return 2;
    case 7: case 11: case 13: return 8;
    default: throw Failure(ORT_NOT_IMPLEMENTED,"Native ONNX host supports dense standard numeric tensors only");
    }
}
const char *type_name(int type) {
    switch (type) {
    case 1:return "tensor(float)"; case 2:return "tensor(uint8)";
    case 3:return "tensor(int8)"; case 4:return "tensor(uint16)";
    case 5:return "tensor(int16)"; case 6:return "tensor(int32)";
    case 7:return "tensor(int64)"; case 9:return "tensor(bool)";
    case 10:return "tensor(float16)"; case 11:return "tensor(double)";
    case 12:return "tensor(uint32)"; case 13:return "tensor(uint64)";
    default:return "unsupported";
    }
}
std::string quoted(const char *s) {
    std::string out="\"";
    constexpr char hex[]="0123456789abcdef";
    for (const unsigned char *p=reinterpret_cast<const unsigned char *>(s); *p; ++p) {
        if (*p=='"' || *p=='\\') { out+='\\'; out+=static_cast<char>(*p); }
        else if (*p<32) { out+="\\u00"; out+=hex[*p>>4]; out+=hex[*p&15]; }
        else out+=static_cast<char>(*p);
    }
    return out+'"';
}
std::string metadata_list(const Session &s, bool output) {
    auto api=s.runtime->api;
    size_t count=0;
    ort_check(api,output?api->SessionGetOutputCount(s.value,&count):api->SessionGetInputCount(s.value,&count));
    OrtAllocator *allocator=nullptr;
    ort_check(api,api->GetAllocatorWithDefaultOptions(&allocator));
    std::string json="[";
    for (size_t i=0;i<count;++i) {
        if (i) json+=',';
        char *name=nullptr;
        ort_check(api,output?api->SessionGetOutputName(s.value,i,allocator,&name):api->SessionGetInputName(s.value,i,allocator,&name));
        try { json+="{\"name\":"+quoted(name); }
        catch (...) { allocator->Free(allocator,name); throw; }
        allocator->Free(allocator,name);
        Owned<OrtTypeInfo> type(api->ReleaseTypeInfo);
        ort_check(api,output?api->SessionGetOutputTypeInfo(s.value,i,&type.value):api->SessionGetInputTypeInfo(s.value,i,&type.value));
        const OrtTensorTypeAndShapeInfo *tensor=nullptr;
        ort_check(api,api->CastTypeInfoToTensorInfo(type.value,&tensor));
        if (!tensor) throw Failure(ORT_NOT_IMPLEMENTED,"Native ONNX host requires dense tensor graph inputs and outputs");
        ONNXTensorElementDataType dtype; size_t rank=0;
        ort_check(api,api->GetTensorElementType(tensor,&dtype));
        (void)element_bytes(dtype);
        ort_check(api,api->GetDimensionsCount(tensor,&rank));
        std::vector<int64_t> dims(rank);
        std::vector<const char *> symbols(rank);
        ort_check(api,api->GetDimensions(tensor,dims.data(),rank));
        ort_check(api,api->GetSymbolicDimensions(tensor,symbols.data(),rank));
        json+=",\"type\":"+quoted(type_name(dtype))+",\"shape\":[";
        for (size_t d=0;d<rank;++d) {
            if (d) json+=',';
            if (dims[d]>=0) json+=std::to_string(dims[d]);
            else if (symbols[d] && *symbols[d]) json+=quoted(symbols[d]);
            else json+="null";
        }
        json+="]}";
    }
    return json+"]";
}
struct TensorInfo {
    int type;
    std::vector<int64_t> shape;
    size_t bytes;
};
TensorInfo tensor_info(const Result &r,size_t index) {
    ensure(index<r.values.size(),"Native ONNX output index is out of bounds");
    auto api=r.session->runtime->api;
    Owned<OrtTensorTypeAndShapeInfo> info(api->ReleaseTensorTypeAndShapeInfo);
    ort_check(api,api->GetTensorTypeAndShape(r.values[index],&info.value));
    ONNXTensorElementDataType type; size_t rank=0,count=0;
    ort_check(api,api->GetTensorElementType(info.value,&type));
    ort_check(api,api->GetDimensionsCount(info.value,&rank));
    ort_check(api,api->GetTensorShapeElementCount(info.value,&count));
    size_t width=element_bytes(type);
    ensure(count<=SIZE_MAX/width,"Native ONNX output byte count overflows");
    TensorInfo result{type,std::vector<int64_t>(rank),count*width};
    ort_check(api,api->GetDimensions(info.value,result.shape.data(),rank));
    return result;
}
}

extern "C" CN_EXPORT int cn_ort_create(const wchar_t *path,const void *model,size_t bytes,
    uint64_t *id,char *error,size_t capacity) {
    return guarded(error,capacity,[&] {
        ensure(id && model && bytes && span(model,bytes),"Native ONNX model buffer and result are required");
        auto s=std::make_shared<Session>(); s->runtime=runtime(path);
        auto api=s->runtime->api;
        Owned<OrtSessionOptions> options(api->ReleaseSessionOptions);
        ort_check(api,api->CreateSessionOptions(&options.value));
        ort_check(api,api->SetSessionGraphOptimizationLevel(options.value,ORT_ENABLE_ALL));
        ort_check(api,api->CreateSessionFromArray(s->runtime->env,model,bytes,options.value,&s->value));
        s->metadata="{\"inputs\":"+metadata_list(*s,false)+",\"outputs\":"+metadata_list(*s,true)+
            ",\"runtime_version\":"+quoted(s->runtime->version.c_str())+"}";
        auto &r=registry(); std::lock_guard lock(r.mutex);
        uint64_t key=next_handle(r); r.sessions.emplace(key,s); *id=key;
    });
}
extern "C" CN_EXPORT int cn_ort_metadata(uint64_t id,char *json,size_t capacity,size_t *required,
    char *error,size_t error_capacity) {
    return guarded(error,error_capacity,[&] {
        ensure(required && (json || !capacity),"Invalid native ONNX metadata buffer");
        auto s=session_for(id); *required=s->metadata.size()+1;
        if (!json && !capacity) return;
        ensure(capacity>=*required,"Native ONNX metadata buffer is too small");
        std::memcpy(json,s->metadata.c_str(),*required);
    });
}
extern "C" CN_EXPORT int cn_ort_run(uint64_t id,const cn_ort_input *inputs,size_t input_count,
    const char *const *outputs,size_t output_count,uint64_t *result,char *error,size_t capacity) {
    return guarded(error,capacity,[&] {
        ensure(result && (inputs || !input_count) && outputs && output_count,"Invalid native ONNX run buffers");
        ensure(input_count<=static_cast<size_t>(PTRDIFF_MAX)/sizeof(cn_ort_input) &&
            output_count<=static_cast<size_t>(PTRDIFF_MAX)/sizeof(void *),"Native ONNX binding count overflows");
        ensure(span(inputs,input_count*sizeof(cn_ort_input),alignof(cn_ort_input)) &&
            span(outputs,output_count*sizeof(void *),alignof(void *)),"Native ONNX binding span is invalid");
        auto batch=std::make_shared<Result>(); batch->session=session_for(id);
        auto rt=batch->session->runtime; auto api=rt->api;
        Owned<OrtMemoryInfo> memory(api->ReleaseMemoryInfo);
        ort_check(api,api->CreateCpuMemoryInfo(OrtArenaAllocator,OrtMemTypeDefault,&memory.value));
        std::vector<const char *> names; std::vector<const OrtValue *> values;
        names.reserve(input_count); values.reserve(input_count); batch->inputs.reserve(input_count);
        for (size_t i=0;i<input_count;++i) {
            const auto &src=inputs[i];
            ensure(src.name && (src.shape || !src.rank) && src.rank<=static_cast<size_t>(PTRDIFF_MAX)/sizeof(int64_t),"Invalid native ONNX input descriptor");
            ensure(span(src.shape,src.rank*sizeof(int64_t),alignof(int64_t)),"Native ONNX shape span is invalid");
            size_t count=1,width=element_bytes(src.type);
            for (size_t d=0;d<src.rank;++d) {
                ensure(src.shape[d]>=0,"Native ONNX input dimensions must be nonnegative");
                size_t dim=static_cast<size_t>(src.shape[d]);
                ensure(!dim || count<=SIZE_MAX/dim,"Native ONNX input shape overflows"); count*=dim;
            }
            ensure(count<=static_cast<size_t>(PTRDIFF_MAX)/width && src.bytes==count*width && span(src.data,src.bytes),"Invalid native ONNX input byte count");
            auto v=std::make_unique<InputValue>(); v->runtime=rt;
            v->bytes.resize(std::max<size_t>(src.bytes,1));
            if (src.bytes) std::memcpy(v->bytes.data(),src.data,src.bytes);
            ort_check(api,api->CreateTensorWithDataAsOrtValue(memory.value,v->bytes.data(),src.bytes,
                src.shape,src.rank,static_cast<ONNXTensorElementDataType>(src.type),&v->value));
            names.push_back(src.name); values.push_back(v->value); batch->inputs.push_back(std::move(v));
        }
        for (size_t i=0;i<output_count;++i) ensure(outputs[i]!=nullptr,"Native ONNX output name is null");
        batch->values.resize(output_count,nullptr);
        ort_check(api,api->Run(batch->session->value,nullptr,names.data(),values.data(),input_count,
            outputs,output_count,batch->values.data()));
        auto &r=registry(); std::lock_guard lock(r.mutex);
        uint64_t key=next_handle(r); r.results.emplace(key,batch); *result=key;
    });
}
extern "C" CN_EXPORT int cn_ort_result_info(uint64_t id,size_t index,int *type,size_t *rank,
    int64_t *shape,size_t shape_capacity,size_t *bytes,char *error,size_t capacity) {
    return guarded(error,capacity,[&] {
        ensure(type && rank && bytes && (shape || !shape_capacity),"Invalid native ONNX output descriptor");
        auto r=result_for(id); auto info=tensor_info(*r,index);
        *type=info.type; *rank=info.shape.size(); *bytes=info.bytes;
        if (!shape && !shape_capacity) return;
        ensure(shape_capacity>=info.shape.size(),"Native ONNX output shape buffer is too small");
        if (!info.shape.empty()) std::memcpy(shape,info.shape.data(),info.shape.size()*sizeof(int64_t));
    });
}
extern "C" CN_EXPORT int cn_ort_result_copy(uint64_t id,size_t index,void *data,size_t bytes,
    char *error,size_t capacity) {
    return guarded(error,capacity,[&] {
        auto r=result_for(id); auto info=tensor_info(*r,index);
        ensure(bytes==info.bytes && span(data,bytes),"Invalid native ONNX output byte buffer");
        void *source=nullptr; auto api=r->session->runtime->api;
        ort_check(api,api->GetTensorMutableData(r->values[index],&source));
        if (bytes) std::memcpy(data,source,bytes);
    });
}
extern "C" CN_EXPORT int cn_ort_release_session(uint64_t id,char *error,size_t capacity) {
    return guarded(error,capacity,[&] {
        std::shared_ptr<Session> removed;
        { auto &r=registry(); std::lock_guard lock(r.mutex); removed=lookup(id,r.sessions); r.sessions.erase(id); }
    });
}
extern "C" CN_EXPORT int cn_ort_release_result(uint64_t id,char *error,size_t capacity) {
    return guarded(error,capacity,[&] {
        std::shared_ptr<Result> removed;
        { auto &r=registry(); std::lock_guard lock(r.mutex); removed=lookup(id,r.results); r.results.erase(id); }
    });
}
extern "C" CN_EXPORT int cn_ort_live_handles(size_t *sessions,size_t *results) {
    return guarded(nullptr,0,[&] {
        auto &r=registry(); std::lock_guard lock(r.mutex);
        if (sessions) {*sessions=r.sessions.size();} if (results) *results=r.results.size();
    });
}
