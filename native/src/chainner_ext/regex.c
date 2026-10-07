/* Port of chainner_ext 0.3.10 crates/bindings/src/regex.rs, MIT OR Apache-2.0. */
/* RustRegex, RegexMatch and MatchGroup over chainner_regex.h (regex-py's Regex and
 * RegexMatch: char positions through PosTranslator), and the _C_API capsule that
 * _chainner_graph.pyd reaches the engine through. */
#include "ext.h"
#include "chainner_regex.h"
#include <stdlib.h>
#include <string.h>

PyTypeObject *chn_regex_type, *chn_match_type, *chn_group_type;

typedef struct regex_object {
    PyObject_HEAD
    chn_regex *re;
    PyObject *names; /* tuple: capture index -> name (str) or None */
} regex_object;

typedef struct match_object {
    PyObject_HEAD
    PyObject *names; /* the regex's names */
    size_t count;
    size_t *spans;   /* count x (start, end) char positions; CHN_REGEX_NO_SLOT: None */
} match_object;

typedef struct group_object {
    PyObject_HEAD
    size_t start, end;
} group_object;

static PyObject *raise_status(chn_regex_status status, chn_regex_string *message)
{
    if (status == CHN_REGEX_PANIC) {
        PyObject *text = PyUnicode_DecodeUTF8(message->ptr ? message->ptr : "", (Py_ssize_t)message->len, "replace");
        if (text) { PyErr_SetObject(chn_panic_type, text); Py_DECREF(text); }
    } else if (status == CHN_REGEX_ERROR_NOMEM) {
        PyErr_NoMemory();
    } else if (status == CHN_REGEX_UNIMPLEMENTED) {
        PyErr_SetString(PyExc_NotImplementedError, "the C regex engine (chainner_regex) is not built yet");
    } else {
        PyErr_Format(PyExc_RuntimeError, "unexpected regex engine status %d", (int)status);
    }
    chn_regex_string_drop(message);
    return NULL;
}

/* ---------------------------------------------------------------------------------- */
/* MatchGroup */

static PyObject *group_new(size_t start, size_t end)
{
    group_object *group = PyObject_New(group_object, chn_group_type);
    if (group) { group->start = start; group->end = end; }
    return (PyObject *)group;
}

static PyObject *group_start(PyObject *self, void *closure)
{
    (void)closure;
    return PyLong_FromSize_t(((group_object *)self)->start);
}

static PyObject *group_end(PyObject *self, void *closure)
{
    (void)closure;
    return PyLong_FromSize_t(((group_object *)self)->end);
}

static PyObject *group_len(PyObject *self, void *closure)
{
    (void)closure;
    const group_object *group = (const group_object *)self;
    return PyLong_FromSize_t(group->end - group->start);
}

static PyGetSetDef group_getset[] = {
    {"start", group_start, NULL, "", NULL},
    {"end", group_end, NULL, "", NULL},
    {"len", group_len, NULL, "", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

/* ---------------------------------------------------------------------------------- */
/* RegexMatch */

/* RegexMatch::from_captures: every group's start, then end, through one translator. */
static PyObject *match_new(const regex_object *regex,
    const size_t *slots, size_t count, chn_regex_pos_translator *translator)
{
    match_object *match = PyObject_New(match_object, chn_match_type);
    if (!match) return NULL;
    match->names = Py_NewRef(regex->names);
    match->count = count;
    match->spans = malloc((count ? count : 1) * 2 * sizeof(size_t));
    if (!match->spans) { Py_DECREF(match); return PyErr_NoMemory(); }
    for (size_t i = 0; i < 2 * count; i += 2) {
        if (slots[i] == CHN_REGEX_NO_SLOT || slots[i + 1] == CHN_REGEX_NO_SLOT) {
            match->spans[i] = match->spans[i + 1] = CHN_REGEX_NO_SLOT;
            continue;
        }
        for (size_t k = 0; k < 2; ++k) {
            chn_regex_string panic = {NULL, 0};
            chn_regex_status status = chn_regex_pos_translator_get_char_pos(translator,
                slots[i + k], &match->spans[i + k], &panic);
            if (status != CHN_REGEX_OK) {
                Py_DECREF(match);
                return raise_status(status, &panic);
            }
        }
    }
    return (PyObject *)match;
}

static void match_dealloc(PyObject *self)
{
    match_object *match = (match_object *)self;
    Py_XDECREF(match->names);
    free(match->spans);
    PyTypeObject *type = Py_TYPE(self);
    ((freefunc)PyType_GetSlot(type, Py_tp_free))(self);
    Py_DECREF(type);
}

static PyObject *match_start(PyObject *self, void *closure)
{
    (void)closure;
    return PyLong_FromSize_t(((match_object *)self)->spans[0]);
}

static PyObject *match_end(PyObject *self, void *closure)
{
    (void)closure;
    return PyLong_FromSize_t(((match_object *)self)->spans[1]);
}

static PyObject *match_len(PyObject *self, void *closure)
{
    (void)closure;
    const match_object *match = (const match_object *)self;
    return PyLong_FromSize_t(match->spans[1] - match->spans[0]);
}

static PyObject *match_group(const match_object *match, uint64_t index)
{
    if (index >= match->count || match->spans[2 * index] == CHN_REGEX_NO_SLOT) Py_RETURN_NONE;
    return group_new(match->spans[2 * index], match->spans[2 * index + 1]);
}

static PyObject *match_get(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *const names[] = {"index"};
    PyObject *values[1];
    uint64_t index;
    if (chn_parse(args, kwargs, "RegexMatch.get", names, 1, 1, values) ||
        chn_u64(values[0], "index", &index)) return NULL;
    return match_group((const match_object *)self, index);
}

static PyObject *match_get_by_name(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *const names[] = {"name"};
    PyObject *values[1];
    Py_ssize_t length;
    if (chn_parse(args, kwargs, "RegexMatch.get_by_name", names, 1, 1, values) ||
        !chn_utf8(values[0], "name", &length)) return NULL;
    const match_object *match = (const match_object *)self;
    Py_ssize_t count = PyTuple_Size(match->names);
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject *name = PyTuple_GetItem(match->names, i);
        if (name == Py_None) continue;
        int equal = PyObject_RichCompareBool(name, values[0], Py_EQ);
        if (equal < 0) return NULL;
        if (equal) return match_group(match, (uint64_t)i);
    }
    Py_RETURN_NONE;
}

static PyGetSetDef match_getset[] = {
    {"start", match_start, NULL, "", NULL},
    {"end", match_end, NULL, "", NULL},
    {"len", match_len, NULL, "", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyMethodDef match_methods[] = {
    {"get", (PyCFunction)(void (*)(void))match_get, METH_VARARGS | METH_KEYWORDS, "get($self, index)\n--\n\n"},
    {"get_by_name", (PyCFunction)(void (*)(void))match_get_by_name, METH_VARARGS | METH_KEYWORDS,
     "get_by_name($self, name)\n--\n\n"},
    {NULL, NULL, 0, NULL},
};

/* ---------------------------------------------------------------------------------- */
/* RustRegex */

static PyObject *regex_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    static const char *const names[] = {"patter"};
    PyObject *values[1];
    Py_ssize_t length;
    if (chn_parse(args, kwargs, "RustRegex.__new__", names, 1, 1, values)) return NULL;
    const char *pattern = chn_utf8(values[0], "patter", &length);
    if (!pattern) return NULL;
    chn_regex *re = NULL;
    chn_regex_string error = {NULL, 0};
    chn_regex_status status = chn_regex_compile(pattern, (size_t)length, NULL, &re, &error);
    if (status == CHN_REGEX_ERROR_INVALID) {
        PyObject *text = PyUnicode_DecodeUTF8(error.ptr ? error.ptr : "", (Py_ssize_t)error.len, "replace");
        if (text) { PyErr_SetObject(PyExc_ValueError, text); Py_DECREF(text); }
        chn_regex_string_drop(&error);
        return NULL;
    }
    if (status != CHN_REGEX_OK) return raise_status(status, &error);
    size_t captures = chn_regex_captures_len(re);
    PyObject *tuple = PyTuple_New((Py_ssize_t)captures);
    for (size_t i = 0; tuple && i < captures; ++i) {
        size_t name_length;
        const char *name = chn_regex_capture_name(re, i, &name_length);
        PyObject *item = name ? PyUnicode_DecodeUTF8(name, (Py_ssize_t)name_length, "strict") : Py_NewRef(Py_None);
        if (!item || PyTuple_SetItem(tuple, (Py_ssize_t)i, item) < 0) Py_CLEAR(tuple);
    }
    regex_object *self = tuple ? PyObject_New(regex_object, type) : NULL;
    if (!self) { Py_XDECREF(tuple); chn_regex_free(re); return NULL; }
    self->re = re;
    self->names = tuple;
    return (PyObject *)self;
}

static void regex_dealloc(PyObject *self)
{
    regex_object *regex = (regex_object *)self;
    chn_regex_free(regex->re);
    Py_XDECREF(regex->names);
    PyTypeObject *type = Py_TYPE(self);
    ((freefunc)PyType_GetSlot(type, Py_tp_free))(self);
    Py_DECREF(type);
}

static PyObject *regex_pattern(PyObject *self, void *closure)
{
    (void)closure;
    size_t length;
    const char *pattern = chn_regex_pattern(((regex_object *)self)->re, &length);
    return PyUnicode_DecodeUTF8(pattern, (Py_ssize_t)length, "strict");
}

static PyObject *regex_groups(PyObject *self, void *closure)
{
    (void)closure;
    return PyLong_FromSize_t(chn_regex_groups(((regex_object *)self)->re));
}

static PyObject *regex_groupindex(PyObject *self, void *closure)
{
    (void)closure;
    const regex_object *regex = (const regex_object *)self;
    PyObject *dict = PyDict_New();
    Py_ssize_t count = PyTuple_Size(regex->names);
    for (Py_ssize_t i = 0; dict && i < count; ++i) {
        PyObject *name = PyTuple_GetItem(regex->names, i);
        if (name == Py_None) continue;
        PyObject *index = PyLong_FromSsize_t(i);
        if (!index || PyDict_SetItem(dict, name, index) < 0) Py_CLEAR(dict);
        Py_XDECREF(index);
    }
    return dict;
}

static PyObject *regex_search(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *const names[] = {"text", "pos"};
    PyObject *values[2];
    Py_ssize_t length;
    uint64_t pos = 0;
    if (chn_parse(args, kwargs, "RustRegex.search", names, 2, 1, values)) return NULL;
    const char *text = chn_utf8(values[0], "text", &length);
    if (!text || (values[1] && values[1] != Py_None && chn_u64(values[1], "pos", &pos))) return NULL;
    const regex_object *regex = (const regex_object *)self;
    size_t count = chn_regex_captures_len(regex->re);
    size_t *slots = malloc(2 * count * sizeof(size_t));
    if (!slots) return PyErr_NoMemory();
    size_t byte_pos = chn_regex_to_byte_pos(text, (size_t)length, (size_t)pos);
    chn_regex_string panic = {NULL, 0};
    chn_regex_status status = chn_regex_search_at(regex->re, text, (size_t)length, byte_pos,
        slots, 2 * count, &panic);
    PyObject *result = NULL;
    if (status == CHN_REGEX_NO_MATCH) {
        result = Py_NewRef(Py_None);
    } else if (status != CHN_REGEX_OK) {
        raise_status(status, &panic);
    } else {
        chn_regex_pos_translator translator;
        chn_regex_pos_translator_init(&translator, text, (size_t)length);
        result = match_new(regex, slots, count, &translator);
        chn_regex_pos_translator_drop(&translator);
    }
    free(slots);
    return result;
}

static PyObject *regex_findall(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *const names[] = {"text"};
    PyObject *values[1];
    Py_ssize_t length;
    if (chn_parse(args, kwargs, "RustRegex.findall", names, 1, 1, values)) return NULL;
    const char *text = chn_utf8(values[0], "text", &length);
    if (!text) return NULL;
    const regex_object *regex = (const regex_object *)self;
    size_t count = chn_regex_captures_len(regex->re);
    size_t *slots = malloc(2 * count * sizeof(size_t));
    PyObject *list = PyList_New(0);
    if (!slots || !list) { free(slots); Py_XDECREF(list); return PyErr_NoMemory(); }
    chn_regex_iter iter;
    chn_regex_iter_init(&iter, regex->re, text, (size_t)length);
    chn_regex_pos_translator translator;
    chn_regex_pos_translator_init(&translator, text, (size_t)length);
    for (;;) {
        chn_regex_status status = chn_regex_iter_next_captures(&iter, slots, 2 * count);
        if (status == CHN_REGEX_NO_MATCH) break;
        PyObject *match = NULL;
        if (status != CHN_REGEX_OK) {
            chn_regex_string none = {NULL, 0};
            raise_status(status, &none);
        } else {
            match = match_new(regex, slots, count, &translator);
        }
        if (!match || PyList_Append(list, match) < 0) {
            Py_XDECREF(match);
            Py_CLEAR(list);
            break;
        }
        Py_DECREF(match);
    }
    chn_regex_pos_translator_drop(&translator);
    free(slots);
    return list;
}

static PyObject *split_common(PyObject *self, PyObject *args, PyObject *kwargs, int captures)
{
    static const char *const names[] = {"text"};
    PyObject *values[1];
    Py_ssize_t length;
    if (chn_parse(args, kwargs, captures ? "RustRegex.split" : "RustRegex.split_without_captures",
            names, 1, 1, values)) return NULL;
    const char *text = chn_utf8(values[0], "text", &length);
    if (!text) return NULL;
    const regex_object *regex = (const regex_object *)self;
    chn_regex_spans spans = {NULL, 0, 0};
    chn_regex_status status = captures ?
        chn_regex_split(regex->re, text, (size_t)length, &spans) :
        chn_regex_split_without_captures(regex->re, text, (size_t)length, &spans);
    if (status != CHN_REGEX_OK) {
        chn_regex_string none = {NULL, 0};
        chn_regex_spans_drop(&spans);
        return raise_status(status, &none);
    }
    PyObject *list = PyList_New((Py_ssize_t)spans.len);
    for (size_t i = 0; list && i < spans.len; ++i) {
        PyObject *piece = PyUnicode_DecodeUTF8(text + spans.ptr[i].start,
            (Py_ssize_t)(spans.ptr[i].end - spans.ptr[i].start), "strict");
        if (!piece || PyList_SetItem(list, (Py_ssize_t)i, piece) < 0) Py_CLEAR(list);
    }
    chn_regex_spans_drop(&spans);
    return list;
}

static PyObject *regex_split(PyObject *self, PyObject *args, PyObject *kwargs)
{
    return split_common(self, args, kwargs, 1);
}

static PyObject *regex_split_without_captures(PyObject *self, PyObject *args, PyObject *kwargs)
{
    return split_common(self, args, kwargs, 0);
}

static PyGetSetDef regex_getset[] = {
    {"pattern", regex_pattern, NULL, "", NULL},
    {"groups", regex_groups, NULL, "", NULL},
    {"groupindex", regex_groupindex, NULL, "", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyMethodDef regex_methods[] = {
    {"search", (PyCFunction)(void (*)(void))regex_search, METH_VARARGS | METH_KEYWORDS,
     "search($self, text, pos=None)\n--\n\n"},
    {"findall", (PyCFunction)(void (*)(void))regex_findall, METH_VARARGS | METH_KEYWORDS,
     "findall($self, text)\n--\n\n"},
    {"split", (PyCFunction)(void (*)(void))regex_split, METH_VARARGS | METH_KEYWORDS,
     "split($self, text)\n--\n\n"},
    {"split_without_captures", (PyCFunction)(void (*)(void))regex_split_without_captures,
     METH_VARARGS | METH_KEYWORDS, "split_without_captures($self, text)\n--\n\n"},
    {NULL, NULL, 0, NULL},
};

/* ---------------------------------------------------------------------------------- */

/* chn_regex_api::regex_of: an exact RustRegex's handle, else NULL (subclasses excluded). */
static const chn_regex *regex_of(void *object)
{
    PyObject *regex = (PyObject *)object;
    return Py_TYPE(regex) == chn_regex_type ? ((const regex_object *)regex)->re : NULL;
}

static const chn_regex_api regex_api = {
    CHN_REGEX_API_VERSION,
    chn_regex_string_drop,
    chn_regex_compile,
    chn_regex_free,
    chn_regex_pattern,
    chn_regex_groups,
    chn_regex_captures_len,
    chn_regex_capture_name,
    chn_regex_search_at,
    chn_regex_iter_init,
    chn_regex_iter_next,
    chn_regex_iter_next_captures,
    chn_regex_spans_drop,
    chn_regex_split,
    chn_regex_split_without_captures,
    chn_regex_to_byte_pos,
    chn_regex_pos_translator_init,
    chn_regex_pos_translator_get_char_pos,
    chn_regex_pos_translator_drop,
    regex_of,
};

static PyTypeObject *new_type(const char *name, int size, PyType_Slot *slots)
{
    PyType_Spec spec = {name, size, 0, Py_TPFLAGS_DEFAULT, slots};
    return (PyTypeObject *)PyType_FromSpec(&spec);
}

int chn_regex_init(PyObject *module)
{
    PyType_Slot regex_slots[] = {
        {Py_tp_new, (void *)regex_new},
        {Py_tp_dealloc, (void *)regex_dealloc},
        {Py_tp_getset, regex_getset},
        {Py_tp_methods, regex_methods},
        {Py_tp_doc, "RustRegex(patter)\n--\n\n"},
        {0, NULL},
    };
    PyType_Slot match_slots[] = {
        {Py_tp_new, (void *)chn_no_constructor},
        {Py_tp_dealloc, (void *)match_dealloc},
        {Py_tp_getset, match_getset},
        {Py_tp_methods, match_methods},
        {0, NULL},
    };
    PyType_Slot group_slots[] = {
        {Py_tp_new, (void *)chn_no_constructor},
        {Py_tp_getset, group_getset},
        {0, NULL},
    };
    chn_regex_type = new_type(CHN_TYPE_NAME("RustRegex"), (int)sizeof(regex_object), regex_slots);
    chn_match_type = new_type(CHN_TYPE_NAME("RegexMatch"), (int)sizeof(match_object), match_slots);
    chn_group_type = new_type(CHN_TYPE_NAME("MatchGroup"), (int)sizeof(group_object), group_slots);
    if (!chn_regex_type || !chn_match_type || !chn_group_type) return -1;
    PyObject *capsule = PyCapsule_New((void *)&regex_api, CHN_REGEX_CAPSULE_NAME, NULL);
    if (!capsule) return -1;
    Py_INCREF((PyObject *)chn_regex_type);
    Py_INCREF((PyObject *)chn_group_type);
    Py_INCREF((PyObject *)chn_match_type);
    if (PyModule_AddObject(module, "RustRegex", (PyObject *)chn_regex_type) < 0 ||
        PyModule_AddObject(module, "MatchGroup", (PyObject *)chn_group_type) < 0 ||
        PyModule_AddObject(module, "RegexMatch", (PyObject *)chn_match_type) < 0 ||
        PyModule_AddObject(module, "_C_API", capsule) < 0)
        return -1;
    return 0;
}
