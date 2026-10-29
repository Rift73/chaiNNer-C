// Get Native Resolution's analysis session over one ISA build of the vendored Vernier 0.2 core
// (third_party/vernier/PROVENANCE.md): this file is compiled once per level with the core, under the
// namespace the target's command line names, and exports that level's GetnativeUnit table. The
// JSON is Vernier's report(), parameters(), summarize() and batchSummary() (report.cpp, batch.cpp)
// rebuilt without Qt: the same field names, values and meanings, keys sorted as QJsonObject sorts
// them. Additions: each analysis's kernel evidence, and in the summary a short record per analysis
// (its verdict, statuses and leading candidates), the kernel decision and the selection (the frame
// and candidate the node's outputs carry).
#include "getnative_unit.hpp"
#include "getnative_kernel.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace native {
namespace {
// A JSON value as QJsonValue holds one; non-finite numbers are null, as Vernier's number() makes them.
class Json {
public:
    Json() = default;
    Json(std::nullptr_t) {}
    Json(bool value) : type_(Type::Bool), number_(value ? 1 : 0) {}
    Json(int value) : type_(Type::Number), number_(value) {}
    Json(double value) : type_(std::isfinite(value) ? Type::Number : Type::Null), number_(value) {}
    Json(const char *value) : type_(Type::String), text_(value) {}
    Json(std::string value) : type_(Type::String), text_(std::move(value)) {}
    static Json array(std::initializer_list<Json> items = {}) {
        Json json;
        json.type_ = Type::Array;
        json.items_ = items;
        return json;
    }
    static Json object(std::initializer_list<std::pair<std::string, Json>> members = {}) {
        Json json;
        json.type_ = Type::Object;
        for (const auto &member : members) json[member.first] = member.second;
        return json;
    }
    // Text that is already JSON, written as it is.
    static Json raw(std::string text) {
        Json json;
        json.type_ = Type::Raw;
        json.text_ = std::move(text);
        return json;
    }
    Json &operator[](const std::string &key) {
        for (auto &member : members_)
            if (member.first == key) return member.second;
        members_.emplace_back(key, Json());
        return members_.back().second;
    }
    const Json *find(const std::string &key) const {
        for (const auto &member : members_)
            if (member.first == key) return &member.second;
        return nullptr;
    }
    void remove(const std::string &key) {
        members_.erase(std::remove_if(members_.begin(), members_.end(), [&](const auto &m) { return m.first == key; }),
                       members_.end());
    }
    void append(Json value) { items_.push_back(std::move(value)); }
    std::vector<Json> &items() { return items_; }
    const std::vector<Json> &items() const { return items_; }
    double number() const { return number_; }
    const std::string &text() const { return text_; }
    // QJsonDocument::Indented's layout.
    std::string dump() const {
        std::string out;
        write(out, 0);
        out += '\n';
        return out;
    }

private:
    enum class Type { Null, Bool, Number, String, Raw, Array, Object };
    Type type_ = Type::Null;
    double number_ = 0;
    std::string text_;
    std::vector<Json> items_;
    std::vector<std::pair<std::string, Json>> members_;

    // A lone surrogate arrives in UTF-8's three-byte form for its code point (ED A0..BF 80..BF, the Python
    // bridge's surrogatepass), which is not valid UTF-8: it is written as a \u escape, which JSON allows.
    static void quote(std::string &out, const std::string &text) {
        out += '"';
        for (size_t i = 0; i < text.size(); ++i) {
            const unsigned char c = static_cast<unsigned char>(text[i]);
            if (c == 0xED && i + 2 < text.size() && (static_cast<unsigned char>(text[i + 1]) & 0xE0) == 0xA0 &&
                (static_cast<unsigned char>(text[i + 2]) & 0xC0) == 0x80) {
                char escape[8];
                std::snprintf(escape, sizeof escape, "\\u%04x",
                              0xD000u | ((static_cast<unsigned char>(text[i + 1]) & 0x3Fu) << 6) |
                                  (static_cast<unsigned char>(text[i + 2]) & 0x3Fu));
                out += escape;
                i += 2;
            } else if (c == '"' || c == '\\') {
                out += '\\';
                out += char(c);
            } else if (c == '\n') {
                out += "\\n";
            } else if (c == '\t') {
                out += "\\t";
            } else if (c == '\r') {
                out += "\\r";
            } else if (c < 0x20) {
                char escape[8];
                std::snprintf(escape, sizeof escape, "\\u%04x", c);
                out += escape;
            } else {
                out += char(c);
            }
        }
        out += '"';
    }
    void write(std::string &out, int depth) const {
        const std::string indent(size_t(depth + 1) * 4, ' '), close(size_t(depth) * 4, ' ');
        switch (type_) {
        case Type::Null: out += "null"; return;
        case Type::Bool: out += number_ != 0 ? "true" : "false"; return;
        case Type::Number: {
            char buffer[32];
            out.append(buffer, std::to_chars(buffer, buffer + sizeof buffer, number_).ptr);
            return;
        }
        case Type::String: quote(out, text_); return;
        case Type::Raw: out += text_; return;
        case Type::Array:
            if (items_.empty()) {
                out += "[]";
                return;
            }
            out += "[\n";
            for (size_t i = 0; i < items_.size(); ++i) {
                out += indent;
                items_[i].write(out, depth + 1);
                out += i + 1 < items_.size() ? ",\n" : "\n";
            }
            out += close + "]";
            return;
        case Type::Object: {
            if (members_.empty()) {
                out += "{}";
                return;
            }
            std::vector<const std::pair<std::string, Json> *> sorted;
            for (const auto &member : members_) sorted.push_back(&member);
            std::sort(sorted.begin(), sorted.end(), [](auto a, auto b) { return a->first < b->first; });
            out += "{\n";
            for (size_t i = 0; i < sorted.size(); ++i) {
                out += indent;
                quote(out, sorted[i]->first);
                out += ": ";
                sorted[i]->second.write(out, depth + 1);
                out += i + 1 < sorted.size() ? ",\n" : "\n";
            }
            out += close + "}";
            return;
        }
        }
    }
};

// Vernier's Frame without its samples.
struct Frame {
    std::string path, sample, provenance, lightDomain, colorAssumptions;
    double requestedSeconds = 0;
    std::array<double, 3> linearWeights{.2126, .7152, .0722};
    bool keyframeOnly = false;
};
struct Record {
    Result result;
    Frame frame;
    std::shared_ptr<KernelEvidence> evidence;
};
struct Session {
    Options options;
    std::vector<Record> records;
};
using View = std::pair<const Result *, const Frame *>;
using Grid = std::pair<int, int>;

const char *familyName(Family f) {
    switch (f) {
    case Family::Bilinear: return "bilinear";
    case Family::Bicubic: return "bicubic";
    case Family::Lanczos: return "lanczos";
    case Family::Spline16: return "spline16";
    case Family::Spline36: return "spline36";
    case Family::Spline64: return "spline64";
    }
    throw std::invalid_argument("Unknown kernel family");
}
Json kernelParams(const Kernel &k) {
    if (k.family == Family::Bicubic) return Json::object({{"b", k.b}, {"c", k.c}});
    if (k.family == Family::Lanczos) return Json::object({{"taps", k.taps}});
    return Json::object();
}
Json gridJson(Grid grid) { return Json::array({grid.first, grid.second}); }

Json parameters(const Result &r, size_t index) {
    if (index >= r.candidates.size()) return Json::object();
    const auto &c = r.candidates[index];
    const bool sameGrid = c.x.size == r.candidates.front().x.size && c.y.size == r.candidates.front().y.size;
    const auto op = exportGeometry(c);
    auto kernel = [](const Candidate &candidate) {
        const auto &k = candidate.x.kernel;
        return Json::object({{"family", familyName(k.family)},
                             {"params", kernelParams(k)},
                             {"continuous_approximation", candidate.continuous}});
    };
    Json alternatives = Json::array();
    for (size_t i = 0; i < r.candidates.size(); ++i) {
        const auto &a = r.candidates[i];
        if (a.score <= r.candidates.front().score / .92) {
            auto k = kernel(a);
            k["rank"] = int(i + 1);
            k["relative_score"] = a.score / std::max(r.candidates.front().score, 1e-30);
            alternatives.append(k);
        }
    }
    auto selectedKernel = kernel(c);
    selectedKernel["axes"] = "Same kernel assumed on both axes";
    selectedKernel["alternatives"] = alternatives;
    std::string toLinear = "c", toEncoded = "l";
    const auto &domain = r.lightDomain;
    if (domain == "linear-bt709") {
        toLinear = "sign(c) * (abs(c) < 0.081 ? abs(c)/4.5 : pow((abs(c)+0.099)/1.099, 1/0.45))";
        toEncoded = "sign(l) * (abs(l) < 0.018 ? 4.5*abs(l) : 1.099*pow(abs(l),0.45)-0.099)";
    } else if (domain == "linear-srgb") {
        toLinear = "sign(c) * (abs(c) <= 0.04045 ? abs(c)/12.92 : pow((abs(c)+0.055)/1.055,2.4))";
        toEncoded = "sign(l) * (abs(l) <= 0.0031308 ? 12.92*abs(l) : 1.055*pow(abs(l),1/2.4)-0.055)";
    } else if (domain != "encoded") {
        const std::string gamma = domain == "linear-gamma22" ? "2.2" : "2.4";
        toLinear = "sign(c)*pow(abs(c)," + gamma + ")";
        toEncoded = "sign(l)*pow(abs(l),1/" + gamma + ")";
    }
    const bool encoded = r.lightDomain == "encoded";
    return Json::object({
        {"schema", "vernier.parameters.v3"},
        {"selection", index == r.recommendedCandidate ? "automatic_recommendation" : "manual_override"},
        {"light_domain", r.lightDomain},
        {"light_status", "not_identified"},
        {"kernel_status_scope", "conditional_on_light_domain"},
        {"color_assumptions", r.colorAssumptions},
        {"required_preprocessing",
         encoded ? "Use the original encoded luma code values."
                 : "Convert to full-range encoded RGB using the source matrix/range; invert the selected transfer per "
                   "RGB channel BEFORE applying the descaler. The detector analyzes a fixed weighted linear RGB sum, "
                   "not gamma-transformed Y'. For color reconstruction, descale each linear RGB channel and re-encode "
                   "afterward."},
        {"linear_analysis_weights", Json::array({r.linearWeights[0], r.linearWeights[1], r.linearWeights[2]})},
        {"out_of_range_rule", "Signed extension of transfer functions; no clipping"},
        {"transfer_functions",
         Json::object({{"to_linear", toLinear},
                       {"to_encoded", toEncoded},
                       {"black_level", 0},
                       {"note", "linear-bt709 is the camera OETF inverse; linear-gamma24 models a zero-black display "
                                "gamma, as in ordinary zimg BT.709-to-linear conversion. Verify conventions in your "
                                "descaler."}})},
        {"selection_rule", "For unresolved close fits, prefer a preset with the smallest support within the 8% "
                           "ambiguity band. This is a parsimony choice, not identification."},
        {"grid_status", sameGrid ? r.gridStatus : "inconclusive"},
        {"automatic_grid_status", r.gridStatus},
        {"kernel_status", index == 0 ? r.kernelStatus : "unresolved"},
        {"observed", Json::object({{"width", r.width},
                                   {"height", r.height},
                                   {"plane", encoded ? "encoded_luma" : "weighted_linear_rgb"},
                                   {"samples", "See light_domain, required_preprocessing and analysis provenance."}})},
        {"native", Json::object({{"width", c.x.size}, {"height", c.y.size}})},
        {"geometry",
         Json::object({
             {"extent_x", op.x.extent},
             {"extent_y", op.y.extent},
             {"offset_x", (op.x.size - op.x.extent) * .5 + op.x.shift},
             {"offset_y", (op.y.size - op.y.extent) * .5 + op.y.shift},
             {"x_status", c.x.fractionalSupported ? "fractional_supported_heuristically" : "integer_conservative"},
             {"y_status", c.y.fractionalSupported ? "fractional_supported_heuristically" : "integer_conservative"},
             {"mapping_x", "native_x = (observed_x + 0.5) * geometry.extent_x / observed.width + geometry.offset_x - 0.5"},
             {"mapping_y",
              "native_y = (observed_y + 0.5) * geometry.extent_y / observed.height + geometry.offset_y - 0.5"},
             {"refined_fit", Json::object({{"extent_x", c.x.extent},
                                           {"extent_y", c.y.extent},
                                           {"offset_x", (c.x.size - c.x.extent) * .5 + c.x.shift},
                                           {"offset_y", (c.y.size - c.y.extent) * .5 + c.y.shift},
                                           {"relative_gain_x", c.x.refinementGain},
                                           {"relative_gain_y", c.y.refinementGain}})},
         })},
        {"kernel", selectedKernel},
        {"border", Json::object({{"model", "clamp"}, {"identified", false}, {"unscored_margin_pixels", 12}})},
        {"normalization", "Per output row: accumulate weights at clamped source indices, then divide weights by their sum."},
        {"selected_rank", int(index + 1)},
        {"selected_relative_score", c.score / std::max(1e-30, r.candidates.front().score)},
        {"automatic_verdict", r.verdict},
        {"automatic_kernel_gap", r.separation},
        {"limitations",
         Json::array({"Grid/kernel status is heuristic, not a probability or production-pipeline identification.",
                      "Continuous cubic coefficients approximate processing; they do not identify the production kernel.",
                      "Fractional geometry requires >1% gain on both alternating line subsets. Those subsets are "
                      "correlated, not independent statistical replicates."})},
    });
}

Json axis(const AxisFit &f) {
    return Json::object({{"samples", f.size},
                         {"extent", f.extent},
                         {"shift", f.shift},
                         {"kernel", f.kernel.name()},
                         {"family", int(f.kernel.family)},
                         {"b", f.kernel.b},
                         {"c", f.kernel.c},
                         {"taps", f.kernel.taps},
                         {"score", f.score},
                         {"mse", f.mse},
                         {"prominence", f.prominence},
                         {"variance_scaled_withheld_ratio", f.predictive < 0 ? Json() : Json(f.predictive)},
                         {"refinement_gain", f.refinementGain},
                         {"fractional_supported_heuristically", f.fractionalSupported},
                         {"identifiable", f.identifiable},
                         {"min_cholesky_pivot_ratio", f.pivotRatio},
                         {"informative_pixels", f.informative}});
}

std::string utcNow() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    char text[32];
    std::strftime(text, sizeof text, "%Y-%m-%dT%H:%M:%SZ", &utc);
    return text;
}

Json report(const Record &record) {
    const auto &r = record.result;
    const auto &f = record.frame;
    Json candidates = Json::array(), horizontal = Json::array(), vertical = Json::array(), kernels = Json::array();
    Json foldX = Json::array(), foldY = Json::array(), alternativesX = Json::array(), alternativesY = Json::array();
    for (const auto &c : r.candidates)
        candidates.append(Json::object({{"x", axis(c.x)},
                                        {"y", axis(c.y)},
                                        {"score", c.score},
                                        {"shared_geometry_score", c.sharedGeometryScore},
                                        {"even_line_score", c.evenScore},
                                        {"odd_line_score", c.oddScore},
                                        {"continuous_approximation", c.continuous},
                                        {"full_frame_rmse", c.rmse < 0 ? Json() : Json(c.rmse)},
                                        {"spatial_support", c.spatialSupport < 0 ? Json() : Json(c.spatialSupport)}}));
    for (const auto &a : r.alternativesX) alternativesX.append(axis(a));
    for (const auto &a : r.alternativesY) alternativesY.append(axis(a));
    for (const auto &a : r.horizontal) horizontal.append(axis(a));
    for (const auto &a : r.vertical) vertical.append(axis(a));
    for (const auto &k : r.options.kernels) kernels.append(k.name());
    auto folds = [](const std::vector<FoldPoint> &points, Json &dest) {
        for (const auto &p : points)
            dest.append(Json::object({{"extent", p.extent},
                                      {"coherence", p.coherence},
                                      {"bins", p.bins},
                                      {"high_energy", p.highEnergy},
                                      {"permuted", p.permuted},
                                      {"excess", p.excess},
                                      {"participation", p.participation},
                                      {"prominence", p.prominence},
                                      {"signals", p.sampleLines}}));
    };
    folds(r.foldX, foldX);
    folds(r.foldY, foldY);
    const auto &o = r.options;
    return Json::object({
        {"schema", "vernier.analysis.v2"},
        {"application", "Vernier 0.2"},
        {"created_utc", utcNow()},
        {"light_domain", r.lightDomain},
        {"color_assumptions", f.colorAssumptions},
        {"keyframes_only_decode", f.keyframeOnly},
        {"source", f.path},
        {"sample", f.sample},
        {"requested_seconds", f.requestedSeconds},
        {"provenance", f.provenance},
        {"source_width", r.width},
        {"source_height", r.height},
        {"analysis_seconds", r.seconds},
        {"verdict", r.verdict},
        {"explanation", r.explanation},
        {"kernel_separation", r.separation},
        {"parameters", parameters(r, r.recommendedCandidate)},
        {"recommended_rank", int(r.recommendedCandidate + 1)},
        {"competing_grids", r.competingGrids},
        {"grid_status", r.gridStatus},
        {"kernel_status", r.kernelStatus},
        {"kernel_axes_and_splits_agree", r.kernelConsistent},
        {"rejected_singular_trial_fits", r.rejectedSingularFits},
        {"search", Json::object({{"min_height", o.minHeight},
                                 {"max_height", o.maxHeight},
                                 {"min_width", o.minWidth},
                                 {"max_width", o.maxWidth},
                                 {"lines", o.lines},
                                 {"refinement_lines", o.refinementLines},
                                 {"threads", o.threads},
                                 {"processors", "process affinity unchanged"},
                                 {"refine", o.refine},
                                 {"kernels", kernels}})},
        {"limitations",
         Json::array({"Heuristic scores, not calibrated probabilities or p-values.",
                      "Luma-only separable single-stage operator models; mixed pipelines, nonlinear processing and "
                      "compression may be unidentifiable.",
                      "Five spectral peaks and three spatial notches per axis/family are retained. All preset "
                      "geometries are refined locally. Bicubic B/C search uses even lines and is bounded to [0,1]; "
                      "odd-line fit is recorded.",
                      "Kernel ranking uses each candidate's own geometry; shared-geometry scores are secondary. Extent "
                      "and kernel uncertainty are not confidence intervals.",
                      "Boundary is assumed clamped; 12 observed pixels per side are excluded from scoring.",
                      "Variance-scaled withheld ratios assume independent equal-variance noise. Compression violates "
                      "that assumption; these are diagnostics, not likelihood tests."})},
        {"candidates", candidates},
        {"horizontal", horizontal},
        {"vertical", vertical},
        {"fold_x", foldX},
        {"fold_y", foldY},
        {"alternatives_x", alternativesX},
        {"alternatives_y", alternativesY},
        {"kernel_evidence", Json::raw(kernelEvidenceJson(*record.evidence))},
    });
}

Json summarize(const std::vector<View> &views, int plannedFrames, bool partial) {
    std::map<std::string, std::vector<size_t>> groups;
    std::set<std::pair<std::string, double>> frames;
    for (size_t i = 0; i < views.size(); ++i) {
        const auto &r = *views[i].first;
        const auto &f = *views[i].second;
        groups[f.path + "\n" + std::to_string(r.width) + "x" + std::to_string(r.height) + "\n" + r.lightDomain]
            .push_back(i);
        frames.emplace(f.path, f.requestedSeconds);
    }
    Json output = Json::array();
    for (const auto &[key, ids] : groups) {
        const auto &first = *views[ids.front()].first;
        const auto &source = *views[ids.front()].second;
        std::map<Grid, std::vector<size_t>> votes;
        std::map<std::string, int> abstentions;
        int supported = 0, boundary = 0;
        double seconds = 0;
        for (auto id : ids) {
            const auto &r = *views[id].first;
            seconds += r.seconds;
            if (r.verdict == "Search boundary") ++boundary;
            if (r.gridStatus == "supported" && !r.candidates.empty()) {
                const auto &c = r.candidates.front();
                votes[{c.x.size, c.y.size}].push_back(id);
                ++supported;
            } else {
                ++abstentions[r.verdict];
            }
        }
        std::vector<std::pair<Grid, std::vector<size_t>>> ordered(votes.begin(), votes.end());
        std::sort(ordered.begin(), ordered.end(), [](const auto &a, const auto &b) {
            return a.second.size() != b.second.size() ? a.second.size() > b.second.size() : a.first < b.first;
        });
        const int lead = ordered.empty() ? 0 : int(ordered.front().second.size());
        const int second = ordered.size() < 2 ? 0 : int(ordered[1].second.size());
        const double count = double(ids.size());
        std::string state;
        if (supported < 3 || supported < .25 * count) state = "insufficient";
        else if (lead >= 3 && lead >= .9 * supported && second < 2 && boundary < .25 * count) state = "consistent";
        else if (lead > supported / 2.) state = "majority_with_dissent";
        else state = "split";
        Json grids = Json::array(), timeline = Json::array();
        for (const auto &[grid, members] : ordered) {
            Json frameIds = Json::array();
            for (auto id : members) frameIds.append(double(id));
            grids.append(Json::object({{"width", grid.first},
                                       {"height", grid.second},
                                       {"votes", int(members.size())},
                                       {"analysis_ids", frameIds}}));
        }
        for (auto id : ids) {
            const auto &r = *views[id].first;
            const auto &f = *views[id].second;
            Json item = Json::object({{"analysis_id", double(id)},
                                      {"seconds", f.requestedSeconds},
                                      {"vote", r.gridStatus == "supported" ? std::string("counted") : r.verdict}});
            if (!r.candidates.empty())
                item["leading_grid"] = gridJson({r.candidates.front().x.size, r.candidates.front().y.size});
            timeline.append(item);
        }
        Json abstain = Json::object();
        for (const auto &[reason, number] : abstentions) abstain[reason] = number;
        struct Model {
            Kernel kernel;
            double logs = 0;
            int coverage = 0, best = 0, identified = 0;
        };
        std::map<std::string, Model> models;
        if (!ordered.empty())
            for (auto id : ordered.front().second) {
                const auto &r = *views[id].first;
                const auto grid = ordered.front().first;
                double minimum = std::numeric_limits<double>::infinity();
                for (const auto &c : r.candidates)
                    if (!c.continuous && c.x.size == grid.first && c.y.size == grid.second && std::isfinite(c.score))
                        minimum = std::min(minimum, c.score);
                if (!std::isfinite(minimum)) continue;
                for (size_t rank = 0; rank < r.candidates.size(); ++rank) {
                    const auto &c = r.candidates[rank];
                    if (c.continuous || c.x.size != grid.first || c.y.size != grid.second || !std::isfinite(c.score))
                        continue;
                    auto &m = models[c.x.kernel.name()];
                    m.kernel = c.x.kernel;
                    m.logs += std::log(std::max(c.score, 1e-30) / std::max(minimum, 1e-30));
                    ++m.coverage;
                    if (c.score == minimum) ++m.best;
                    if (rank == 0 && r.kernelStatus == "supported_model") ++m.identified;
                }
            }
        std::vector<std::pair<std::string, Model>> ranked(models.begin(), models.end());
        std::sort(ranked.begin(), ranked.end(), [lead](const auto &a, const auto &b) {
            const bool ac = a.second.coverage >= .8 * lead, bc = b.second.coverage >= .8 * lead;
            if (ac != bc) return ac > bc;
            return a.second.logs / a.second.coverage < b.second.logs / b.second.coverage;
        });
        double bestMean = 0;
        int eligible = 0;
        for (const auto &m : ranked)
            if (m.second.coverage >= .8 * lead) {
                if (!eligible) bestMean = std::exp(m.second.logs / m.second.coverage);
                ++eligible;
            }
        Json modelRows = Json::array(), band = Json::array();
        std::string recommendation;
        double radius = 1e30;
        int identifiedElsewhere = 0;
        for (size_t i = 0; i < ranked.size(); ++i) {
            const auto &[name, m] = ranked[i];
            const bool covered = m.coverage >= .8 * lead;
            const double mean = std::exp(m.logs / m.coverage), relative = bestMean > 0 ? mean / bestMean : 1;
            modelRows.append(Json::object({{"name", name},
                                           {"family", familyName(m.kernel.family)},
                                           {"params", kernelParams(m.kernel)},
                                           {"frames", m.coverage},
                                           {"eligible", covered},
                                           {"geometric_mean_ratio", mean},
                                           {"relative_to_best_eligible", relative},
                                           {"best_in", m.best},
                                           {"supported_in", m.identified}}));
            if (covered && relative <= 1 / .92) {
                band.append(name);
                if (m.kernel.radius() < radius) {
                    radius = m.kernel.radius();
                    recommendation = name;
                }
            }
            if (i > 0) identifiedElsewhere += m.identified;
        }
        bool identified = state == "consistent" && eligible >= 2 && !ranked.empty() &&
                          ranked[0].second.identified >= 3 && identifiedElsewhere == 0;
        if (identified) {
            const auto &next = ranked[1].second;
            identified = bestMean / std::exp(next.logs / next.coverage) <= .92;
        }
        Json group = Json::object({
            {"source", source.path},
            {"source_width", first.width},
            {"source_height", first.height},
            {"light_domain", first.lightDomain},
            {"analyzed", int(ids.size())},
            {"supported", supported},
            {"abstained", int(ids.size()) - supported},
            {"state", state},
            {"search_boundary_frames", boundary},
            {"grid_votes", grids},
            {"abstentions", abstain},
            {"timeline", timeline},
            {"mean_analysis_seconds", seconds / count},
            {"kernel", Json::object({{"status", identified ? "supported_model" : "unresolved"},
                                     {"table", modelRows},
                                     {"ambiguity_band", band},
                                     {"parsimony_choice", recommendation},
                                     {"rule", "Geometric mean of per-frame ratios to the best preset at the same grid. "
                                              "Rank only presets covering at least 80% of members. Smallest-support "
                                              "preset within the 8% band is a recommendation, not identification."}})},
            {"geometry", "Integer grids, phases and continuous B/C coefficients are not averaged. Use a per-frame "
                         "parameter export; summary is evidence, not a fabricated resampling operator."},
        });
        if (!ordered.empty()) group["leading_grid"] = gridJson(ordered.front().first);
        group["light_status"] = "not_identified";
        group["kernel_status_scope"] = "conditional_on_light_domain";
        output.append(group);
    }
    return Json::object({
        {"schema", "vernier.summary.v1"},
        {"planned_frames", plannedFrames},
        {"analyzed_frames", int(frames.size())},
        {"domain_analyses", int(views.size())},
        {"partial", partial},
        {"groups", output},
        {"rules", Json::object({{"minimum_supported_frames", 3},
                                {"minimum_supported_fraction", .25},
                                {"consistent_leader_fraction", .9},
                                {"minimum_kernel_coverage", .8},
                                {"grid_votes", "One automatic supported-grid vote per frame per light domain; exact "
                                               "dimensions only. Manual selections never change votes."}})},
        {"caveats",
         Json::array({"Frames from one source are correlated; counts and geometric means are not probabilities.",
                      "Sources, observed dimensions and light domains are summarized separately. Raw residuals are "
                      "not comparable across light domains.",
                      "Unsupported frames remain abstentions. A cap is an upper bound; fewer candidates yield fewer "
                      "analyzed frames.",
                      "Linear RGB hypotheses can be weakened by reconstructed chroma, compression and unknown color "
                      "transforms."})},
    });
}

// batch.cpp's batchSummary: the summary, then one paired grid vote per source frame across its light domains.
Json batchSummary(const std::vector<Record> &records, const GetnativeBatch &batch) {
    std::vector<View> views;
    for (const auto &r : records) views.emplace_back(&r.result, &r.frame);
    const bool partial = batch.error[0] != 0 || batch.completed < batch.planned;
    auto summary = summarize(views, batch.planned, partial);
    summary["completed_frames_all_domains"] = batch.completed;
    summary["sampling"] = Json::raw(batch.sampling);
    summary["error"] = batch.error;
    std::map<std::pair<std::string, double>, std::vector<size_t>> pairs;
    for (size_t i = 0; i < records.size(); ++i)
        pairs[{records[i].frame.path, records[i].frame.requestedSeconds}].push_back(i);
    std::vector<Result> joint;
    std::vector<Frame> metadata;
    std::vector<int> originalIds;
    for (const auto &[id, members] : pairs) {
        const auto &front = records[members.front()].result;
        Result result;
        result.width = front.width;
        result.height = front.height;
        result.lightDomain = "across-tested-domains";
        result.verdict = front.verdict;
        const Candidate *chosen = nullptr;
        bool conflict = false, boundary = false;
        for (auto m : members)
            if (records[m].result.verdict == "Search boundary") boundary = true;
        for (auto m : members) {
            const auto &r = records[m].result;
            if (r.gridStatus == "supported" && !r.candidates.empty()) {
                const auto &c = r.candidates.front();
                if (chosen && (c.x.size != chosen->x.size || c.y.size != chosen->y.size)) conflict = true;
                else chosen = &c;
            }
        }
        if (members.size() < size_t(batch.domains)) result.verdict = "Incomplete light-domain coverage";
        else if (boundary) result.verdict = "Search boundary";
        else if (conflict) result.verdict = "Grid depends on light";
        else if (chosen) {
            result.gridStatus = "supported";
            result.verdict = "Agreed among supported light domains";
            result.candidates.push_back(*chosen);
        }
        Frame frame;
        frame.path = id.first;
        frame.requestedSeconds = id.second;
        joint.push_back(std::move(result));
        metadata.push_back(std::move(frame));
        originalIds.push_back(int(members.front()));
    }
    std::vector<View> paired;
    for (size_t i = 0; i < joint.size(); ++i) paired.emplace_back(&joint[i], &metadata[i]);
    auto aggregate = summarize(paired, batch.planned, partial);
    Json groups = aggregate["groups"];
    for (auto &group : groups.items()) {
        group.remove("kernel");
        group.remove("mean_analysis_seconds");
        for (auto &t : group["timeline"].items()) t["analysis_id"] = originalIds[size_t(t["analysis_id"].number())];
        for (auto &v : group["grid_votes"].items())
            for (auto &id : v["analysis_ids"].items()) id = originalIds[size_t(id.number())];
        group["light_status"] = "not_identified";
        group["rule"] = "One vote per source frame only after all requested domains complete, with no search-boundary "
                        "result and agreement among all supported grids. Conflicting domains abstain. No comparison of "
                        "raw residuals across domains.";
    }
    summary["paired_grid_consensus"] = groups;
    summary["light_status"] = "not_identified";
    summary["requested_domains_per_frame"] = batch.domains;
    return summary;
}

const Record &recordAt(const Session &session, int frame) {
    if (frame < 0 || size_t(frame) >= session.records.size())
        throw std::invalid_argument("No analysis " + std::to_string(frame) + " in this session");
    return session.records[size_t(frame)];
}

// The index of a result's candidate at a grid with the lowest finite score among the presets (continuous == false)
// or among the free bicubics (continuous == true); the candidate count when it has none there.
size_t bestAt(const Result &r, Grid grid, bool continuous) {
    size_t best = r.candidates.size();
    for (size_t k = 0; k < r.candidates.size(); ++k) {
        const auto &c = r.candidates[k];
        if (c.continuous == continuous && Grid{c.x.size, c.y.size} == grid && std::isfinite(c.score) &&
            (best == r.candidates.size() || c.score < r.candidates[best].score))
            best = k;
    }
    return best;
}

// What the node's outputs carry, from the As stored (encoded) analyses: the consensus grid (the supported votes'
// leader, else the most frequent leading grid), the kernel (the cross-frame decision when it identifies one or
// narrows a band, else the smallest-support preset within the 8 % band: of the summary's geometric-mean table for
// several frames, of the frame's own ranking for one), and the frame whose candidate for that kernel at the grid
// fits best relative to its best preset, supported frames first. The free bicubic is never carried: where the
// frame's own ranking recommends it (no preset within 8 % of it), its best-scoring preset is carried and the
// selection's free_bicubic gives the free bicubic's B, C, its advantage and its rank (for Candidate).
// candidate >= 0 overrides the rank (manual).
void select(const Session &session, Json &summary, int candidate) {
    std::vector<size_t> encoded;
    for (size_t i = 0; i < session.records.size(); ++i)
        if (session.records[i].frame.lightDomain == "encoded") encoded.push_back(i);
    std::map<Grid, int> supportedVotes, leading;
    for (auto id : encoded) {
        const auto &r = session.records[id].result;
        if (r.candidates.empty()) continue;
        const Grid grid{r.candidates.front().x.size, r.candidates.front().y.size};
        ++leading[grid];
        if (r.gridStatus == "supported") ++supportedVotes[grid];
    }
    const auto &votes = supportedVotes.empty() ? leading : supportedVotes;
    Grid consensus{0, 0};
    int most = 0;
    for (const auto &[grid, number] : votes)
        if (number > most) {
            most = number;
            consensus = grid;
        }
    std::vector<size_t> members;
    std::vector<std::shared_ptr<KernelEvidence>> evidence;
    for (auto id : encoded) {
        const auto &r = session.records[id].result;
        if (!r.candidates.empty() && Grid{r.candidates.front().x.size, r.candidates.front().y.size} == consensus) {
            members.push_back(id);
            evidence.push_back(session.records[id].evidence);
        }
    }
    const auto decision = decideKernel(evidence, consensus.first, consensus.second);
    summary["kernel_decision"] = Json::raw(kernelDecisionJson(decision));
    if (members.empty()) {
        if (candidate >= 0) throw std::invalid_argument("No analysis found a candidate to choose from");
        summary["selection"] = Json();
        return;
    }
    std::string carried, rule;
    if ((decision.status == "supported_model" || decision.status == "ambiguous_band") && !decision.carried.empty()) {
        carried = decision.carried;
        rule = "kernel decision " + decision.status + " (" + decision.rule + " rule)";
    } else if (encoded.size() > 1) {
        for (const auto &group : summary["groups"].items()) {
            const auto *domain = group.find("light_domain");
            const auto *grid = group.find("leading_grid");
            if (!domain || domain->text() != "encoded" || !grid ||
                Grid{int(grid->items()[0].number()), int(grid->items()[1].number())} != consensus)
                continue;
            const auto &kernel = *group.find("kernel");
            if (kernel.find("status")->text() == "supported_model") {
                carried = kernel.find("table")->items().front().find("name")->text();
                rule = "identified across frames (geometric-mean table, 8 % margin)";
            } else if (!kernel.find("parsimony_choice")->text().empty()) {
                carried = kernel.find("parsimony_choice")->text();
                rule = "smallest support within the 8 % band of the geometric-mean table (parsimony, not "
                       "identification)";
            }
        }
    }
    // The frame's own choice: its recommended candidate, or its best-scoring preset at the grid when that is the
    // free bicubic.
    bool freeBicubic = false;
    auto frameChoice = [&](const Result &r) {
        const size_t preset = bestAt(r, consensus, false);
        freeBicubic = r.candidates[r.recommendedCandidate].continuous && preset < r.candidates.size();
        if (freeBicubic)
            return std::pair{preset,
                             std::string("the best-scoring preset (no preset is within 8 % of the free bicubic)")};
        return std::pair{r.recommendedCandidate,
                         r.kernelStatus == "supported_model"
                             ? std::string("identified on the frame (8 % separation, axes and line splits agree)")
                             : std::string("smallest support within the 8 % band of the frame's best preset "
                                           "(parsimony, not identification)")};
    };
    if (carried.empty()) {
        const auto &r = session.records[members.front()].result;
        const auto [k, why] = frameChoice(r);
        carried = r.candidates[k].x.kernel.name();
        rule = why;
    }
    size_t frame = members.front(), index = session.records[frame].result.recommendedCandidate;
    std::tuple<bool, double, size_t> best{true, std::numeric_limits<double>::infinity(), frame};
    bool found = false;
    for (auto id : members) {
        const auto &r = session.records[id].result;
        const size_t top = bestAt(r, consensus, false);
        if (top == r.candidates.size()) continue;
        for (size_t k = 0; k < r.candidates.size(); ++k) {
            const auto &c = r.candidates[k];
            if (c.continuous || Grid{c.x.size, c.y.size} != consensus || c.x.kernel.name() != carried ||
                !std::isfinite(c.score))
                continue;
            const std::tuple<bool, double, size_t> key{r.gridStatus != "supported",
                                                       c.score / r.candidates[top].score, id};
            if (!found || key < best) {
                found = true;
                best = key;
                frame = id;
                index = k;
            }
        }
    }
    if (!found) {
        const auto &r = session.records[frame].result;
        std::tie(index, rule) = frameChoice(r);
        carried = r.candidates[index].x.kernel.name();
    }
    const auto &chosen = session.records[frame].result;
    const bool manual = candidate >= 0;
    if (manual) {
        if (size_t(candidate) >= chosen.candidates.size())
            throw std::invalid_argument("Candidate " + std::to_string(candidate + 1) + " is past the " +
                                        std::to_string(chosen.candidates.size()) + " ranked candidates of analysis " +
                                        std::to_string(frame));
        index = size_t(candidate);
        carried = chosen.candidates[index].x.kernel.name();
        rule = "manual override: rank " + std::to_string(candidate + 1) + " of analysis " + std::to_string(frame);
    }
    Json selection = Json::object({{"analysis_id", double(frame)},
                                   {"candidate_index", double(index)},
                                   {"rank", double(index + 1)},
                                   {"kernel", carried},
                                   {"rule", rule},
                                   {"selection", manual ? "manual_override" : "automatic_recommendation"},
                                   {"consensus_grid", gridJson(consensus)},
                                   {"frames_on_consensus_grid", int(members.size())}});
    const size_t cubic = bestAt(chosen, consensus, true);
    if (freeBicubic && !manual && cubic < chosen.candidates.size() &&
        chosen.candidates[cubic].score < chosen.candidates[index].score) {
        const auto &c = chosen.candidates[cubic];
        selection["free_bicubic"] = Json::object({{"b", c.x.kernel.b},
                                                  {"c", c.x.kernel.c},
                                                  {"advantage", chosen.candidates[index].score / c.score - 1},
                                                  {"rank", double(cubic + 1)}});
    }
    summary["selection"] = selection;
}

void *create(const cn_getnative_options &o) {
    auto session = std::make_unique<Session>();
    auto &options = session->options;
    options.minHeight = o.min_height;
    options.maxHeight = o.max_height;
    options.minWidth = o.min_width;
    options.maxWidth = o.max_width;
    options.lines = o.lines;
    options.refinementLines = o.refinement_lines;
    options.threads = o.threads > 0 ? o.threads : defaultWorkers();
    options.refine = o.refine != 0;
    switch (o.kernels) {
    case 0: break;
    case 1: options.kernels = {{Family::Bilinear}}; break;
    case 2: options.kernels = {{Family::Bicubic, 0, .5}}; break;
    case 3: options.kernels = {{Family::Lanczos, 0, 0, 3}}; break;
    default: throw std::invalid_argument("Unknown kernel selection");
    }
    return session.release();
}

void destroy(void *state) { delete static_cast<Session *>(state); }

void addFrame(void *state, const double *plane, int width, int height, const cn_getnative_frame &facts,
              std::atomic_bool *cancel) {
    auto &session = *static_cast<Session *>(state);
    if (!plane || width < 1 || height < 1) throw std::invalid_argument("Invalid analysis plane");
    const Image image{width, height, std::vector<double>(plane, plane + size_t(width) * size_t(height))};
    Record record;
    auto &f = record.frame;
    f.path = facts.source;
    f.sample = facts.sample;
    f.provenance = facts.provenance;
    f.lightDomain = facts.light_domain;
    f.colorAssumptions = facts.color_assumptions;
    f.requestedSeconds = facts.requested_seconds;
    f.linearWeights = {facts.weights[0], facts.weights[1], facts.weights[2]};
    f.keyframeOnly = facts.keyframe_only != 0;
    record.result = analyze(image, session.options, cancel);
    record.result.lightDomain = f.lightDomain;
    record.result.colorAssumptions = f.colorAssumptions;
    record.result.linearWeights = f.linearWeights;
    record.evidence = measureKernelEvidence(image, record.result, session.options, cancel);
    session.records.push_back(std::move(record));
}

std::string curvesCsv(const Result &r) {
    std::string csv = "axis,kernel,native_samples,extent,shift,score,mse,prominence\n";
    auto add = [&](const std::vector<AxisFit> &fits, const char *axisName) {
        for (const auto &f : fits) {
            char numbers[160];
            std::snprintf(numbers, sizeof numbers, ",%d,%.12g,%.12g,%.12g,%.12g,%.12g\n", f.size, f.extent, f.shift,
                          f.score, f.mse, f.prominence);
            csv += std::string(axisName) + ",\"" + f.kernel.name() + "\"" + numbers;
        }
    };
    add(r.horizontal, "horizontal");
    add(r.vertical, "vertical");
    return csv;
}

std::string render(void *state, GetnativeText what, int frame, int candidate, const GetnativeBatch &batch) {
    const auto &session = *static_cast<Session *>(state);
    switch (what) {
    case GetnativeText::Report: return report(recordAt(session, frame)).dump();
    case GetnativeText::Parameters: {
        const auto &r = recordAt(session, frame).result;
        const size_t index = candidate < 0 ? r.recommendedCandidate : size_t(candidate);
        if (index >= r.candidates.size())
            throw std::invalid_argument("Analysis " + std::to_string(frame) + " has no candidate " +
                                        std::to_string(index + 1));
        return parameters(r, index).dump();
    }
    case GetnativeText::Summary: {
        auto summary = batchSummary(session.records, batch);
        Json analyses = Json::array();
        for (size_t i = 0; i < session.records.size(); ++i) {
            const auto &record = session.records[i];
            const auto &r = record.result;
            Json leading = Json::array();
            for (size_t k = 0; k < std::min<size_t>(3, r.candidates.size()); ++k) {
                const auto &c = r.candidates[k];
                leading.append(Json::object({{"width", c.x.size},
                                             {"height", c.y.size},
                                             {"kernel", c.x.kernel.name()},
                                             {"continuous_approximation", c.continuous},
                                             {"relative_score", c.score / std::max(r.candidates.front().score, 1e-30)}}));
            }
            analyses.append(Json::object({{"analysis_id", double(i)},
                                          {"light_domain", r.lightDomain},
                                          {"sample", record.frame.sample},
                                          {"requested_seconds", record.frame.requestedSeconds},
                                          {"source_width", r.width},
                                          {"source_height", r.height},
                                          {"analysis_seconds", r.seconds},
                                          {"verdict", r.verdict},
                                          {"explanation", r.explanation},
                                          {"grid_status", r.gridStatus},
                                          {"kernel_status", r.kernelStatus},
                                          {"kernel_separation", r.separation},
                                          {"competing_grids", r.competingGrids},
                                          {"recommended_rank", int(r.recommendedCandidate + 1)},
                                          {"leading_candidates", leading},
                                          {"kernel_evidence", Json::raw(kernelEvidenceJson(*record.evidence))}}));
        }
        summary["analyses"] = analyses;
        select(session, summary, candidate);
        return summary.dump();
    }
    case GetnativeText::Curves: return curvesCsv(recordAt(session, frame).result);
    }
    throw std::invalid_argument("Unknown text");
}

size_t curve(void *state, int frame, bool horizontal, int *sizes, double *scores, size_t capacity) {
    const auto &r = recordAt(*static_cast<Session *>(state), frame).result;
    std::map<int, double> best;
    for (const auto &f : horizontal ? r.horizontal : r.vertical)
        if (std::isfinite(f.score)) {
            auto [place, inserted] = best.emplace(f.size, f.score);
            if (!inserted) place->second = std::min(place->second, f.score);
        }
    size_t i = 0;
    for (const auto &[size, score] : best) {
        if (i < capacity) {
            sizes[i] = size;
            scores[i] = score;
        }
        ++i;
    }
    return i;
}
}
}

const GetnativeUnit *CN_GETNATIVE_UNIT() {
    static const GetnativeUnit unit{native::create, native::destroy, native::addFrame, native::render, native::curve};
    return &unit;
}
