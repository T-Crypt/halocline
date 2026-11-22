// src/models/inkling2/adapter.cpp - registers Inkling 2 Mini (`inkling2`): its shape from the pack's metadata, its
// tokenizer (a GPT-2 BPE vocabulary of 200,058 tokens with the GPT-4o pre-tokenizer), its special tokens and its tool
// calls (`<|message_model|>NAME<|content_invoke_tool_json|>{"name":..,"args":{..}}<|end_message|>`).
#include "models/inkling2/adapter.hpp"
#include "models/inkling2/model.hpp"

#include "halo/artifact/gguf_reader.hpp"
#include "halo/model/flags.hpp"
#include "halo/program/options.hpp"
#include "halo/program/gpt4o_pretok.hpp"

#include <cstdlib>

namespace halo::models::inkling2 {
namespace {

class Adapter final : public model::ModelAdapter {
public:
    const char* arch() const override { return "inkling2"; }

    std::string validate(const GgufFile& meta) const override {
        Hparams h;
        std::string err;
        return read_hparams(meta, h, err) ? std::string() : err;
    }

    model::ModelDesc describe(const GgufFile& meta) const override {
        Hparams h;
        std::string err;
        model::ModelDesc d;
        d.arch = "inkling2";
        if (!read_hparams(meta, h, err)) return d;
        d.n_layers = h.n_visit;
        d.n_embd = h.n_embd;
        d.n_vocab = h.n_vocab;
        d.n_expert = h.n_expert;
        d.n_expert_used = h.n_used;
        d.n_ff_exp = h.n_ff;
        d.context_length = h.context_length;
        return d;
    }

    model::Result load(const model::LoadRequest& req, std::unique_ptr<model::LoadedModel>& out) const override {
        if (req.paths.empty()) return {model::Status::Error, "inkling2: no model file"};
        std::string err;
        out = load_model(req, err);
        if (!out) return {model::Status::Error, err};
        return {};
    }

    model::TokenizerSpec tokenizer_spec() const override { return {"tokenizer.ggml.tokens", "inkling"}; }

    PreTokenizer pretokenizer() const override { return {"inkling2", program::tokenizer::pretok_gpt4o}; }

    model::TokenNames token_names() const override {
        // the template ends a reply with <|content_model_end_sampling|> (the GGUF's eos); a reply opens with the model's
        // message token
        // the template writes a picture as <|content_image|><|unused_200054|><|end_message|>: the placeholder is its rows
        return {{"<|content_model_end_sampling|>", "<|endoftext|>"}, "<|message_model|>", "<|unused_200054|>", "", ""};
    }

    program::grammar::ToolForm tool_form() const override {
        program::grammar::ToolForm f;
        f.call_open = "<|message_model|>";
        f.call_close = "<|end_message|>";
        f.fn_open = "";
        f.fn_mid = "<|content_invoke_tool_json|>";
        f.fn_close = "";
        f.trigger = "<|content_invoke_tool_json|>";
        return f;
    }

    const char* model_name() const override { return "Inkling-2-Mini"; }

    // --kv bf16|int8|k8v4|q4: the 8 full-attention caches (the load checks the value); --spec N: windows of N rows, the
    // DSpark drafter filling N - 1 of them
    bool parse_flag(const std::string& a, model::FlagArgs& args, program::CommonOptions& o, int& code) const override {
        (void) code;
        if (a == "--kv") o.kv = args.next("--kv");
        else if (a == "--spec") { o.spec = std::atoi(args.next("--spec")); o.spec_given = true; }
        else return false;
        return true;
    }
};

}  // namespace

std::unique_ptr<model::ModelAdapter> make_adapter() { return std::make_unique<Adapter>(); }

}  // namespace halo::models::inkling2
