#include "llama-expert-base.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace {

// cut a trailing comment: a '#' at the start of the line or behind whitespace
// ends the meaningful part, so the documented inline comments parse
std::string strip_comment(const std::string & line) {
    for (size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '#' && (i == 0 || std::isspace((unsigned char) line[i - 1]))) {
            return line.substr(0, i);
        }
    }
    return line;
}

// whitespace-separated fields of `line`, empty fields dropped
std::vector<std::string> split_ws(const std::string & line) {
    std::vector<std::string> out;
    size_t                   i = 0;
    while (i < line.size()) {
        while (i < line.size() && std::isspace((unsigned char) line[i])) {
            i++;
        }
        const size_t b = i;
        while (i < line.size() && !std::isspace((unsigned char) line[i])) {
            i++;
        }
        if (i > b) {
            out.push_back(line.substr(b, i - b));
        }
    }
    return out;
}

std::vector<std::string> split_commas(const std::string & s) {
    std::vector<std::string> out;
    size_t                   b = 0;
    for (size_t i = 0; i <= s.size(); i++) {
        if (i == s.size() || s[i] == ',') {
            size_t e = i;
            while (b < e && std::isspace((unsigned char) s[b])) {
                b++;
            }
            while (e > b && std::isspace((unsigned char) s[e - 1])) {
                e--;
            }
            if (e > b) {
                out.push_back(s.substr(b, e - b));
            }
            b = i + 1;
        }
    }
    return out;
}

std::string to_lower(std::string s) {
    for (char & c : s) {
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

bool is_sep(char c) {
    return c == '_' || c == ':' || c == '.' || c == '-' || c == '/';
}

// directory part of `path`, including the trailing separator ("" when the file
// has no directory part)
std::string dir_of(const std::string & path) {
    const size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? std::string() : path.substr(0, pos + 1);
}

bool is_abs(const std::string & path) {
    return !path.empty() && (path[0] == '/' || path[0] == '\\' || (path.size() > 1 && path[1] == ':'));
}

} // namespace

bool llama_expert_base_tool_match(const std::string & tool, const std::string & name) {
    const std::string t = to_lower(tool);
    const std::string n = to_lower(name);
    if (n.empty()) {
        return false;
    }
    if (t == n) {
        return true;
    }
    // the size check must come after the equality test: equal lengths that are
    // not equal would underflow the suffix index below
    if (t.size() <= n.size()) {
        return false;
    }
    // a namespaced tool matches its trailing name: mcp__fs__read_file -> read_file
    return is_sep(t[t.size() - n.size() - 1]) && t.compare(t.size() - n.size(), n.size(), n) == 0;
}

std::string llama_expert_base_template::select(const std::vector<std::string> & tools) const {
    for (const rule & r : rules) {
        for (const std::string & t : tools) {
            for (const std::string & n : r.tools) {
                if (llama_expert_base_tool_match(t, n)) {
                    return r.set;
                }
            }
        }
    }
    return default_set;
}

void llama_expert_base_template_parse(const std::string & path, llama_expert_base_template & out) {
    out = llama_expert_base_template{};

    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("expert base template: cannot open '" + path + "'");
    }

    const std::string dir = dir_of(path);
    const auto        resolve = [&](const std::string & file) {
        return is_abs(file) || dir.empty() ? file : dir + file;
    };

    std::string line;
    int         lineno      = 0;
    bool        have_header = false;

    const auto fail = [&](const std::string & msg) {
        throw std::runtime_error("expert base template: " + path + ":" + std::to_string(lineno) + ": " + msg);
    };

    while (std::getline(in, line)) {
        lineno++;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const std::vector<std::string> f = split_ws(strip_comment(line));
        if (f.empty() || f[0][0] == '#') {
            continue;
        }

        if (!have_header) {
            if (f.size() != 2 || f[0] != "llama-expert-base-template" || f[1] != "v1") {
                fail("expected header 'llama-expert-base-template v1'");
            }
            have_header = true;
            continue;
        }

        const std::string & key = f[0];
        if (key == "base") {
            if (f.size() != 2) {
                fail("expected 'base <file>'");
            }
            if (!out.base_file.empty()) {
                fail("duplicate base set");
            }
            out.base_file = resolve(f[1]);
        } else if (key == "set") {
            if (f.size() != 3) {
                fail("expected 'set <name> <file>'");
            }
            if (out.set_files.count(f[1]) != 0) {
                fail("duplicate set '" + f[1] + "'");
            }
            out.set_files[f[1]] = resolve(f[2]);
        } else if (key == "default") {
            if (f.size() != 2) {
                fail("expected 'default <set>'");
            }
            if (!out.default_set.empty()) {
                fail("duplicate default set");
            }
            out.default_set = f[1];
        } else if (key == "fence") {
            if (f.size() != 2) {
                fail("expected 'fence <set>'");
            }
            if (!out.fence_set.empty()) {
                fail("duplicate fence set");
            }
            out.fence_set = f[1];
        } else if (key == "tools") {
            // the tool list may contain spaces after the commas, so everything
            // but the last field is the list
            if (f.size() < 3) {
                fail("expected 'tools <name>[,<name>...] <set>'");
            }
            llama_expert_base_template::rule r;
            r.set = f.back();
            for (size_t i = 1; i + 1 < f.size(); i++) {
                for (const std::string & t : split_commas(f[i])) {
                    r.tools.push_back(t);
                }
            }
            out.rules.push_back(std::move(r));
        } else {
            fail("unknown keyword '" + key + "'");
        }
    }

    if (!have_header) {
        throw std::runtime_error("expert base template: '" + path + "' has no header");
    }
    if (out.base_file.empty()) {
        throw std::runtime_error("expert base template: '" + path + "' declares no base set");
    }
    if (!out.default_set.empty() && out.set_files.count(out.default_set) == 0) {
        throw std::runtime_error("expert base template: '" + path + "': undeclared default set '" + out.default_set + "'");
    }
    if (!out.fence_set.empty() && out.set_files.count(out.fence_set) == 0) {
        throw std::runtime_error("expert base template: '" + path + "': undeclared fence set '" + out.fence_set + "'");
    }
    for (const llama_expert_base_template::rule & r : out.rules) {
        if (r.tools.empty()) {
            throw std::runtime_error("expert base template: '" + path + "': rule for set '" + r.set + "' lists no tool");
        }
        if (out.set_files.count(r.set) == 0) {
            throw std::runtime_error("expert base template: '" + path + "': undeclared set '" + r.set + "'");
        }
    }
}
