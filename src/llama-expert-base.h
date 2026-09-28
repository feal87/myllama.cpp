#pragma once

// Parsing and matching for the MoE base-expert template
// (--pin-experts-template FILE).
//
// A template names the expert sets to keep resident in the disk decode cache
// and the conditions that pick between them:
//
//   llama-expert-base-template v1
//   base   base.txt                     # always resident
//   set    prose     prose.txt          # a named set
//   set    shellnav  shellnav.txt
//   set    code      code.txt
//   default prose                      # no tool matched
//   tools  bash,shell,run,cmd   shellnav
//   tools  edit,write,read      code
//
// The sets themselves keep the "llama-expert-base v1" format and are read by
// the disk stage; this file only maps conditions to set names. Rule order
// decides: the first rule matching any tool of the turn wins, the default set
// is used when none does.
//
// A tool name matches a rule name when it is equal to it or ends with it behind
// a separator, so a namespaced tool (mcp__fs__read_file) matches "read_file" and
// "file" but not "read". Matching is case-insensitive.
//
// Nothing here touches the model, the cache or the filesystem beyond reading
// the template, so it stays out of the platform-specific disk stage code.

#include <map>
#include <string>
#include <vector>

struct llama_expert_base_template {
    // expert set kept resident in every mode
    std::string base_file;
    // set used when no rule matches the turn's tools (empty = base only)
    std::string default_set;
    // set name -> expert set file, relative to the template's directory
    std::map<std::string, std::string> set_files;
    // conditions in file order; the first match wins
    struct rule {
        std::vector<std::string> tools;
        std::string              set;
    };
    std::vector<rule> rules;

    // set selected by the turn's tool names, empty when nothing matches and no
    // default was declared
    std::string select(const std::vector<std::string> & tools) const;
};

// Reads and validates `path`. Relative set files resolve against the
// template's own directory. Throws std::runtime_error with file:line on a
// malformed template, an unknown keyword, a duplicate set name or a rule that
// names a set that was not declared.
void llama_expert_base_template_parse(const std::string & path, llama_expert_base_template & out);

// true when `tool` matches `name` (equal, or a separator-delimited suffix)
bool llama_expert_base_tool_match(const std::string & tool, const std::string & name);
