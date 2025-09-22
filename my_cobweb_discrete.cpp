/**
 * Redesign of Cobweb with implementations necessary for Chunking-Cobweb framework!
 * --------------------------------------------------------------------------------
 */
// cobweb_discrete.cpp
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <queue>
#include <unordered_map>
#include <functional>
#include <random>
#include <tuple>
#include <unordered_set>
#include <chrono>
#include <cmath>
#include <variant>

#include "rapidjson/document.h"
#include "rapidjson/reader.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/writer.h"
#include "rapidjson/ostreamwrapper.h"
#include "rapidjson/stringbuffer.h"
#include <fstream>
#include <string>
#include <stack>

#include "assert.h"
#include "cached_string.hpp"
#include "BS_thread_pool.hpp"
#include "helper.h"

namespace nb = nanobind;

#define NULL_STRING CachedString("\0")
#define BEST 0
#define NEW 1
#define MERGE 2
#define SPLIT 3
#define BEST_NEW 4

// typedef CachedString ATTR_TYPE;
// typedef CachedString VALUE_TYPE;
// typedef double COUNT_TYPE;
// typedef std::unordered_map<std::string, std::unordered_map<std::string, COUNT_TYPE>> INSTANCE_TYPE;
// typedef std::unordered_map<VALUE_TYPE, COUNT_TYPE> VAL_COUNT_TYPE;
// typedef std::unordered_map<ATTR_TYPE, VAL_COUNT_TYPE> AV_COUNT_TYPE;
// typedef std::unordered_map<ATTR_TYPE, std::unordered_set<VALUE_TYPE>> AV_KEY_TYPE;
// typedef std::unordered_map<ATTR_TYPE, COUNT_TYPE> ATTR_COUNT_TYPE;
// typedef std::pair<double, int> OPERATION_TYPE;

typedef int ATTR_TYPE;
typedef int VALUE_TYPE;
typedef double COUNT_TYPE;
typedef std::unordered_map<int, std::unordered_map<int, COUNT_TYPE>> INSTANCE_TYPE;
typedef std::unordered_map<VALUE_TYPE, COUNT_TYPE> VAL_COUNT_TYPE;
typedef std::unordered_map<ATTR_TYPE, VAL_COUNT_TYPE> AV_COUNT_TYPE;
typedef std::unordered_map<ATTR_TYPE, std::unordered_set<VALUE_TYPE>> AV_KEY_TYPE;
typedef std::unordered_map<ATTR_TYPE, COUNT_TYPE> ATTR_COUNT_TYPE;
typedef std::pair<double, int> OPERATION_TYPE;

class CobwebTree;
class CobwebNode;

std::unordered_map<int, double> lgammaCache;
std::unordered_map<int, std::unordered_map<int, int>> binomialCache;
std::unordered_map<int, std::unordered_map<double, double>> entropy_k_cache;

// do this so that we only store integers in the stack, not strings
// won't have any impact on the tree
const std::unordered_map<std::string, int> ATTRIBUTE_MAP = {
    {"alpha", 1000000},
    {"weight_attr", 10000001},
    {"objective", 10000002},
    {"children_norm", 10000003},
    {"norm_attributes", 10000004},
    {"root", 10000005},
    {"count", 100000011},
    {"a_count", 100000012},
    {"sum_n_logn", 100000013},
    {"av_count", 100000014},
    {"children", 100000015}};

VALUE_TYPE most_likely_choice(std::vector<std::tuple<VALUE_TYPE, double>> choices)
{
    std::vector<std::tuple<double, double, VALUE_TYPE>> vals;

    for (auto &[val, prob] : choices)
    {
        if (prob < 0)
        {
            std::cout << "most_likely_choice: all weights must be greater than or equal to 0" << std::endl;
        }
        vals.push_back(std::make_tuple(prob, custom_rand(), val));
    }
    sort(vals.rbegin(), vals.rend());

    return std::get<2>(vals[0]);
}

VALUE_TYPE weighted_choice(std::vector<std::tuple<VALUE_TYPE, double>> choices)
{
    std::cout << "weighted_choice: Not implemented yet" << std::endl;
    return std::get<0>(choices[0]);
}

class CobwebNode
{

public:
    CobwebTree *tree;
    CobwebNode *parent;
    std::vector<CobwebNode *> children;

    COUNT_TYPE count;
    ATTR_COUNT_TYPE a_count;
    ATTR_COUNT_TYPE sum_n_logn;
    AV_COUNT_TYPE av_count;

    CobwebNode();
    CobwebNode(CobwebNode *otherNode);
    void increment_counts(const AV_COUNT_TYPE &instance);
    void update_counts_from_node(CobwebNode *node);
    double entropy_attr_insert(ATTR_TYPE attr, const AV_COUNT_TYPE &instance);
    double entropy_insert(const AV_COUNT_TYPE &instance);
    double entropy_attr_merge(ATTR_TYPE attr, CobwebNode *other, const AV_COUNT_TYPE &instance);
    double entropy_merge(CobwebNode *other, const AV_COUNT_TYPE
                                                &instance);
    CobwebNode *get_best_level(INSTANCE_TYPE instance);
    CobwebNode *get_basic_level();
    double category_utility();
    double entropy_attr(ATTR_TYPE attr);
    double entropy();
    double partition_utility();
    std::tuple<double, int> get_best_operation(const AV_COUNT_TYPE
                                                   &instance,
                                               CobwebNode *best1, CobwebNode *best2, double best1Cu);
    std::tuple<double, CobwebNode *, CobwebNode *>
    two_best_children(const AV_COUNT_TYPE &instance);
    std::vector<double> log_prob_children_given_instance(const AV_COUNT_TYPE &instance);
    std::vector<double> log_prob_children_given_instance_ext(INSTANCE_TYPE instance);
    std::vector<double> prob_children_given_instance(const AV_COUNT_TYPE &instance);
    std::vector<double> prob_children_given_instance_ext(INSTANCE_TYPE instance);
    double log_prob_instance(const AV_COUNT_TYPE &instance);
    double log_prob_instance_missing(const AV_COUNT_TYPE &instance);
    double log_prob_instance_ext(INSTANCE_TYPE instance);
    double log_prob_instance_missing_ext(INSTANCE_TYPE instance);
    double log_prob_class_given_instance(const AV_COUNT_TYPE &instance,
                                         bool use_root_counts = false);
    double log_prob_class_given_instance_ext(INSTANCE_TYPE instance,
                                             bool use_root_counts = false);
    double pu_for_insert(CobwebNode *child, const AV_COUNT_TYPE
                                                &instance);
    double pu_for_new_child(const AV_COUNT_TYPE &instance);
    double pu_for_merge(CobwebNode *best1, CobwebNode *best2, const AV_COUNT_TYPE &instance);
    double pu_for_split(CobwebNode *best);
    bool is_exact_match(const AV_COUNT_TYPE &instance);
    size_t _hash();
    std::string __str__();
    std::string concept_hash();
    std::string pretty_print(int depth = 0);
    int depth();
    bool is_parent(CobwebNode *otherConcept);
    int num_concepts();
    std::string avcounts_to_json();
    std::string ser_avcounts();
    std::string a_count_to_json();
    std::string sum_n_logn_to_json();
    std::string dump_json();
    std::string output_json();
    std::vector<std::tuple<VALUE_TYPE, double>>
    get_weighted_values(ATTR_TYPE attr, bool allowNone = true);
    std::unordered_map<int, std::unordered_map<int, double>> predict_probs();
    std::unordered_map<int, std::unordered_map<int, double>> predict_log_probs();
    std::unordered_map<int, std::unordered_map<int, double>> predict_weighted_probs(INSTANCE_TYPE instance);
    std::unordered_map<int, std::unordered_map<int, double>> predict_weighted_leaves_probs(INSTANCE_TYPE instance);
    VALUE_TYPE predict(ATTR_TYPE attr, std::string choiceFn = "most likely",
                       bool allowNone = true);
    double probability(ATTR_TYPE attr, VALUE_TYPE val);

    // NEW setter to safely replace av_count and recompute dependent fields
    void set_av_count(const AV_COUNT_TYPE &new_av_count);
};

class CobwebTree
{

public:
    float alpha;
    float log_alpha;
    bool weight_attr;
    int objective;
    bool children_norm;
    bool norm_attributes;
    CobwebNode *root;
    AV_KEY_TYPE attr_vals;

    CobwebTree(float alpha, bool weight_attr, int objective, bool children_norm, bool norm_attributes)
    {
        this->alpha = alpha;
        this->log_alpha = log(alpha);
        this->weight_attr = weight_attr;
        this->objective = objective;
        this->children_norm = children_norm;
        this->norm_attributes = norm_attributes;

        this->root = new CobwebNode();
        this->root->tree = this;
        this->attr_vals = AV_KEY_TYPE();
    }

    std::string __str__()
    {
        return this->root->__str__();
    }

    std::string dump_json()
    {
        std::string output = "{";

        output += "\"alpha\": " + doubleToString(this->alpha) + ",\n";
        output += "\"weight_attr\": " + std::to_string(this->weight_attr) + ",\n";
        output += "\"objective\": " + std::to_string(this->objective) + ",\n";
        output += "\"children_norm\": " + std::to_string(this->children_norm) + ",\n";
        output += "\"norm_attributes\": " + std::to_string(this->norm_attributes) + ",\n";
        output += "\"root\": " + this->root->dump_json();
        output += "}\n";

        return output;
    }

    class MyHandler : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, MyHandler>
    {
    public:
        CobwebTree *cobwebTree;
        std::stack<CobwebNode *> nodeStack;
        std::stack<int> keyStack; // Stack to keep track of the nested keys

        MyHandler(CobwebTree *tree) : cobwebTree(tree) {}

        bool Key(const char *str, rapidjson::SizeType length, bool copy)
        {
            std::string key(str, length);
            if (ATTRIBUTE_MAP.find(key) != ATTRIBUTE_MAP.end())
            {
                keyStack.push(ATTRIBUTE_MAP.at(key));
            }
            else
            {
                int key = std::stoi(std::string(str, length));
                keyStack.push(key);
            }
            return true;
        }

        bool Bool(bool b)
        {
            if (keyStack.top() == ATTRIBUTE_MAP.at("weight_attr"))
            {
                cobwebTree->weight_attr = b;
                keyStack.pop();
            }
            else if (keyStack.top() == ATTRIBUTE_MAP.at("children_norm"))
            {
                cobwebTree->children_norm = b;
                keyStack.pop();
            }
            else if (keyStack.top() == ATTRIBUTE_MAP.at("norm_attributes"))
            {
                cobwebTree->norm_attributes = b;
                keyStack.pop();
            }
            return true;
        }

        bool Uint(unsigned i)
        {
            if (keyStack.top() == ATTRIBUTE_MAP.at("objective"))
            {
                cobwebTree->objective = i;
                keyStack.pop();
            }
            else if (keyStack.top() == ATTRIBUTE_MAP.at("weight_attr"))
            {
                cobwebTree->weight_attr = i;
                keyStack.pop();
            }
            else if (keyStack.top() == ATTRIBUTE_MAP.at("children_norm"))
            {
                cobwebTree->children_norm = i;
                keyStack.pop();
            }
            else if (keyStack.top() == ATTRIBUTE_MAP.at("norm_attributes"))
            {
                cobwebTree->norm_attributes = i;
                keyStack.pop();
            }
            return true;
        }

        int pop_x(int x)
        {
            std::stack<int> tempStack = keyStack;
            if (x > (int)tempStack.size())
            {
                return -1;
            }
            for (int i = 0; i < x; i++)
            {
                tempStack.pop();
            }
            return tempStack.top();
        }

        bool Double(double d)
        {
            int currentKey = keyStack.top();
            if (currentKey == ATTRIBUTE_MAP.at("alpha"))
            {
                std::cout << "alpha: " << d << std::endl;
                std::cout << "currentKey: " << currentKey << std::endl;
                cobwebTree->alpha = d;
                keyStack.pop();
            }
            else if (currentKey == ATTRIBUTE_MAP.at("count"))
            {
                CobwebNode *currentNode = nodeStack.top();
                currentNode->count = d;
                keyStack.pop();
            }
            else if (pop_x(1) == ATTRIBUTE_MAP.at("a_count"))
            {
                CobwebNode *currentNode = nodeStack.top();
                currentNode->a_count[currentKey] = d;
                keyStack.pop();
            }
            else if (pop_x(1) == ATTRIBUTE_MAP.at("sum_n_logn"))
            {
                CobwebNode *currentNode = nodeStack.top();
                currentNode->sum_n_logn[currentKey] = d;
                keyStack.pop();
            }
            else if (pop_x(2) == ATTRIBUTE_MAP.at("av_count"))
            {
                CobwebNode *currentNode = nodeStack.top();
                currentNode->av_count[pop_x(1)][currentKey] = d;
                keyStack.pop();
            }
            else
            {
                keyStack.pop();
            }
            return true;
        }

        bool StartObject()
        {
            if (keyStack.empty())
            {
                return true;
            }
            if (keyStack.top() == ATTRIBUTE_MAP.at("root"))
            {
                CobwebNode *rootNode = new CobwebNode();
                rootNode->tree = cobwebTree;
                rootNode->parent = nullptr;

                nodeStack.push(rootNode);
                cobwebTree->root = rootNode;
            }
            else if (keyStack.top() == ATTRIBUTE_MAP.at("a_count") || keyStack.top() == ATTRIBUTE_MAP.at("sum_n_logn") || keyStack.top() == ATTRIBUTE_MAP.at("av_count"))
            {
                return true;
            }
            else if (keyStack.top() == ATTRIBUTE_MAP.at("children") && !nodeStack.empty())
            {
                CobwebNode *newNode = new CobwebNode();
                newNode->tree = cobwebTree;
                newNode->parent = nodeStack.top();

                nodeStack.top()->children.push_back(newNode);
                nodeStack.push(newNode);
            }
            return true;
        }

        bool EndObject(rapidjson::SizeType memberCount)
        {
            if (keyStack.empty() && nodeStack.empty())
            {
                return true;
            }
            else if (keyStack.top() == ATTRIBUTE_MAP.at("root"))
            {
                nodeStack.pop();
            }
            else if (keyStack.top() == ATTRIBUTE_MAP.at("children") && !nodeStack.empty())
            {
                nodeStack.pop();
            }

            if (keyStack.top() == ATTRIBUTE_MAP.at("children") && !nodeStack.empty())
            {
                return true;
            }
            else
            {
                keyStack.pop();
            }
            return true;
        }

        bool StartArray()
        {
            return true;
        }

        bool EndArray(rapidjson::SizeType elementCount)
        {
            keyStack.pop();
            return true;
        }

        void display_progress()
        {
            // Placeholder for progress display logic
        }
    };

    void write_json_node(rapidjson::Writer<rapidjson::OStreamWrapper> &writer, CobwebNode *node)
    {
        writer.StartObject();
        writer.Key("count");
        writer.Double(node->count);

        writer.Key("a_count");
        writer.StartObject();
        for (auto &[attr, count] : node->a_count)
        {
            writer.Key(std::to_string(attr).c_str());
            writer.Double(count);
        }
        writer.EndObject();

        writer.Key("sum_n_logn");
        writer.StartObject();
        for (auto &[attr, count] : node->sum_n_logn)
        {
            writer.Key(std::to_string(attr).c_str());
            writer.Double(count);
        }
        writer.EndObject();

        writer.Key("av_count");
        writer.StartObject();
        for (auto &[attr, val_map] : node->av_count)
        {
            writer.Key(std::to_string(attr).c_str());
            writer.StartObject();
            for (auto &[val, count] : val_map)
            {
                writer.Key(std::to_string(val).c_str());
                writer.Double(count);
            }
            writer.EndObject();
        }
        writer.EndObject();

        writer.Key("children");
        writer.StartArray();
        for (auto &child : node->children)
        {
            write_json_node(writer, child);
        }
        writer.EndArray();

        writer.EndObject();
    }

    void write_json_stream(const std::string &save_path)
    {
        std::ofstream ofs(save_path, std::ios::binary);
        rapidjson::OStreamWrapper osw(ofs);
        rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);

        writer.StartObject();
        writer.Key("alpha");
        writer.Double(this->alpha);

        writer.Key("weight_attr");
        writer.Uint(this->weight_attr);

        writer.Key("objective");
        writer.Uint(this->objective);

        writer.Key("children_norm");
        writer.Uint(this->children_norm);

        writer.Key("norm_attributes");
        writer.Uint(this->norm_attributes);

        writer.Key("root");
        write_json_node(writer, this->root);

        writer.EndObject();
    }

    void load_json_stream(std::string json_model_path)
    {
        FILE *fp = fopen(json_model_path.c_str(), "rb");
        if (!fp)
        {
            std::cerr << "Could not open file: " << json_model_path << std::endl;
            return;
        }

        const int bufferSize = 65536; // 256KB
        char buffer[bufferSize];
        rapidjson::FileReadStream is(fp, buffer, sizeof(buffer));
        rapidjson::Reader reader;
        MyHandler handler(this);

        if (!reader.Parse(is, handler))
        {
            std::cout << "Error while parsing JSON." << std::endl;
        }

        for (auto &[attr, val_map] : this->root->av_count)
        {
            for (auto &[val, cnt] : val_map)
            {
                this->attr_vals[attr].insert(val);
            }
        }

        fclose(fp);
    }

    void clear()
    {
        delete this->root;
        this->root = new CobwebNode();
        this->root->tree = this;
        this->attr_vals = AV_KEY_TYPE();
    }

    std::tuple<CobwebNode *, std::unordered_map<std::string, double>, std::vector<std::string>> ifit_helper(const INSTANCE_TYPE &instance, size_t mode, bool debug = false)
    {
        // AV_COUNT_TYPE cached_instance;
        // for (auto &[attr, val_map] : instance)
        // {
        //     for (auto &[val, cnt] : val_map)
        //     {
        //         cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
        //     }
        // }
        // return this->cobweb(cached_instance, mode);
        return this->cobweb(instance, mode, debug);
    }

    std::tuple<CobwebNode *, std::unordered_map<std::string, double>, std::vector<std::string>> ifit(INSTANCE_TYPE instance, size_t mode, bool debug = false)
    {
        return this->ifit_helper(instance, mode, debug);
    }

    void fit(std::vector<INSTANCE_TYPE> instances, size_t mode, int iterations = 1, bool randomizeFirst = true)
    {
        for (int i = 0; i < iterations; i++)
        {
            if (i == 0 && randomizeFirst)
            {
                shuffle(instances.begin(), instances.end(), std::default_random_engine());
            }
            for (auto &instance : instances)
            {
                this->ifit(instance, mode);
            }
            shuffle(instances.begin(), instances.end(), std::default_random_engine());
        }
    }

    int chooseRandomAction()
    {
        std::random_device rd;                            // Seed for the random number engine
        std::mt19937 gen(rd());                           // Mersenne Twister random number engine
        std::uniform_int_distribution<> dis(BEST, SPLIT); // Distribution range: BEST to SPLIT

        return dis(gen); // Generate a random action
    }

    std::tuple<CobwebNode *, std::unordered_map<std::string, double>, std::vector<std::string>>
    cobweb(const AV_COUNT_TYPE &instance, size_t mode, bool debug = false)
    {
        for (auto &[attr, val_map] : instance)
        {
            for (auto &[val, cnt] : val_map)
            {
                attr_vals[attr].insert(val);
            }
        }

        CobwebNode *current = root;

        std::unordered_map<std::string, double> operation_stats;
        double while_loop_count = 0;
        double while_loop_time = 0;
        double is_match_count = 0;
        double is_match_IC_time = 0;
        double fringe_split_count = 0;
        double fringe_split_time = 0;
        double fringe_split_IC_time = 0;
        double four_ops_count = 0;

        double TBC_count = 0;
        double TBC_time = 0;

        double GBO_count = 0;
        double GBO_time = 0;

        double BEST_count = 0;
        double BEST_IC_time = 0;

        double NEW_count = 0;
        double NEW_IC_time = 0;
        double NEW_time = 0;

        double MERGE_count = 0;
        double MERGE_IC_time = 0;
        double MERGE_UC_time = 0;
        double MERGE_time = 0;

        double SPLIT_count = 0;
        double SPLIT_time = 0;

        std::vector<std::string> debug_logs;

        auto start_while = std::chrono::high_resolution_clock::now();
        while (true)
        {
            while_loop_count += 1;
            if (current->children.empty() && (current->count == 0 || current->is_exact_match(instance)))
            {
                is_match_count += 1;

                if (debug)
                {
                    std::ostringstream ss;
                    ss << "{\"action\":\"NEW\",\"node\":\"" << current->concept_hash() << "\",\"parent\":\""
                       << (current->parent ? current->parent->concept_hash() : "null") << "\"}";
                    debug_logs.push_back(ss.str());
                }

                auto start = std::chrono::high_resolution_clock::now();
                current->increment_counts(instance);
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                is_match_IC_time += elapsed.count();
                break;
            }
            else if (current->children.empty())
            {
                fringe_split_count += 1;
                auto start_fs = std::chrono::high_resolution_clock::now();

                if (current->parent == nullptr)
                {
                    // Root-preserving: create a new internal node above root
                    CobwebNode *new_node = new CobwebNode();
                    new_node->tree = this;

                    // Set new_node as the root
                    new_node->children.push_back(current);
                    current->parent = new_node;
                    root = new_node;

                    // Log as NEW (debug)
                    if (debug)
                    {
                        std::ostringstream ss;
                        ss << "{\"action\":\"NEW\",\"node\":\"" << new_node->concept_hash() 
                        << "\",\"parent\":\"null\"}";
                        debug_logs.push_back(ss.str());
                    }

                    // Increment counts for the new node
                    auto start_ic_new_node = std::chrono::high_resolution_clock::now();
                    new_node->increment_counts(instance);
                    auto end_ic_new_node = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed_ic_new_node = end_ic_new_node - start_ic_new_node;
                    fringe_split_IC_time += elapsed_ic_new_node.count();

                    // Create a new leaf under new_node for this instance
                    CobwebNode *new_leaf = new CobwebNode();
                    new_leaf->tree = this;
                    new_leaf->parent = new_node;

                    auto start_leaf_ic = std::chrono::high_resolution_clock::now();
                    new_leaf->increment_counts(instance);
                    auto end_leaf_ic = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed_leaf_ic = end_leaf_ic - start_leaf_ic;
                    fringe_split_IC_time += elapsed_leaf_ic.count();

                    new_node->children.push_back(new_leaf);
                }
                else
                {
                    // Original fringe-split behavior for non-root
                    CobwebNode *new_node = new CobwebNode(current);
                    current->parent = new_node;
                    new_node->children.push_back(current);

                    if (new_node->parent == nullptr)
                    {
                        root = new_node;
                    }
                    else
                    {
                        auto &siblings = new_node->parent->children;
                        siblings.erase(std::remove(siblings.begin(), siblings.end(), current), siblings.end());
                        siblings.push_back(new_node);
                    }

                    if (debug)
                    {
                        std::ostringstream ss;
                        ss << "{\"action\":\"NEW\",\"node\":\"" << new_node->concept_hash() << "\",\"parent\":\""
                        << (new_node->parent ? new_node->parent->concept_hash() : "null") << "\"}";
                        debug_logs.push_back(ss.str());
                    }

                    auto start2 = std::chrono::high_resolution_clock::now();
                    new_node->increment_counts(instance);
                    auto end2 = std::chrono::high_resolution_clock::now();
                    fringe_split_IC_time += std::chrono::duration<double>(end2 - start2).count();

                    current = new CobwebNode();
                    current->parent = new_node;
                    current->tree = this;

                    auto start3 = std::chrono::high_resolution_clock::now();
                    current->increment_counts(instance);
                    auto end3 = std::chrono::high_resolution_clock::now();
                    fringe_split_IC_time += std::chrono::duration<double>(end3 - start3).count();

                    new_node->children.push_back(current);
                }

                auto end_fs = std::chrono::high_resolution_clock::now();
                fringe_split_time += std::chrono::duration<double>(end_fs - start_fs).count();
                break;

                // // std::cout << "fringe split" << std::endl;
                // fringe_split_count += 1;

                // auto start_fs = std::chrono::high_resolution_clock::now();
                // CobwebNode *new_node = new CobwebNode(current);
                // current->parent = new_node;
                // new_node->children.push_back(current);

                // if (new_node->parent == nullptr)
                // {
                //     root = new_node;
                // }
                // else
                // {
                //     new_node->parent->children.erase(remove(new_node->parent->children.begin(),
                //                                             new_node->parent->children.end(), current),
                //                                      new_node->parent->children.end());
                //     new_node->parent->children.push_back(new_node);
                // }

                // auto start2 = std::chrono::high_resolution_clock::now();
                // new_node->increment_counts(instance);
                // auto end2 = std::chrono::high_resolution_clock::now();
                // std::chrono::duration<double> elapsed2 = end2 - start2;
                // fringe_split_IC_time += elapsed2.count();

                // current = new CobwebNode();
                // current->parent = new_node;
                // current->tree = this;

                // auto start3 = std::chrono::high_resolution_clock::now();
                // current->increment_counts(instance);
                // auto end3 = std::chrono::high_resolution_clock::now();
                // std::chrono::duration<double> elapsed3 = end3 - start3;
                // fringe_split_IC_time += elapsed3.count();

                // new_node->children.push_back(current);
                // auto end_fs = std::chrono::high_resolution_clock::now();
                // std::chrono::duration<double> elapsed_fs = end_fs - start_fs;
                // fringe_split_time += elapsed_fs.count();
                // break;
            }
            else
            {
                four_ops_count += 1;

                auto start_tbc = std::chrono::high_resolution_clock::now();
                auto [best1_mi, best1, best2] = current->two_best_children(instance);
                auto end_tbc = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_tbc = end_tbc - start_tbc;
                TBC_time += elapsed_tbc.count();
                TBC_count += 1;

                size_t bestAction = 0;
                auto start_gbo = std::chrono::high_resolution_clock::now();
                if (mode == 0)
                {
                    auto [_, action] = current->get_best_operation(instance, best1, best2, best1_mi);
                    bestAction = action;
                }
                else if (mode == 1)
                {
                    bestAction = BEST;
                }
                else if (mode == 2)
                {
                    bestAction = 0;
                    if (best1 == nullptr)
                    {
                        std::cout << "best1 is null" << std::endl;
                        // randomly choose between 0 and 1
                        bestAction = rand() % 2;
                    }
                    else
                    {
                        if (best2 == nullptr)
                        {
                            bestAction = rand() % 3;
                            if (bestAction == 2)
                            {
                                bestAction = 3;
                            }
                        }
                        else
                        {
                            bestAction = rand() % 4;
                        }
                    }
                }
                else if (mode == 3)
                {
                    bestAction = BEST;
                    if (best1 != nullptr && best2 != nullptr)
                    {
                        double epsilon = 1e-2;
                        double p = ((double)rand() / (RAND_MAX));
                        if (p < epsilon)
                        {
                            auto [_, action] = current->get_best_operation(instance, best1, best2, best1_mi);
                            bestAction = action;
                        }
                    }
                }
                // else if (mode == BEST_NEW)
                // {
                //     // Only consider BEST vs NEW.
                //     if (best1 == nullptr)
                //     {
                //         bestAction = NEW;
                //     }
                //     else
                //     {
                //         double best_pu = current->pu_for_insert(best1, instance);
                //         double new_pu = current->pu_for_new_child(instance);
                //         if (best_pu > new_pu)
                //             bestAction = BEST;
                //         else if (new_pu > best_pu)
                //             bestAction = NEW;
                //         else
                //             bestAction = (rand() % 2 == 0) ? BEST : NEW;
                //     }
                // }
                else
                {
                    bestAction = 0;
                }
                auto end_gbo = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_gbo = end_gbo - start_gbo;
                GBO_time += elapsed_gbo.count();
                GBO_count += 1;

                if (bestAction == BEST)
                {
                    BEST_count += 1;
                    if (debug)
                    {
                        std::ostringstream ss;
                        ss << "{\"action\":\"BEST\",\"node\":\"" << (best1 ? best1->concept_hash() : std::string("null")) << "\"}";
                        debug_logs.push_back(ss.str());
                    }
                    auto start_best = std::chrono::high_resolution_clock::now();
                    current->increment_counts(instance);
                    auto end_best = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed_best = end_best - start_best;
                    BEST_IC_time += elapsed_best.count();

                    current = best1;
                }
                else if (bestAction == NEW)
                {
                    NEW_count += 1;
                    if (debug)
                    {
                        std::ostringstream ss;
                        ss << "{\"action\":\"NEW\",\"parent\":\"" << current->concept_hash() << "\"}";
                        debug_logs.push_back(ss.str());
                    }
                    auto start_new = std::chrono::high_resolution_clock::now();

                    auto start_ic1 = std::chrono::high_resolution_clock::now();
                    current->increment_counts(instance);
                    auto end_ic1 = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed_ic1 = end_ic1 - start_ic1;
                    NEW_IC_time += elapsed_ic1.count();

                    CobwebNode *new_child = new CobwebNode();
                    new_child->parent = current;
                    new_child->tree = this;

                    auto start_ic2 = std::chrono::high_resolution_clock::now();
                    new_child->increment_counts(instance);
                    auto end_ic2 = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed_ic2 = end_ic2 - start_ic2;
                    NEW_IC_time += elapsed_ic2.count();

                    current->children.push_back(new_child);
                    current = new_child;

                    bestAction = 0;
                    auto end_new = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed_new = end_new - start_new;
                    NEW_time += elapsed_new.count();
                    break;
                }
                else if (bestAction == MERGE)
                {
                    MERGE_count += 1;
                    if (debug)
                    {
                        std::ostringstream ss;
                        ss << "{\"action\":\"MERGE\",\"parent\":\"" << current->concept_hash()
                           << "\",\"children\":[\"" << (best1 ? best1->concept_hash() : std::string("null")) << "\",\""
                           << (best2 ? best2->concept_hash() : std::string("null")) << "\"]}";
                        debug_logs.push_back(ss.str());
                    }
                    auto start_merge = std::chrono::high_resolution_clock::now();

                    auto start_mic1 = std::chrono::high_resolution_clock::now();
                    current->increment_counts(instance);
                    auto end_mic1 = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed_mic1 = end_mic1 - start_mic1;
                    MERGE_IC_time += elapsed_mic1.count();

                    CobwebNode *new_child = new CobwebNode();
                    new_child->parent = current;
                    new_child->tree = this;

                    auto start_uc = std::chrono::high_resolution_clock::now();
                    new_child->update_counts_from_node(best1);
                    new_child->update_counts_from_node(best2);
                    auto end_uc = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed_uc = end_uc - start_uc;
                    MERGE_UC_time += elapsed_uc.count();

                    best1->parent = new_child;
                    best2->parent = new_child;
                    new_child->children.push_back(best1);
                    new_child->children.push_back(best2);
                    current->children.erase(remove(current->children.begin(),
                                                   current->children.end(), best1),
                                            current->children.end());
                    current->children.erase(remove(current->children.begin(),
                                                   current->children.end(), best2),
                                            current->children.end());
                    current->children.push_back(new_child);
                    current = new_child;

                    auto end_merge = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed_merge = end_merge - start_merge;
                    MERGE_time += elapsed_merge.count();
                }
                else if (bestAction == SPLIT)
                {
                    SPLIT_count += 1;
                    if (debug)
                    {
                        std::ostringstream ss;
                        ss << "{\"action\":\"SPLIT\",\"node\":\"" << (best1 ? best1->concept_hash() : std::string("null")) << "\",\"parent\":\"" << current->concept_hash() << "\"}";
                        debug_logs.push_back(ss.str());
                    }
                    auto start_split = std::chrono::high_resolution_clock::now();
                    current->children.erase(remove(current->children.begin(),
                                                   current->children.end(), best1),
                                            current->children.end());
                    for (auto &c : best1->children)
                    {
                        c->parent = current;
                        c->tree = this;
                        current->children.push_back(c);
                    }
                    delete best1;
                    auto end_split = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed_split = end_split - start_split;
                    SPLIT_time += elapsed_split.count();
                }
                else
                {
                    throw "Best action choice \"" + std::to_string(bestAction) +
                        "\" (best=0, new=1, merge=2, split=3) not a recognized option. This should be impossible...";
                }
            }
        }
        auto end_while = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_while = end_while - start_while;
        while_loop_time += elapsed_while.count();

        operation_stats["while_loop_count"] = while_loop_count;
        operation_stats["while_loop_time"] = while_loop_time;
        operation_stats["is_match_count"] = is_match_count;
        operation_stats["is_match_IC_time"] = is_match_IC_time;
        operation_stats["fringe_split_count"] = fringe_split_count;
        operation_stats["fringe_split_time"] = fringe_split_time;
        operation_stats["fringe_split_IC_time"] = fringe_split_IC_time;
        operation_stats["four_ops_count"] = four_ops_count;
        operation_stats["TBC_count"] = TBC_count;
        operation_stats["TBC_time"] = TBC_time;
        operation_stats["GBO_count"] = GBO_count;
        operation_stats["GBO_time"] = GBO_time;
        operation_stats["BEST_count"] = BEST_count;
        operation_stats["BEST_IC_time"] = BEST_IC_time;
        operation_stats["BEST_time"] = BEST_IC_time;
        operation_stats["NEW_count"] = NEW_count;
        operation_stats["NEW_IC_time"] = NEW_IC_time;
        operation_stats["NEW_time"] = NEW_time;
        operation_stats["MERGE_count"] = MERGE_count;
        operation_stats["MERGE_IC_time"] = MERGE_IC_time;
        operation_stats["MERGE_UC_time"] = MERGE_UC_time;
        operation_stats["MERGE_time"] = MERGE_time;
        operation_stats["SPLIT_count"] = SPLIT_count;
        operation_stats["SPLIT_time"] = SPLIT_time;

        return std::make_tuple(current, operation_stats, debug_logs);
    }

    

    CobwebNode *_cobweb_categorize(const AV_COUNT_TYPE &instance)
    {
        
        auto current = this->root;

        while (true)
        {
            if (current->children.empty())
            {
                return current;
            }

            auto parent = current;
            current = nullptr;
            double best_logp;

            for (auto &child : parent->children)
            {
                double logp = child->log_prob_class_given_instance(instance, false);
                if (current == nullptr || logp > best_logp)
                {
                    best_logp = logp;
                    current = child;
                }
            }
        }
    }

    CobwebNode *categorize_helper(const INSTANCE_TYPE &instance)
    {
        return this->_cobweb_categorize(instance);
    }

    CobwebNode *categorize(const INSTANCE_TYPE instance)
    {
        return this->categorize_helper(instance);
    }

    std::tuple<std::unordered_map<std::string, double>, std::unordered_map<int, std::unordered_map<int, double>>> predict_probs_mixture_helper(const AV_COUNT_TYPE &instance, double ll_path, int max_nodes, bool greedy, bool missing)
    {

        std::unordered_map<int, std::unordered_map<int, double>> out;
        for (auto &[attr, val_set] : this->attr_vals)
        {
            for (auto &val : val_set)
            {
                out[attr][val] = -INFINITY;
            }
        }
        std::unordered_map<int, std::unordered_map<int, std::vector<double>>> weighted_pred_probs;

        std::unordered_map<std::string, double> operation_stats;

        int nodes_expanded = 0;
        double total_weight = 0;
        bool first_weight = true;

        double log_prob_missing_time = 0;
        double log_prob_instance_time = 0;
        double while_count = 0;
        double while_time = 0;

        double predict_prob_log_time = 0;

        double lse_loop_time = 0;
        double log_prob_children_time = 0;
        double log_prob_loop_time = 0;

        double exp_subtract_time = 0;

        double root_ll_inst = 0;
        if (missing)
        {
            auto log_prob_missing_time_start = std::chrono::high_resolution_clock::now();
            root_ll_inst = this->root->log_prob_instance_missing(instance);
            auto log_prob_missing_time_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_log_prob_missing_time = log_prob_missing_time_end - log_prob_missing_time_start;
            log_prob_missing_time += elapsed_log_prob_missing_time.count();
        }
        else
        {
            auto log_prob_instance_time_start = std::chrono::high_resolution_clock::now();
            root_ll_inst = this->root->log_prob_instance(instance);
            auto log_prob_instance_time_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_log_prob_instance_time = log_prob_instance_time_end - log_prob_instance_time_start;
            log_prob_instance_time += elapsed_log_prob_instance_time.count();
        }

        auto queue = std::priority_queue<
            std::tuple<double, double, CobwebNode *>>();

        queue.push(std::make_tuple(root_ll_inst, 0.0, this->root));

        auto start_while = std::chrono::high_resolution_clock::now();
        while (queue.size() > 0)
        {
            while_count += 1;

            auto node = queue.top();
            queue.pop();
            nodes_expanded += 1;

            if (greedy)
            {
                queue = std::priority_queue<
                    std::tuple<double, double, CobwebNode *>>();
            }

            auto curr_score = std::get<0>(node);
            auto curr_ll = std::get<1>(node);
            auto curr = std::get<2>(node);

            if (first_weight)
            {
                total_weight = curr_score;
                first_weight = false;
            }
            else
            {
                total_weight = logsumexp(total_weight, curr_score);
            }

            auto start_predict_prob_log_time = std::chrono::high_resolution_clock::now();
            auto curr_log_probs = curr->predict_log_probs();
            auto end_predict_prob_log_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_predict_prob_log_time = end_predict_prob_log_time - start_predict_prob_log_time;
            predict_prob_log_time += elapsed_predict_prob_log_time.count();

            auto start_lse_loop = std::chrono::high_resolution_clock::now();
            for (auto &[attr, val_set] : curr_log_probs)
            {
                for (auto &[val, log_p] : val_set)
                {
                    out[attr][val] = eff_logsumexp(out[attr][val], curr_score + log_p);
                }
            }
            auto end_lse_loop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_lse_loop = end_lse_loop - start_lse_loop;
            lse_loop_time += elapsed_lse_loop.count();

            if (nodes_expanded >= max_nodes)
                break;

            auto start_log_prob_children = std::chrono::high_resolution_clock::now();
            std::vector<double> log_children_probs = curr->log_prob_children_given_instance(instance);
            auto end_log_prob_children = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_log_prob_children = end_log_prob_children - start_log_prob_children;
            log_prob_children_time += elapsed_log_prob_children.count();

            auto start_log_prob_loop = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < curr->children.size(); ++i)
            {
                auto child = curr->children[i];
                double child_ll_inst = 0;
                if (missing)
                {
                    child_ll_inst = child->log_prob_instance_missing(instance);
                }
                else
                {
                    child_ll_inst = child->log_prob_instance(instance);
                }
                auto child_ll_given_parent = log_children_probs[i];
                auto child_ll = child_ll_given_parent + curr_ll;
                queue.push(std::make_tuple(child_ll_inst + child_ll, child_ll, child));
            }
            auto end_log_prob_loop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_log_prob_loop = end_log_prob_loop - start_log_prob_loop;
            log_prob_loop_time += elapsed_log_prob_loop.count();
        }
        auto end_while = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_while = end_while - start_while;
        while_time += elapsed_while.count();

        auto start_exp_subtract = std::chrono::high_resolution_clock::now();
        for (auto &[attr, val_set] : out)
        {
            for (auto &[val, p] : val_set)
            {
                out[attr][val] = exp(out[attr][val] - total_weight);
            }
        }
        auto end_exp_subtract = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_exp_subtract = end_exp_subtract - start_exp_subtract;
        exp_subtract_time += elapsed_exp_subtract.count();

        operation_stats["log_prob_missing_time"] = log_prob_missing_time;
        operation_stats["log_prob_instance_time"] = log_prob_instance_time;
        operation_stats["while_count"] = while_count;
        operation_stats["while_time"] = while_time;
        operation_stats["predict_prob_log_time"] = predict_prob_log_time;
        operation_stats["lse_loop_time"] = lse_loop_time;
        operation_stats["log_prob_children_time"] = log_prob_children_time;
        operation_stats["log_prob_loop_time"] = log_prob_loop_time;
        operation_stats["exp_subtract_time"] = exp_subtract_time;

        return std::make_tuple(operation_stats, out);
    }

    std::tuple<std::unordered_map<std::string, double>, std::unordered_map<int, std::unordered_map<int, double>>> predict_probs_mixture(INSTANCE_TYPE instance, int max_nodes, bool greedy, bool missing)
    {
        return this->predict_probs_mixture_helper(instance, 0.0,
                                                  max_nodes, greedy, missing);
    }

    std::vector<std::unordered_map<int, std::unordered_map<int, double>>> predict_probs_mixture_parallel(std::vector<INSTANCE_TYPE> instances, int max_nodes, bool greedy, bool missing, int num_threads)
    {

        BS::thread_pool pool = BS::thread_pool(num_threads);

        std::vector<std::unordered_map<int, std::unordered_map<int, double>>> out(instances.size());

        auto start = std::chrono::high_resolution_clock::now();

        pool.detach_sequence<unsigned int>(0, instances.size(),
                                           [this, &instances, &out, max_nodes, greedy, missing](const unsigned int i)
                                           {
                                               auto [operation_stats, probs] = this->predict_probs_mixture(instances[i], max_nodes, greedy, missing);
                                               out[i] = probs;
                                           });

        while (true)
        {
            if (!pool.wait_for(std::chrono::milliseconds(1000)))
            {
                double progress = (instances.size() - pool.get_tasks_total()) / double(instances.size());
                auto current = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> elapsed = current - start;
                displayProgressBar(70, progress, elapsed.count() / 1000.0);
            }
            else
            {
                break;
            }
        }

        pool.wait();

        return out;
    }
};

// ------------------------- CobwebNode implementations -------------------------

inline CobwebNode::CobwebNode()
{
    count = 0;
    sum_n_logn = ATTR_COUNT_TYPE();
    a_count = ATTR_COUNT_TYPE();
    parent = nullptr;
    tree = nullptr;
}

inline CobwebNode::CobwebNode(CobwebNode *otherNode)
{
    count = 0;
    sum_n_logn = ATTR_COUNT_TYPE();
    a_count = ATTR_COUNT_TYPE();

    parent = otherNode->parent;
    tree = otherNode->tree;

    update_counts_from_node(otherNode);

    for (auto child : otherNode->children)
    {
        children.push_back(new CobwebNode(child));
    }
}

inline void CobwebNode::increment_counts(const AV_COUNT_TYPE &instance)
{
    this->count += 1;
    for (auto &[attr, val_map] : instance)
    {
        for (auto &[val, cnt] : val_map)
        {
            this->a_count[attr] += cnt;

            if (attr > 0)
            {
                if (this->av_count.count(attr) && this->av_count.at(attr).count(val))
                {
                    double tf = this->av_count.at(attr).at(val) + this->tree->alpha;
                    this->sum_n_logn[attr] -= tf * log(tf);
                }
            }

            this->av_count[attr][val] += cnt;

            if (attr > 0)
            {
                double tf = this->av_count.at(attr).at(val) + this->tree->alpha;
                this->sum_n_logn[attr] += tf * log(tf);
            }
        }
    }
}

inline void CobwebNode::update_counts_from_node(CobwebNode *node)
{
    this->count += node->count;

    for (auto &[attr, val_map] : node->av_count)
    {
        this->a_count[attr] += node->a_count.at(attr);

        for (auto &[val, cnt] : val_map)
        {
            if (attr > 0)
            {
                if (this->av_count.count(attr) && this->av_count.at(attr).count(val))
                {
                    double tf = this->av_count.at(attr).at(val) + this->tree->alpha;
                    this->sum_n_logn[attr] -= tf * log(tf);
                }
            }

            this->av_count[attr][val] += cnt;

            if (attr > 0)
            {
                double tf = this->av_count.at(attr).at(val) + this->tree->alpha;
                this->sum_n_logn[attr] += tf * log(tf);
            }
        }
    }
}

inline double CobwebNode::entropy_attr_insert(ATTR_TYPE attr, const AV_COUNT_TYPE &instance)
{
    if (attr < 0)
        return 0.0;

    float alpha = this->tree->alpha;
    int num_vals_total = this->tree->attr_vals.at(attr).size();
    int num_vals_in_c = 0;
    COUNT_TYPE attr_count = 0;

    double ratio = 1.0;
    if (this->tree->weight_attr)
    {
        ratio = (1.0 * this->tree->root->a_count.at(attr)) / (this->tree->root->count);
    }

    if (this->av_count.count(attr))
    {
        attr_count = this->a_count.at(attr);
        num_vals_in_c = this->av_count.at(attr).size();
    }

    double sum_n_logn = 0.0;
    if (this->sum_n_logn.count(attr))
    {
        sum_n_logn = this->sum_n_logn.at(attr);
    }

    if (instance.count(attr))
    {
        for (auto &[val, cnt] : instance.at(attr))
        {
            attr_count += cnt;
            COUNT_TYPE prior_av_count = 0.0;
            if (this->av_count.count(attr) && this->av_count.at(attr).count(val))
            {
                prior_av_count = this->av_count.at(attr).at(val);
                COUNT_TYPE tf = prior_av_count + this->tree->alpha;
                sum_n_logn -= tf * log(tf);
            }
            else
            {
                num_vals_in_c += 1;
            }
            COUNT_TYPE tf = prior_av_count + cnt + this->tree->alpha;
            sum_n_logn += (tf)*log(tf);
        }
    }

    int n0 = num_vals_total - num_vals_in_c;
    double info = -ratio * ((1 / (attr_count + num_vals_total * alpha)) *
                                (sum_n_logn + n0 * alpha * log(alpha)) -
                            log(attr_count +
                                num_vals_total * alpha));
    return info;
}

inline double CobwebNode::entropy_insert(const AV_COUNT_TYPE &instance)
{
    double info = 0.0;

    for (auto &[attr, av_inner] : this->av_count)
    {
        if (attr < 0)
            continue;
        info += this->entropy_attr_insert(attr, instance);
    }

    for (auto &[attr, av_inner] : instance)
    {
        if (attr < 0)
            continue;
        if (this->av_count.count(attr))
            continue;
        info += this->entropy_attr_insert(attr, instance);
    }

    return info;
}

inline double CobwebNode::entropy_attr_merge(ATTR_TYPE attr,
                                             CobwebNode *other, const AV_COUNT_TYPE &instance)
{
    if (attr < 0)
        return 0.0;

    float alpha = this->tree->alpha;
    int num_vals_total = this->tree->attr_vals.at(attr).size();
    int num_vals_in_c = 0;
    COUNT_TYPE attr_count = 0;

    double ratio = 1.0;
    if (this->tree->weight_attr)
    {
        ratio = (1.0 * this->tree->root->a_count.at(attr)) / (this->tree->root->count);
    }

    if (this->av_count.count(attr))
    {
        attr_count = this->a_count.at(attr);
        num_vals_in_c = this->av_count.at(attr).size();
    }

    double sum_n_logn = 0.0;
    if (this->sum_n_logn.count(attr))
    {
        sum_n_logn = this->sum_n_logn.at(attr);
    }

    if (other->av_count.count(attr))
    {
        for (auto &[val, other_av_count] : other->av_count.at(attr))
        {
            COUNT_TYPE instance_av_count = 0.0;

            if (instance.count(attr) && instance.at(attr).count(val))
            {
                instance_av_count = instance.at(attr).at(val);
            }

            attr_count += other_av_count + instance_av_count;
            COUNT_TYPE prior_av_count = 0.0;
            if (this->av_count.count(attr) && this->av_count.at(attr).count(val))
            {
                prior_av_count = this->av_count.at(attr).at(val);
                COUNT_TYPE tf = prior_av_count + alpha;
                sum_n_logn -= tf * log(tf);
            }
            else
            {
                num_vals_in_c += 1;
            }

            COUNT_TYPE new_tf = prior_av_count + other_av_count + instance_av_count + alpha;
            sum_n_logn += (new_tf)*log(new_tf);
        }
    }

    if (instance.count(attr))
    {
        for (auto &[val, instance_av_count] : instance.at(attr))
        {
            if (other->av_count.count(attr) && other->av_count.at(attr).count(val))
            {
                continue;
            }
            COUNT_TYPE other_av_count = 0.0;

            attr_count += other_av_count + instance_av_count;
            COUNT_TYPE prior_av_count = 0.0;
            if (this->av_count.count(attr) && this->av_count.at(attr).count(val))
            {
                prior_av_count = this->av_count.at(attr).at(val);
                COUNT_TYPE tf = prior_av_count + alpha;
                sum_n_logn -= tf * log(tf);
            }
            else
            {
                num_vals_in_c += 1;
            }

            COUNT_TYPE new_tf = prior_av_count + other_av_count + instance_av_count + alpha;
            sum_n_logn += (new_tf)*log(new_tf);
        }
    }

    int n0 = num_vals_total - num_vals_in_c;
    double info = -ratio * ((1 / (attr_count + num_vals_total * alpha)) *
                                (sum_n_logn + n0 * alpha * log(alpha)) -
                            log(attr_count +
                                num_vals_total * alpha));
    return info;
}

inline double CobwebNode::entropy_merge(CobwebNode *other,
                                        const AV_COUNT_TYPE &instance)
{
    double info = 0.0;

    for (auto &[attr, inner_vals] : this->av_count)
    {
        if (attr < 0)
            continue;
        info += this->entropy_attr_merge(attr, other, instance);
    }

    for (auto &[attr, inner_vals] : other->av_count)
    {
        if (attr < 0)
            continue;
        if (this->av_count.count(attr))
            continue;
        info += this->entropy_attr_merge(attr, other, instance);
    }

    for (auto &[attr, inner_vals] : instance)
    {
        if (attr < 0)
            continue;
        if (this->av_count.count(attr))
            continue;
        if (other->av_count.count(attr))
            continue;
        info += entropy_attr_merge(attr, other, instance);
    }

    return info;
}

inline CobwebNode *CobwebNode::get_best_level(
    INSTANCE_TYPE instance)
{
    CobwebNode *curr = this;
    CobwebNode *best = this;
    double best_ll = this->log_prob_class_given_instance(instance, true);

    while (curr->parent != nullptr)
    {
        curr = curr->parent;
        double curr_ll = curr->log_prob_class_given_instance(instance, true);

        if (curr_ll > best_ll)
        {
            best = curr;
            best_ll = curr_ll;
        }
    }

    return best;
}

inline CobwebNode *CobwebNode::get_basic_level()
{
    CobwebNode *curr = this;
    CobwebNode *best = this;
    double best_cu = this->category_utility();

    while (curr->parent != nullptr)
    {
        curr = curr->parent;
        double curr_cu = curr->category_utility();

        if (curr_cu > best_cu)
        {
            best = curr;
            best_cu = curr_cu;
        }
    }

    return best;
}

inline double CobwebNode::entropy_attr(ATTR_TYPE attr)
{
    if (attr < 0)
        return 0.0;

    float alpha = this->tree->alpha;
    int num_vals_total = this->tree->attr_vals.at(attr).size();
    int num_vals_in_c = 0;
    COUNT_TYPE attr_count = 0;

    if (this->av_count.count(attr))
    {
        attr_count = this->a_count.at(attr);
        num_vals_in_c = this->av_count.at(attr).size();
    }

    double ratio = 1.0;
    if (this->tree->weight_attr)
    {
        ratio = (1.0 * this->tree->root->a_count.at(attr)) / (this->tree->root->count);
    }

    double sum_n_logn = 0.0;
    if (this->sum_n_logn.count(attr))
    {
        sum_n_logn = this->sum_n_logn.at(attr);
    }

    int n0 = num_vals_total - num_vals_in_c;
    double info = -ratio * ((1 / (attr_count + num_vals_total * alpha)) *
                                (sum_n_logn + n0 * alpha * log(alpha)) -
                            log(attr_count +
                                num_vals_total * alpha));
    return info;
}

inline double CobwebNode::entropy()
{
    double info = 0.0;
    for (auto &[attr, inner_av] : this->av_count)
    {
        if (attr < 0)
            continue;
        info += this->entropy_attr(attr);
    }

    return info;
}

inline std::tuple<double, int> CobwebNode::get_best_operation(
    const AV_COUNT_TYPE &instance, CobwebNode *best1,
    CobwebNode *best2, double best1_pu)
{

    if (best1 == nullptr)
    {
        throw "Need at least one best child.";
    }
    std::vector<std::tuple<double, double, int>> operations;
    operations.push_back(std::make_tuple(best1_pu,
                                         custom_rand(),
                                         BEST));
    operations.push_back(std::make_tuple(pu_for_new_child(instance),
                                         custom_rand(),
                                         NEW));
    if (children.size() > 2 && best2 != nullptr)
    {
        operations.push_back(std::make_tuple(pu_for_merge(best1, best2,
                                                          instance),
                                             custom_rand(),
                                             MERGE));
    }

    if (best1->children.size() > 0)
    {
        operations.push_back(std::make_tuple(pu_for_split(best1),
                                             custom_rand(),
                                             SPLIT));
    }

    sort(operations.rbegin(), operations.rend());

    OPERATION_TYPE bestOp = std::make_pair(std::get<0>(operations[0]), std::get<2>(operations[0]));
    return bestOp;
}

inline std::tuple<double, CobwebNode *, CobwebNode *> CobwebNode::two_best_children(
    const AV_COUNT_TYPE &instance)
{

    if (children.empty())
    {
        throw "No children!";
    }

    if (this->tree->objective == 0)
    {
        std::vector<std::tuple<double, double, double, CobwebNode *>> relative_pu;
        for (auto &child : this->children)
        {
            relative_pu.push_back(
                std::make_tuple(
                    (child->count * child->entropy()) -
                        ((child->count + 1) * child->entropy_insert(instance)),
                    child->count,
                    custom_rand(),
                    child));
        }

        sort(relative_pu.rbegin(), relative_pu.rend());
        CobwebNode *best1 = std::get<3>(relative_pu[0]);
        double best1_pu = 0.0;
        CobwebNode *best2 = relative_pu.size() > 1 ? std::get<3>(relative_pu[1]) : nullptr;
        return std::make_tuple(best1_pu, best1, best2);
    }
    else
    {
        std::vector<std::tuple<double, double, double, CobwebNode *>> pus;
        for (auto &child : this->children)
        {
            pus.push_back(
                std::make_tuple(
                    pu_for_insert(child, instance),
                    child->count,
                    custom_rand(),
                    child));
        }
        sort(pus.rbegin(), pus.rend());
        CobwebNode *best1 = std::get<3>(pus[0]);
        double best1_pu = 0.0;
        CobwebNode *best2 = pus.size() > 1 ? std::get<3>(pus[1]) : nullptr;

        return std::make_tuple(best1_pu, best1, best2);
    }
}

inline double CobwebNode::partition_utility()
{
    if (children.empty())
    {
        return 0.0;
    }

    if (!this->tree->norm_attributes)
    {
        double parent_entropy = 0.0;
        double children_entropy = 0.0;
        double concept_entropy = 0.0;

        for (auto &[attr, val_set] : this->tree->attr_vals)
        {
            parent_entropy += this->entropy_attr(attr);
        }

        for (auto &child : children)
        {
            double p_of_child = (1.0 * child->count) / this->count;
            concept_entropy -= p_of_child * log(p_of_child);

            for (auto &[attr, val_set] : this->tree->attr_vals)
            {
                children_entropy += p_of_child * child->entropy_attr(attr);
            }
        }

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1)
        {
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2)
        {
            obj /= (children_entropy + concept_entropy);
        }
        if (this->tree->children_norm)
        {
            obj /= this->children.size();
        }
        return obj;
    }

    double entropy = 0.0;

    for (auto &[attr, val_set] : this->tree->attr_vals)
    {
        double children_entropy = 0.0;
        double concept_entropy = 0.0;

        for (auto &child : children)
        {
            double p_of_child = (1.0 * child->count) / this->count;
            children_entropy += p_of_child * child->entropy_attr(attr);
            concept_entropy -= p_of_child * log(p_of_child);
        }

        double parent_entropy = this->entropy_attr(attr);

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1)
        {
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2)
        {
            obj /= (children_entropy + concept_entropy);
        }

        if (this->tree->children_norm)
        {
            obj /= this->children.size();
        }
        entropy += obj;
    }

    return entropy;
}


inline double CobwebNode::pu_for_insert(CobwebNode *child, const AV_COUNT_TYPE &instance)
{

    // BEGIN INDIVIDUAL
    if (!this->tree->norm_attributes)
    {
        double parent_entropy = 0.0;
        double children_entropy = 0.0;
        double concept_entropy = 0.0;

        for (auto &[attr, val_set] : this->tree->attr_vals)
        {
            parent_entropy += this->entropy_attr_insert(attr, instance);
        }

        for (auto &c : children)
        {
            if (c == child)
            {
                double p_of_child = (c->count + 1.0) / (this->count + 1.0);
                concept_entropy -= p_of_child * log(p_of_child);
                for (auto &[attr, val_set] : this->tree->attr_vals)
                {
                    children_entropy += p_of_child * c->entropy_attr_insert(attr, instance);
                }
            }
            else
            {
                double p_of_child = (1.0 * c->count) / (this->count + 1.0);
                concept_entropy -= p_of_child * log(p_of_child);

                for (auto &[attr, val_set] : this->tree->attr_vals)
                {
                    children_entropy += p_of_child * c->entropy_attr(attr);
                }
            }
        }

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1)
        {
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2)
        {
            obj /= (children_entropy + concept_entropy);
        }
        if (this->tree->children_norm)
        {
            obj /= this->children.size();
        }
        return obj;
    }
    // END INDIVIDUAL

    double entropy = 0.0;

    for (auto &[attr, val_set] : this->tree->attr_vals)
    {
        double children_entropy = 0.0;
        double concept_entropy = 0.0;

        for (auto &c : this->children)
        {
            if (c == child)
            {
                double p_of_child = (c->count + 1.0) / (this->count + 1.0);
                children_entropy += p_of_child * c->entropy_attr_insert(attr, instance);
                concept_entropy -= p_of_child * log(p_of_child);
            }
            else
            {
                double p_of_child = (1.0 * c->count) / (this->count + 1.0);
                children_entropy += p_of_child * c->entropy_attr(attr);
                concept_entropy -= p_of_child * log(p_of_child);
            }
        }

        double parent_entropy = this->entropy_attr_insert(attr, instance);

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1)
        {
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2)
        {
            obj /= (children_entropy + concept_entropy);
        }

        if (this->tree->children_norm)
        {
            obj /= this->children.size();
        }
        entropy += obj;

        // entropy += (parent_entropy - children_entropy) / this->children.size();
        // entropy += (parent_entropy - children_entropy) / parent_entropy / this->children.size();
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy) / this->children.size();
        // entropy += (parent_entropy - children_entropy) / parent_entropy;
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy);
    }

    return entropy;
}

inline double CobwebNode::pu_for_new_child(const AV_COUNT_TYPE &instance)
{

    // TODO maybe modify so that we can evaluate new child without copying
    // instance.
    CobwebNode new_child = CobwebNode();
    new_child.parent = this;
    new_child.tree = this->tree;
    new_child.increment_counts(instance);
    double p_of_new_child = 1.0 / (this->count + 1.0);

    // BEGIN INDIVIDUAL
    if (!this->tree->norm_attributes)
    {
        double parent_entropy = 0.0;
        double children_entropy = 0.0;
        double concept_entropy = -p_of_new_child * log(p_of_new_child);

        for (auto &[attr, val_set] : this->tree->attr_vals)
        {
            children_entropy += p_of_new_child * new_child.entropy_attr(attr);
            parent_entropy += this->entropy_attr_insert(attr, instance);
        }

        for (auto &child : children)
        {
            double p_of_child = (1.0 * child->count) / (this->count + 1.0);
            concept_entropy -= p_of_child * log(p_of_child);

            for (auto &[attr, val_set] : this->tree->attr_vals)
            {
                children_entropy += p_of_child * child->entropy_attr(attr);
            }
        }

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1)
        {
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2)
        {
            obj /= (children_entropy + concept_entropy);
        }
        if (this->tree->children_norm)
        {
            obj /= (this->children.size() + 1);
        }
        return obj;
    }
    // END INDIVIDUAL

    double entropy = 0.0;

    for (auto &[attr, val_set] : this->tree->attr_vals)
    {
        double children_entropy = p_of_new_child * new_child.entropy_attr(attr);
        double concept_entropy = -p_of_new_child * log(p_of_new_child);

        for (auto &c : this->children)
        {
            double p_of_child = (1.0 * c->count) / (this->count + 1.0);
            children_entropy += p_of_child * c->entropy_attr(attr);
            concept_entropy -= p_of_child * log(p_of_child);
        }

        double parent_entropy = this->entropy_attr_insert(attr, instance);

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1)
        {
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2)
        {
            obj /= (children_entropy + concept_entropy);
        }

        if (this->tree->children_norm)
        {
            obj /= (this->children.size() + 1);
        }
        entropy += obj;

        // entropy += (parent_entropy - children_entropy) / (this->children.size() + 1);
        // entropy += (parent_entropy - children_entropy) / parent_entropy / (this->children.size() + 1);
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy) / (this->children.size() + 1);
        // entropy += (parent_entropy - children_entropy) / parent_entropy;
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy);
    }

    return entropy;
}

inline double CobwebNode::pu_for_merge(CobwebNode *best1,
                                       CobwebNode *best2, const AV_COUNT_TYPE &instance)
{

    // BEGIN INDIVIDUAL
    if (!this->tree->norm_attributes)
    {
        double parent_entropy = 0.0;
        double children_entropy = 0.0;
        double p_of_merged = (best1->count + best2->count + 1.0) / (this->count + 1.0);
        double concept_entropy = -p_of_merged * log(p_of_merged);

        for (auto &[attr, val_set] : this->tree->attr_vals)
        {
            parent_entropy += this->entropy_attr_insert(attr, instance);
            children_entropy += p_of_merged * best1->entropy_attr_merge(attr, best2, instance);
        }

        for (auto &child : children)
        {
            if (child == best1 || child == best2)
            {
                continue;
            }
            double p_of_child = (1.0 * child->count) / (this->count + 1.0);
            concept_entropy -= p_of_child * log(p_of_child);

            for (auto &[attr, val_set] : this->tree->attr_vals)
            {
                children_entropy += p_of_child * child->entropy_attr(attr);
            }
        }

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1)
        {
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2)
        {
            obj /= (children_entropy + concept_entropy);
        }
        if (this->tree->children_norm)
        {
            obj /= (this->children.size() - 1);
        }
        return obj;
    }
    // END INDIVIDUAL

    double entropy = 0.0;

    for (auto &[attr, val_set] : this->tree->attr_vals)
    {
        double children_entropy = 0.0;
        double concept_entropy = 0.0;

        for (auto &c : children)
        {
            if (c == best1 || c == best2)
            {
                continue;
            }

            double p_of_child = (1.0 * c->count) / (this->count + 1.0);
            children_entropy += p_of_child * c->entropy_attr(attr);
            concept_entropy -= p_of_child * log(p_of_child);
        }

        double p_of_child = (best1->count + best2->count + 1.0) / (this->count + 1.0);
        children_entropy += p_of_child * best1->entropy_attr_merge(attr, best2, instance);
        concept_entropy -= p_of_child * log(p_of_child);

        double parent_entropy = this->entropy_attr_insert(attr, instance);

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1)
        {
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2)
        {
            obj /= (children_entropy + concept_entropy);
        }

        if (this->tree->children_norm)
        {
            obj /= (this->children.size() - 1);
        }
        entropy += obj;

        // entropy += (parent_entropy - children_entropy) / (this->children.size() - 1);
        // entropy += (parent_entropy - children_entropy) / parent_entropy / (this->children.size() - 1);
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy) / (this->children.size() - 1);
        // entropy += (parent_entropy - children_entropy) / parent_entropy;
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy);
    }

    return entropy;
}

inline double CobwebNode::pu_for_split(CobwebNode *best)
{

    // BEGIN INDIVIDUAL
    if (!this->tree->norm_attributes)
    {
        double parent_entropy = 0.0;
        double children_entropy = 0.0;
        double concept_entropy = 0.0;

        for (auto &[attr, val_set] : this->tree->attr_vals)
        {
            parent_entropy += this->entropy_attr(attr);
        }

        for (auto &child : children)
        {
            if (child == best)
                continue;
            double p_of_child = (1.0 * child->count) / this->count;
            concept_entropy -= p_of_child * log(p_of_child);
            for (auto &[attr, val_set] : this->tree->attr_vals)
            {
                children_entropy += p_of_child * child->entropy_attr(attr);
            }
        }

        for (auto &child : best->children)
        {
            double p_of_child = (1.0 * child->count) / this->count;
            concept_entropy -= p_of_child * log(p_of_child);
            for (auto &[attr, val_set] : this->tree->attr_vals)
            {
                children_entropy += p_of_child * child->entropy_attr(attr);
            }
        }

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1)
        {
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2)
        {
            obj /= (children_entropy + concept_entropy);
        }
        if (this->tree->children_norm)
        {
            obj /= (this->children.size() - 1 + best->children.size());
        }
        return obj;
    }
    // END INDIVIDUAL

    double entropy = 0.0;

    for (auto &[attr, val_set] : this->tree->attr_vals)
    {

        double children_entropy = 0.0;
        double concept_entropy = 0.0;

        for (auto &c : children)
        {
            if (c == best)
                continue;
            double p_of_child = (1.0 * c->count) / this->count;
            children_entropy += p_of_child * c->entropy_attr(attr);
            concept_entropy -= p_of_child * log(p_of_child);
        }

        for (auto &c : best->children)
        {
            double p_of_child = (1.0 * c->count) / this->count;
            children_entropy += p_of_child * c->entropy_attr(attr);
            concept_entropy -= p_of_child * log(p_of_child);
        }

        double parent_entropy = this->entropy_attr(attr);

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1)
        {
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2)
        {
            obj /= (children_entropy + concept_entropy);
        }

        if (this->tree->children_norm)
        {
            obj /= (this->children.size() - 1 + best->children.size());
        }
        entropy += obj;

        // entropy += (parent_entropy - children_entropy) / (this->children.size() - 1 + best->children.size());
        // entropy += (parent_entropy - children_entropy) / parent_entropy / (this->children.size() - 1 + best->children.size());
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy) / (this->children.size() - 1 + best->children.size());
        // entropy += (parent_entropy - children_entropy) / parent_entropy;
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy);
    }

    return entropy;
}

inline bool CobwebNode::is_exact_match(const AV_COUNT_TYPE &instance)
{
    std::unordered_set<ATTR_TYPE> all_attrs;
    for (auto &[attr, tmp] : instance)
        all_attrs.insert(attr);
    for (auto &[attr, tmp] : this->av_count)
        all_attrs.insert(attr);

    for (auto &attr : all_attrs)
    {
        // if (attr.is_hidden())
        //     continue;
        if (attr < 0)
            continue;
        if (instance.count(attr) && !this->av_count.count(attr))
        {
            return false;
        }
        if (this->av_count.count(attr) && !instance.count(attr))
        {
            return false;
        }
        // if (this->av_count.count(attr) && instance.count(attr)) {
        //     double instance_attr_exp_sum = 0.0;
        //     double concept_attr_exp_sum = 0.0;
        //     std::unordered_set<VALUE_TYPE> all_vals;

        //     // Collect all values and exponentiated sums from instance
        //     for (auto &[val, cnt]: instance.at(attr)) {
        //         all_vals.insert(val);
        //         instance_attr_exp_sum += std::exp(cnt);
        //     }

        //     // Collect all values and exponentiated sums from av_count
        //     for (auto &[val, cnt]: this->av_count.at(attr)) {
        //         all_vals.insert(val);
        //         concept_attr_exp_sum += std::exp(cnt);
        //     }

        //     std::vector<double> instance_probs, concept_probs;
        //     for (auto &val: all_vals) {
        //         double instance_prob = 0.0;
        //         double concept_prob = 0.0;

        //         if (instance.at(attr).count(val) && !this->av_count.at(attr).count(val)) {
        //             return false;
        //         }
        //         if (this->av_count.at(attr).count(val) && !instance.at(attr).count(val)) {
        //             return false;
        //         }

        //         if (instance.at(attr).count(val)) {
        //             instance_prob = std::exp(instance.at(attr).at(val)) / instance_attr_exp_sum;
        //         }
        //         std::cout << "Instance Prob: " << instance_prob << std::endl;
        //         instance_probs.push_back(instance_prob);

        //         if (this->av_count.at(attr).count(val)) {
        //             concept_prob = std::exp(this->av_count.at(attr).at(val)) / concept_attr_exp_sum;
        //         }
        //         // std::cout << "Concept Prob: " << concept_prob << std::endl;
        //         concept_probs.push_back(concept_prob);
        //     }

        //     double kl_divergence = 0.0;
        //     for (size_t i = 0; i < instance_probs.size(); ++i) {
        //         if (instance_probs[i] > 0) {
        //             kl_divergence += instance_probs[i] * std::log(instance_probs[i] / concept_probs[i]);
        //         }
        //     }

        //     // print out the kl divergence
        //     // std::cout << "KL Divergence: " << kl_divergence << std::endl;

        //     if (kl_divergence >= 0.02) {
        //         return false;
        //     }
        // }
        if (this->av_count.count(attr) && instance.count(attr))
        {
            double instance_attr_count = 0.0;
            std::unordered_set<VALUE_TYPE> all_vals;
            for (auto &[val, tmp] : this->av_count.at(attr))
                all_vals.insert(val);
            for (auto &[val, cnt] : instance.at(attr))
            {
                all_vals.insert(val);
                instance_attr_count += cnt;
            }

            for (auto &val : all_vals)
            {
                if (instance.at(attr).count(val) && !this->av_count.at(attr).count(val))
                {
                    return false;
                }
                if (this->av_count.at(attr).count(val) && !instance.at(attr).count(val))
                {
                    return false;
                }

                double instance_prob = (1.0 * instance.at(attr).at(val)) / instance_attr_count;
                double concept_prob = (1.0 * this->av_count.at(attr).at(val)) / this->a_count.at(attr);

                if (abs(instance_prob - concept_prob) > 0.00001)
                {
                    return false;
                }
            }
        }
    }
    return true;
}

inline size_t CobwebNode::_hash()
{
    return std::hash<uintptr_t>()(reinterpret_cast<uintptr_t>(this));
}

inline std::string CobwebNode::__str__()
{
    return this->pretty_print();
}

inline std::string CobwebNode::concept_hash()
{
    return std::to_string(this->_hash());
}

inline std::string CobwebNode::pretty_print(int depth)
{
    std::string ret = repeat("\t", depth) + "|-" + avcounts_to_json() + "\n";

    for (auto &c : children)
    {
        ret += c->pretty_print(depth + 1);
    }

    return ret;
}

inline int CobwebNode::depth()
{
    if (this->parent)
    {
        return 1 + this->parent->depth();
    }
    return 0;
}

inline bool CobwebNode::is_parent(CobwebNode *otherConcept)
{
    CobwebNode *temp = otherConcept;
    while (temp != nullptr)
    {
        if (temp == this)
        {
            return true;
        }
        try
        {
            temp = temp->parent;
        }
        catch (std::string e)
        {
            std::cout << temp;
            assert(false);
        }
    }
    return false;
}

inline int CobwebNode::num_concepts()
{
    int childrenCount = 0;
    for (auto &c : children)
    {
        childrenCount += c->num_concepts();
    }
    return 1 + childrenCount;
}

inline std::string CobwebNode::avcounts_to_json()
{
    std::string ret = "{";

    // // ret += "\"_expected_guesses\": {\n";
    // ret += "\"_entropy\": {\n";
    // ret += "\"#ContinuousValue#\": {\n";
    // ret += "\"mean\": " + std::to_string(this->entropy()) + ",\n";
    // ret += "\"std\": 1,\n";
    // ret += "\"n\": 1,\n";
    // ret += "}},\n";

    ret += "\"_category_utility\": {\n";
    ret += "\"#ContinuousValue#\": {\n";
    ret += "\"mean\": " + std::to_string(this->category_utility()) + ",\n";
    ret += "\"std\": 1,\n";
    ret += "\"n\": 1,\n";
    ret += "}},\n";

    // ret += "\"_mutual_info\": {\n";
    // ret += "\"#ContinuousValue#\": {\n";
    // ret += "\"mean\": " + std::to_string(this->mutual_information()) + ",\n";
    // ret += "\"std\": 1,\n";
    // ret += "\"n\": 1,\n";
    // ret += "}},\n";

    int c = 0;
    for (auto &[attr, vAttr] : av_count)
    {
        // ret += "\"" + attr.get_string() + "\": {";
        ret += "\"" + std::to_string(attr) + "\": {";
        int inner_count = 0;
        for (auto &[val, cnt] : vAttr)
        {
            // ret += "\"" + val.get_string() + "\": " + doubleToString(cnt);
            ret += "\"" + std::to_string(val) + "\": " + doubleToString(cnt);
            // std::to_string(cnt);
            if (inner_count != int(vAttr.size()) - 1)
            {
                ret += ", ";
            }
            inner_count++;
        }
        ret += "}";

        if (c != int(av_count.size()) - 1)
        {
            ret += ", ";
        }
        c++;
    }
    ret += "}";
    return ret;
}

inline std::string CobwebNode::ser_avcounts()
{
    std::string ret = "{";

    int c = 0;
    for (auto &[attr, vAttr] : av_count)
    {
        // ret += "\"" + attr.get_string() + "\": {";
        ret += "\"" + std::to_string(attr) + "\": {";
        int inner_count = 0;
        for (auto &[val, cnt] : vAttr)
        {
            // ret += "\"" + val.get_string() + "\": " + doubleToString(cnt);
            ret += "\"" + std::to_string(val) + "\": " + doubleToString(cnt);
            // std::to_string(cnt);
            if (inner_count != int(vAttr.size()) - 1)
            {
                ret += ", ";
            }
            inner_count++;
        }
        ret += "}";

        if (c != int(av_count.size()) - 1)
        {
            ret += ", ";
        }
        c++;
    }
    ret += "}";
    return ret;
}

inline std::string CobwebNode::a_count_to_json()
{
    std::string ret = "{";

    bool first = true;
    for (auto &[attr, cnt] : this->a_count)
    {
        if (!first)
            ret += ",\n";
        else
            first = false;
        // ret += "\"" + attr.get_string() + "\": " + doubleToString(cnt);
        ret += "\"" + std::to_string(attr) + "\": " + doubleToString(cnt);
        // std::to_string(cnt);
    }

    ret += "}";
    return ret;
}

inline std::string CobwebNode::sum_n_logn_to_json()
{
    std::string ret = "{";

    bool first = true;
    for (auto &[attr, cnt] : this->sum_n_logn)
    {
        if (!first)
            ret += ",\n";
        else
            first = false;
        // ret += "\"" + attr.get_string() + "\": " + doubleToString(cnt);
        ret += "\"" + std::to_string(attr) + "\": " + doubleToString(cnt);
        // std::to_string(cnt);
    }

    ret += "}";
    return ret;
}

inline std::string CobwebNode::dump_json()
{
    std::string output = "{";

    // output += "\"concept_id\": " + std::to_string(this->_hash()) + ",\n";
    output += "\"count\": " + doubleToString(this->count) + ",\n";
    output += "\"a_count\": " + this->a_count_to_json() + ",\n";
    output += "\"sum_n_logn\": " + this->sum_n_logn_to_json() + ",\n";
    output += "\"av_count\": " + this->ser_avcounts() + ",\n";

    output += "\"children\": [\n";
    bool first = true;
    for (auto &c : children)
    {
        if (!first)
            output += ",";
        else
            first = false;
        output += c->dump_json();
    }
    output += "]\n";

    output += "}\n";

    return output;
}

inline std::string CobwebNode::output_json()
{
    std::string output = "{";

    output += "\"name\": \"Concept" + std::to_string(this->_hash()) + "\",\n";
    output += "\"size\": " + std::to_string(this->count) + ",\n";
    output += "\"children\": [\n";
    bool first = true;
    for (auto &c : children)
    {
        if (!first)
            output += ",";
        else
            first = false;
        output += c->output_json();
    }
    output += "],\n";

    output += "\"counts\": " + this->avcounts_to_json() + ",\n";
    output += "\"attr_counts\": " + this->a_count_to_json() + "\n";

    output += "}\n";

    return output;
}

// TODO
// TODO This should use the path prob, not the node prob.
// TODO
inline std::unordered_map<int, std::unordered_map<int, double>> CobwebNode::predict_weighted_leaves_probs(INSTANCE_TYPE instance)
{

    // AV_COUNT_TYPE cached_instance;
    // for (auto &[attr, val_map] : instance)
    // {
    //     for (auto &[val, cnt] : val_map)
    //     {
    //         cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
    //     }
    // }

    double concept_weights = 0.0;
    std::unordered_map<int, std::unordered_map<int, double>> out;

    // std::cout << std::endl << "Prob of nodes along path (starting with leaf)" << std::endl;
    auto curr = this;
    while (curr->parent != nullptr)
    {
        auto prev = curr;
        curr = curr->parent;

        for (auto &child : curr->children)
        {
            if (child == prev)
                continue;
            double c_prob = exp(child->log_prob_class_given_instance(instance, true));
            // double c_prob = 1.0;
            // std::cout << c_prob << std::endl;
            concept_weights += c_prob;

            for (auto &[attr, val_set] : this->tree->attr_vals)
            {
                // std::cout << attr << std::endl;
                int num_vals = this->tree->attr_vals.at(attr).size();
                float alpha = this->tree->alpha;
                COUNT_TYPE attr_count = 0;

                if (child->a_count.count(attr))
                {
                    attr_count = child->a_count.at(attr);
                }

                for (auto val : val_set)
                {
                    // std::cout << val << std::endl;
                    COUNT_TYPE av_count = 0;
                    if (child->av_count.count(attr) and child->av_count.at(attr).count(val))
                    {
                        av_count = child->av_count.at(attr).at(val);
                    }

                    double p = ((av_count + alpha) / (attr_count + num_vals * alpha));
                    // std::cout << p << std::endl;
                    // if (attr.get_string() == "class"){
                    //     std::cout << val.get_string() << ", " << c_prob << ", " << p << ", " << p * c_prob << " :: ";
                    // }
                    // out[attr.get_string()][val.get_string()] += p * c_prob;
                    out[attr][val] += p * c_prob;
                }
            }
        }
        // std::cout << std::endl;
    }

    for (auto &[attr, val_set] : this->tree->attr_vals)
    {
        for (auto val : val_set)
        {
            // out[attr.get_string()][val.get_string()] /= concept_weights;
            out[attr][val] /= concept_weights;
        }
    }

    return out;
}

inline std::unordered_map<int, std::unordered_map<int, double>> CobwebNode::predict_weighted_probs(INSTANCE_TYPE instance)
{

    // AV_COUNT_TYPE cached_instance;
    // for (auto &[attr, val_map] : instance)
    // {
    //     for (auto &[val, cnt] : val_map)
    //     {
    //         cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
    //     }
    // }

    double concept_weights = 0.0;
    std::unordered_map<int, std::unordered_map<int, double>> out;

    // std::cout << std::endl << "Prob of nodes along path (starting with leaf)" << std::endl;
    auto curr = this;
    while (curr != nullptr)
    {
        double c_prob = exp(curr->log_prob_class_given_instance(instance, true));
        // double c_prob = 1.0;
        // std::cout << c_prob << std::endl;
        concept_weights += c_prob;

        for (auto &[attr, val_set] : this->tree->attr_vals)
        {
            // std::cout << attr << std::endl;
            int num_vals = this->tree->attr_vals.at(attr).size();
            float alpha = this->tree->alpha;
            COUNT_TYPE attr_count = 0;

            if (curr->a_count.count(attr))
            {
                attr_count = curr->a_count.at(attr);
            }

            for (auto val : val_set)
            {
                // std::cout << val << std::endl;
                COUNT_TYPE av_count = 0;
                if (curr->av_count.count(attr) and curr->av_count.at(attr).count(val))
                {
                    av_count = curr->av_count.at(attr).at(val);
                }

                double p = ((av_count + alpha) / (attr_count + num_vals * alpha));
                // std::cout << p << std::endl;
                // if (attr.get_string() == "class"){
                //     std::cout << val.get_string() << ", " << c_prob << ", " << p << ", " << p * c_prob << " :: ";
                // }
                // out[attr.get_string()][val.get_string()] += p * c_prob;
                out[attr][val] += p * c_prob;
            }
        }
        // std::cout << std::endl;

        curr = curr->parent;
    }

    for (auto &[attr, val_set] : this->tree->attr_vals)
    {
        for (auto val : val_set)
        {
            // out[attr.get_string()][val.get_string()] /= concept_weights;
            out[attr][val] /= concept_weights;
        }
    }

    return out;
}

inline std::unordered_map<int, std::unordered_map<int, double>> CobwebNode::predict_log_probs()
{
    std::unordered_map<int, std::unordered_map<int, double>> out;
    for (auto &[attr, val_set] : this->tree->attr_vals)
    {
        // std::cout << attr << std::endl;
        int num_vals = this->tree->attr_vals.at(attr).size();
        float alpha = this->tree->alpha;
        COUNT_TYPE attr_count = 0;
        float second_term = 0;

        if (this->a_count.count(attr))
        {
            attr_count = this->a_count.at(attr);
            second_term = log(attr_count + num_vals * alpha);
        }
        else
        {
            second_term = log(num_vals) + this->tree->log_alpha;
        }

        for (auto val : val_set)
        {
            // std::cout << val << std::endl;
            COUNT_TYPE av_count = 0;
            if (this->av_count.count(attr) and this->av_count.at(attr).count(val))
            {
                av_count = this->av_count.at(attr).at(val);
                out[attr][val] = (log(av_count + alpha) - second_term);
            }

            // double p = ((av_count + alpha) / (attr_count + num_vals * alpha));
            // out[attr.get_string()][val.get_string()] += p;
            // std::cout << p << std::endl;
            // out[attr.get_string()][val.get_string()] = (log(av_count + alpha) - log(attr_count + num_vals * alpha));
            // out[attr][val] = (log(av_count + alpha) - log(attr_count + num_vals * alpha));
        }
    }

    return out;
}

inline std::unordered_map<int, std::unordered_map<int, double>> CobwebNode::predict_probs()
{
    std::unordered_map<int, std::unordered_map<int, double>> out;
    for (auto &[attr, val_set] : this->tree->attr_vals)
    {
        // std::cout << attr << std::endl;
        int num_vals = this->tree->attr_vals.at(attr).size();
        float alpha = this->tree->alpha;
        COUNT_TYPE attr_count = 0;

        if (this->a_count.count(attr))
        {
            attr_count = this->a_count.at(attr);
        }

        for (auto val : val_set)
        {
            // std::cout << val << std::endl;
            COUNT_TYPE av_count = 0;
            if (this->av_count.count(attr) and this->av_count.at(attr).count(val))
            {
                av_count = this->av_count.at(attr).at(val);
            }

            double p = ((av_count + alpha) / (attr_count + num_vals * alpha));
            // std::cout << p << std::endl;
            // out[attr.get_string()][val.get_string()] += p;
            out[attr][val] += p;
        }
    }

    return out;
}

inline std::vector<std::tuple<VALUE_TYPE, double>> CobwebNode::get_weighted_values(
    ATTR_TYPE attr, bool allowNone)
{

    std::vector<std::tuple<VALUE_TYPE, double>> choices;
    if (!this->av_count.count(attr))
    {
        choices.push_back(std::make_tuple(-1, 1.0));
    }
    double valCount = 0;
    for (auto &[val, tmp] : this->av_count.at(attr))
    {
        COUNT_TYPE count = this->av_count.at(attr).at(val);
        choices.push_back(std::make_tuple(val, (1.0 * count) / this->count));
        valCount += count;
    }
    if (allowNone)
    {
        choices.push_back(std::make_tuple(-1, ((1.0 * (this->count - valCount)) / this->count)));
    }
    return choices;
}

inline VALUE_TYPE CobwebNode::predict(ATTR_TYPE attr, std::string choiceFn, bool allowNone)
{
    std::function<ATTR_TYPE(std::vector<std::tuple<VALUE_TYPE, double>>)> choose;
    if (choiceFn == "most likely" || choiceFn == "m")
    {
        choose = most_likely_choice;
    }
    else if (choiceFn == "sampled" || choiceFn == "s")
    {
        choose = weighted_choice;
    }
    else
        throw "Unknown choice_fn";
    if (!this->av_count.count(attr))
    {
        return -1;
    }
    std::vector<std::tuple<VALUE_TYPE, double>> choices = this->get_weighted_values(attr, allowNone);
    return choose(choices);
}

inline double CobwebNode::probability(ATTR_TYPE attr, VALUE_TYPE val)
{
    if (val == -1)
    {
        double c = 0.0;
        if (this->av_count.count(attr))
        {
            for (auto &[attr, vAttr] : this->av_count)
            {
                for (auto &[val, cnt] : vAttr)
                {
                    c += cnt;
                }
            }
            return (1.0 * (this->count - c)) / this->count;
        }
    }
    if (this->av_count.count(attr) && this->av_count.at(attr).count(val))
    {
        return (1.0 * this->av_count.at(attr).at(val)) / this->count;
    }
    return 0.0;
}

inline double CobwebNode::category_utility()
{
    // double p_of_c = (1.0 * this->count) / this->tree->root->count;
    // return (p_of_c * (this->tree->root->entropy() - this->entropy()));

    double root_entropy = 0.0;
    double child_entropy = 0.0;

    double p_of_child = (1.0 * this->count) / this->tree->root->count;
    for (auto &[attr, val_set] : this->tree->attr_vals)
    {
        root_entropy += this->tree->root->entropy_attr(attr);
        child_entropy += this->entropy_attr(attr);
    }

    return p_of_child * (root_entropy - child_entropy);
}

inline std::vector<double> CobwebNode::log_prob_children_given_instance_ext(INSTANCE_TYPE instance)
{
    // AV_COUNT_TYPE cached_instance;
    // for (auto &[attr, val_map] : instance)
    // {
    //     for (auto &[val, cnt] : val_map)
    //     {
    //         cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
    //     }
    // }

    return this->log_prob_children_given_instance(instance);
}

inline std::vector<double> CobwebNode::log_prob_children_given_instance(const AV_COUNT_TYPE &instance)
{
    std::vector<double> raw_log_probs = std::vector<double>();
    std::vector<double> norm_log_probs = std::vector<double>();

    for (auto &child : this->children)
    {
        raw_log_probs.push_back(child->log_prob_class_given_instance(instance, false));
    }

    double log_p_of_x = logsumexp(raw_log_probs);

    for (auto log_p : raw_log_probs)
    {
        norm_log_probs.push_back(log_p - log_p_of_x);
    }

    return norm_log_probs;
}

inline std::vector<double> CobwebNode::prob_children_given_instance_ext(INSTANCE_TYPE instance)
{
    // AV_COUNT_TYPE cached_instance;
    // for (auto &[attr, val_map] : instance)
    // {
    //     for (auto &[val, cnt] : val_map)
    //     {
    //         cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
    //     }
    // }

    return this->prob_children_given_instance(instance);
}

inline std::vector<double> CobwebNode::prob_children_given_instance(const AV_COUNT_TYPE &instance)
{

    double sum_probs = 0;
    std::vector<double> raw_probs = std::vector<double>();
    std::vector<double> norm_probs = std::vector<double>();

    for (auto &child : this->children)
    {
        double p = exp(child->log_prob_class_given_instance(instance, false));
        sum_probs += p;
        raw_probs.push_back(p);
    }

    for (auto p : raw_probs)
    {
        norm_probs.push_back(p / sum_probs);
    }

    return norm_probs;
}

inline double CobwebNode::log_prob_class_given_instance_ext(INSTANCE_TYPE instance, bool use_root_counts)
{
    // AV_COUNT_TYPE cached_instance;
    // for (auto &[attr, val_map] : instance)
    // {
    //     for (auto &[val, cnt] : val_map)
    //     {
    //         cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
    //     }
    // }

    return this->log_prob_class_given_instance(instance, use_root_counts);
}

inline double CobwebNode::log_prob_class_given_instance(const AV_COUNT_TYPE &instance, bool use_root_counts)
{

    double log_prob = log_prob_instance(instance);

    if (use_root_counts)
    {
        log_prob += log((1.0 * this->count) / this->tree->root->count);
    }
    else
    {
        log_prob += log((1.0 * this->count) / this->parent->count);
    }

    // std::cout << "LOB PROB" << std::to_string(log_prob) << std::endl;

    return log_prob;
}

inline double CobwebNode::log_prob_instance_ext(INSTANCE_TYPE instance)
{
    // AV_COUNT_TYPE cached_instance;
    // for (auto &[attr, val_map] : instance)
    // {
    //     for (auto &[val, cnt] : val_map)
    //     {
    //         cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
    //     }
    // }

    return this->log_prob_instance(instance);
}

inline double CobwebNode::log_prob_instance(const AV_COUNT_TYPE &instance)
{

    double log_prob = 0;

    for (auto &[attr, vAttr] : instance)
    {
        // bool hidden = attr.is_hidden();
        bool hidden = (attr < 0);
        if (hidden || !this->tree->attr_vals.count(attr))
        {
            continue;
        }

        double num_vals = this->tree->attr_vals.at(attr).size();

        for (auto &[val, cnt] : vAttr)
        {
            if (!this->tree->attr_vals.at(attr).count(val))
            {
                continue;
            }

            double alpha = this->tree->alpha;
            double av_count = alpha;
            if (this->av_count.count(attr) && this->av_count.at(attr).count(val))
            {
                av_count += this->av_count.at(attr).at(val);
            }

            // a_count starts with the alphas over all values (even vals not in
            // current node)
            COUNT_TYPE a_count = num_vals * alpha;
            if (this->a_count.count(attr))
            {
                a_count += this->a_count.at(attr);
            }

            // we use cnt here to weight accuracy by counts in the training
            // instance. Usually this is 1, but in  models, it might
            // be something else.
            log_prob += cnt * (log(av_count) - log(a_count));
        }
    }

    return log_prob;
}

inline double CobwebNode::log_prob_instance_missing_ext(INSTANCE_TYPE instance)
{
    // AV_COUNT_TYPE cached_instance;
    // for (auto &[attr, val_map] : instance)
    // {
    //     for (auto &[val, cnt] : val_map)
    //     {
    //         cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
    //     }
    // }

    return this->log_prob_instance_missing(instance);
}

inline double CobwebNode::log_prob_instance_missing(const AV_COUNT_TYPE &instance)
{

    double log_prob = 0;

    for (auto &[attr, val_set] : this->tree->attr_vals)
    {
        // for (auto &[attr, vAttr]: instance) {
        // bool hidden = attr.is_hidden();
        bool hidden = (attr < 0);
        if (hidden)
        {
            continue;
        }

        double num_vals = this->tree->attr_vals.at(attr).size();
        double alpha = this->tree->alpha;

        if (instance.count(attr))
        {
            for (auto &[val, cnt] : instance.at(attr))
            {

                // TODO IS THIS RIGHT???
                // we could treat it as just alpha...
                if (!this->tree->attr_vals.at(attr).count(val))
                {
                    std::cout << "VALUE MISSING TREATING AS ALPHA" << std::endl;
                    // continue;
                }

                double av_count = alpha;
                if (this->av_count.count(attr) && this->av_count.at(attr).count(val))
                {
                    av_count += this->av_count.at(attr).at(val);
                }

                // a_count starts with the alphas over all values (even vals not in
                // current node)
                COUNT_TYPE a_count = num_vals * alpha;
                if (this->a_count.count(attr))
                {
                    a_count += this->a_count.at(attr);
                }

                // we use cnt here to weight accuracy by counts in the training
                // instance. Usually this is 1, but in  models, it might
                // be something else.
                log_prob += cnt * (log(av_count) - log(a_count));
            }
        }
        else
        {
            double cnt = 1.0;
            if (this->tree->weight_attr)
            {
                cnt = (1.0 * this->tree->root->a_count.at(attr)) / (this->tree->root->count);
            }

            int num_vals_in_c = 0;
            if (this->av_count.count(attr))
            {
                auto attr_count = this->a_count.at(attr);
                num_vals_in_c = this->av_count.at(attr).size();
                for (auto &[val, av_count] : this->av_count.at(attr))
                {
                    double p = ((av_count + alpha) / (attr_count + num_vals * alpha));
                    log_prob += cnt * p * log(p);
                }
            }

            int n0 = num_vals - num_vals_in_c;
            double p_missing = alpha / (num_vals * alpha);
            log_prob += cnt * n0 * p_missing * log(p_missing);
        }
    }

    return log_prob;
}

int main(int argc, char *argv[])
{
    // std::vector<AV_COUNT_TYPE> instances;
    // std::vector<CobwebNode *> cs;
    // auto tree = CobwebTree(0.01, false, 2, true, true);

    // for (int i = 0; i < 1000; i++)
    // {
    //     INSTANCE_TYPE inst;
    //     inst["anchor"]["word" + std::to_string(i)] = 1;
    //     inst["anchor2"]["word" + std::to_string(i % 10)] = 1;
    //     inst["anchor3"]["word" + std::to_string(i % 20)] = 1;
    //     inst["anchor4"]["word" + std::to_string(i % 100)] = 1;
    //     // unpack the tuple return
    //     auto [current, _] = tree.ifit(inst, 0);
    //     cs.push_back(current);
    // }

    return 0;
}

// ------------------------- NEW: set_av_count implementation -------------------------

inline void CobwebNode::set_av_count(const AV_COUNT_TYPE &new_av_count)
{
    // Replace the av_count
    this->av_count = new_av_count;

    // Clear derived fields
    this->a_count.clear();
    this->sum_n_logn.clear();
    this->count = 0.0;

    // Recompute a_count, sum_n_logn, and count based on av_count
    for (const auto &pair_attr : this->av_count)
    {
        const ATTR_TYPE attr = pair_attr.first;
        const VAL_COUNT_TYPE &val_map = pair_attr.second;

        double attr_sum = 0.0;
        for (const auto &pair_val : val_map)
        {
            const VALUE_TYPE val = pair_val.first;
            const COUNT_TYPE cnt = pair_val.second;

            this->a_count[attr] += cnt;
            attr_sum += cnt;
            this->count += cnt;

            if (cnt > 0)
            {
                // contribution to sum_n_logn
                this->sum_n_logn[attr] += (cnt * std::log(cnt));
            }
        }

        // If we need sum_n_logn semantics consistent with increment_counts (which uses tf*log(tf) with alpha),
        // we leave the computed sum_n_logn as-is. When alpha is used in entropy calculations, the other code uses
        // tf = count + alpha when needed.
        // If attr_sum > 0 we might adjust based on normalization; for now we keep it consistent with above.
    }
}


// ------------------------- nanobind module -------------------------

NB_MODULE(cobweb_discrete, m)
{
    m.doc() = "Cobweb discrete implementation WITH Chunking Changes";

    nb::class_<CobwebNode>(m, "CobwebNode")
        .def(nb::init<>())
        .def("set_av_count", &CobwebNode::set_av_count)
        .def("pretty_print", &CobwebNode::pretty_print)
        .def("output_json", &CobwebNode::output_json)
        .def("predict_probs", &CobwebNode::predict_probs)
        .def("predict_log_probs", &CobwebNode::predict_log_probs)
        .def("predict_weighted_probs", &CobwebNode::predict_weighted_probs)
        .def("predict_weighted_leaves_probs", &CobwebNode::predict_weighted_leaves_probs)
        .def("predict", &CobwebNode::predict, nb::arg("attr") = "",
             nb::arg("choiceFn") = "most likely",
             nb::arg("allowNone") = true)
        .def("get_best_level", &CobwebNode::get_best_level, nb::rv_policy::reference)
        .def("get_basic_level", &CobwebNode::get_basic_level, nb::rv_policy::reference)
        .def("log_prob_class_given_instance", &CobwebNode::log_prob_class_given_instance_ext)
        .def("log_prob_instance", &CobwebNode::log_prob_instance_ext)
        .def("log_prob_instance_missing", &CobwebNode::log_prob_instance_missing_ext)
        .def("prob_children_given_instance", &CobwebNode::prob_children_given_instance_ext)
        .def("log_prob_children_given_instance", &CobwebNode::log_prob_children_given_instance_ext)
        .def("entropy", &CobwebNode::entropy)
        .def("category_utility", &CobwebNode::category_utility)
        .def("partition_utility", &CobwebNode::partition_utility)
        .def("__str__", &CobwebNode::__str__)
        .def("concept_hash", &CobwebNode::concept_hash)
        .def_ro("count", &CobwebNode::count)
        .def_ro("children", &CobwebNode::children, nb::rv_policy::reference)
        .def_ro("parent", &CobwebNode::parent, nb::rv_policy::reference)
        .def_ro("av_count", &CobwebNode::av_count, nb::rv_policy::reference)
        .def_ro("a_count", &CobwebNode::a_count, nb::rv_policy::reference)
        .def_ro("tree", &CobwebNode::tree, nb::rv_policy::reference);

    nb::class_<CobwebTree>(m, "CobwebTree")
        .def(nb::init<float, bool, int, bool, bool>(),
             nb::arg("alpha") = 1.0,
             nb::arg("weight_attr") = false,
             nb::arg("objective") = 0,
             nb::arg("children_norm") = true,
             nb::arg("norm_attributes") = false)
        .def("ifit", &CobwebTree::ifit,
             nb::arg("instance") = std::vector<AV_COUNT_TYPE>(),
             nb::arg("mode"),
             nb::arg("debug") = false,
              nb::rv_policy::reference)
        .def("fit", &CobwebTree::fit, nb::arg("instances") = std::vector<AV_COUNT_TYPE>(), nb::arg("mode"), nb::arg("iterations") = 1, nb::arg("randomizeFirst") = true)
        .def("categorize", &CobwebTree::categorize, nb::arg("instance") = std::vector<AV_COUNT_TYPE>(),
             // nb::arg("get_best_concept") = false,
             nb::rv_policy::reference)
        .def("predict_probs_mixture", &CobwebTree::predict_probs_mixture)
        .def("predict_probs_mixture_parallel", &CobwebTree::predict_probs_mixture_parallel)
        .def("predict_probs", &CobwebTree::predict_probs_mixture)
        .def("predict_probs_parallel", &CobwebTree::predict_probs_mixture_parallel)
        .def("clear", &CobwebTree::clear)
        .def("__str__", &CobwebTree::__str__)
        // .def("dump_json", &CobwebTree::dump_json)
        // .def("load_json", &CobwebTree::load_json)
        .def("load_json_stream", &CobwebTree::load_json_stream)
        .def("write_json_stream", &CobwebTree::write_json_stream)
        .def_ro("root", &CobwebTree::root, nb::rv_policy::reference);
}
