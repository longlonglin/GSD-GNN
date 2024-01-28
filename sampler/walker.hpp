#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
// #include "log4cxx/logger.h"

using namespace std;
using VertexId = unsigned int; //uint32_t;
using EdgeId = unsigned int; //uint32_t;
using VertexPair = std::pair<VertexId, VertexId>;
using VertexPairCount = std::pair<VertexPair, unsigned int>;
using ValuedVertexPair = std::pair<std::pair<VertexId, VertexId>, float>;

class  csr_matrix{

public:
   std::vector<int> indices;
   std::vector<size_t> indptr; 
   std::vector<float> data;
   csr_matrix(std::vector<int> indices_,std::vector<size_t> indptr_, std::vector<float> data_):indices(indices_), indptr(indptr_),data(data_){
    
   };
};

class BinaryGraphWalker  {
public:
    // static log4cxx::LoggerPtr logger;
 
    BinaryGraphWalker(const std::vector<VertexId>& indices_,
            const std::vector<VertexId>& indptr_, int num_node_
           );
    const std::vector<VertexId> indices;
    const std::vector<VertexId> indptr;
    
    int num_node;
    std::vector<ValuedVertexPair> *counter_merged;

    

    void samplePath(const VertexId u,const VertexId v, int r, unsigned* seed,
            std::vector<VertexPair>& sampled_pair) const;
    VertexId randomWalk(VertexId u, int step, unsigned* seed) const;
    csr_matrix sampling(int round, int num_threads,
            int check_point,int window_size, double w, double p);
    void turn_to_csr(std::vector<int> &indices,
    std::vector<size_t> &indptr,
    std::vector<float> &data,std::vector<size_t>&degree_new,size_t num_edge_new,std::vector<ValuedVertexPair>* counter_merged);
    
    float merge(const std::vector<ValuedVertexPair>& counter,
            std::vector<ValuedVertexPair>& tmp,
            std::vector<VertexPair>& sampled_pairs);
    
    // double generateRandomNumber() {
    //     return dist_(gen_);
    // }
   
    std::vector<ValuedVertexPair>* merge_counters(
        const std::vector<ValuedVertexPair>& counter,
        const std::vector<ValuedVertexPair>& counter_other);
    static csr_matrix run(const std::vector<VertexId>& indices_,
        const std::vector<VertexId>& indptr_, int num_node_, int num_round, int window_size, double w,double p);
};

class StepSelector {
private:
    std::vector<int> steps;
    std::vector<double> probabilities;
    std::random_device rd;
    std::mt19937 gen;
    std::discrete_distribution<> dist;

public:
    StepSelector(std::vector<int>& steps_list, std::vector<double>& probs)
        : steps(steps_list), probabilities(probs), gen(rd()), dist(probabilities.begin(), probabilities.end()) {}

    int select_step() {
        return steps[dist(gen)];
    }
};

class RandomGenerator {
private:
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<> dis;

public:
    RandomGenerator() : gen(rd()), dis(0.0, 1.0) {}

    double generateRandomNumber() {
        return dis(gen);
    }
};