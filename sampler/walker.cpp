#include "walker.hpp"

#include <cassert>
#include <numeric>
#include <fstream>
#include <cassert>
#include <functional>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <omp.h>
#include <iostream>

// include gflags
#include <gflags/gflags.h>
DECLARE_int32(num_threads_svd);
DECLARE_int32(rank);
DECLARE_int32(negative);
DECLARE_string(output_svd);
using namespace std; 
// using namespace log4cxx;

void BinaryGraphWalker::turn_to_csr(std::vector<int> &indices,
    std::vector<size_t> &indptr,
    std::vector<float> &data,std::vector<size_t> &degree_new,size_t num_edge_new,std::vector<ValuedVertexPair>* counter_merged){
        EdgeId idx = 0;
            for (const auto& pair : *counter_merged) {
        VertexId src = pair.first.first;
        VertexId dst = pair.first.second;
   
          idx = indptr[src] + (--degree_new[src]);
          indices[idx] = dst;
          data[idx] = pair.second;
        //   idx = indptr[dst] + (--degree_new[dst]);
        //   indices[idx] = src;
        //   data[idx] = pair.second;  
       
    }
    }



std::pair<std::vector<double>, std::vector<int>> get_arr_value_list(int window_size, double w, int p) {
    std::vector<double> ar;
    for (int i = 0; i <= window_size; ++i) {
        ar.push_back(std::pow(w, i) / (std::pow(std::tgamma(i + 1), p)));  
    }

    double sum_ar = std::accumulate(ar.begin(), ar.end(), 0.0);

    for (double& val : ar) {
        val /= sum_ar;
    }

    double theat0 = ar[0];
    ar.erase(ar.begin());

    sum_ar = std::accumulate(ar.begin(), ar.end(), 0.0);

    for (double& val : ar) {
        val /= sum_ar;
    }

    std::vector<int> value_list;
    for (size_t i = 0; i < ar.size(); ++i) {
        value_list.push_back(i + 1);
    }

    return std::make_pair(ar, value_list);
}


BinaryGraphWalker::BinaryGraphWalker(const std::vector<VertexId>& indices_,
        const std::vector<VertexId>& indptr_, int num_node_)
    : indices(indices_), indptr(indptr_), num_node(num_node_) {
    
    // LOG4CXX_INFO(logger, "unweighted network");
    // cout << "unweighted network\n";
}




VertexId BinaryGraphWalker::randomWalk(VertexId u, int step,
        unsigned* seed) const {
    for (;step--;) {
        // u's neighbors are indices[indptr[i]:indptr[i+1]]
        int offset = rand_r(seed) % (indptr[u+1] - indptr[u]);
        u = indices[indptr[u] + offset];
    }
    return u;
}

void BinaryGraphWalker::samplePath(const VertexId u, const VertexId v, int r, unsigned* seed, 
        std::vector<VertexPair>& sampled_pairs) const {
    // if(generateRandomNumber()<0.5){
    //     std::swap(u, v);
    //     }
    int k = rand_r(seed) % r + 1;
    VertexId u_ = randomWalk(u, k - 1, seed);
    VertexId v_ = randomWalk(v, r - k, seed);
    // add record (u_, v_, 1)

    // if (u_ > v_) {
    //     std::swap(u_, v_);
    // }

    sampled_pairs.push_back(std::make_pair(u_, v_));
}

csr_matrix BinaryGraphWalker::sampling(int round, int num_threads,
        int check_point,int window_size, double w, double p) {
    omp_set_num_threads(num_threads);
    auto values = get_arr_value_list(window_size, w, p);
    std::vector<double> probabilities = values.first;
    std::vector<int> step_values = values.second;
    // 与步长列表对应
    StepSelector selector(step_values, probabilities);
    RandomGenerator randGen;
    std::vector<std::vector<ValuedVertexPair>*> counters;
    for (int i = 0; i < num_threads; ++i) {
        counters.push_back(new std::vector<ValuedVertexPair>);
    }

    #pragma omp parallel default(shared)
    {
        int this_thread = omp_get_thread_num();
        std::string thread_name = std::string("local_server") 
            + std::string("_thread_") + std::to_string(this_thread); // + std::string("_time_") + std::to_string(time(0));

        // LOG4CXX_INFO(logger, "[thread " << this_thread << "]" << " thread name is " << thread_name );
        // cout << "[thread " << this_thread << "]" << " thread name is " << thread_name << '\n';
        unsigned seed = std::hash<std::string>{}(thread_name);

        std::vector<VertexPair> sampled_pairs;
        std::vector<ValuedVertexPair> *&counter = counters[this_thread];
        std::vector<ValuedVertexPair> *counter_tmp = new std::vector<ValuedVertexPair>;

        // LOG4CXX_INFO(logger, "[thread " << this_thread << "]" << " set seed " << seed);
        // cout << "[thread " << this_thread << "]" << " set seed " << seed << '\n';
        int my_round= ceil((double)round / num_threads);
        VertexId u_start=0;
        VertexId v_start=0;
        for (int i=0; i<my_round; ++i) {
            for (VertexId u=0; u+1 < indptr.size(); ++u) {
                for (size_t j=indptr[u]; j<indptr[u+1]; ++j) {
                    VertexId v = indices[j];
                        
                        u_start = u;
                        v_start = v;
                        if(randGen.generateRandomNumber() < 0.5){
                                std::swap(u_start, v_start);
                                                }
                        int lengthofstep = selector.select_step();
                        
                        samplePath(u_start, v_start, lengthofstep, &seed, sampled_pairs);
                 
                }
            }
            if ((i + 1) % check_point == 0 || i + 1 == my_round) {
                float max_val = merge(*counter, *counter_tmp, sampled_pairs);
                std::swap(counter, counter_tmp);
                sampled_pairs.clear();
                counter_tmp->clear();
                
            }
        }
      
        delete counter_tmp;
    }
    cout<<"threads to sampled_pairs check\n";
 
    while (counters.size() > 1) {
       
        size_t n_half = (counters.size() + 1) >> 1;
        omp_set_num_threads(counters.size() >> 1); 

        #pragma omp parallel default(shared)
        {
            int this_thread = omp_get_thread_num();
            
            std::vector<ValuedVertexPair> *counter_tmp = merge_counters(*counters[this_thread], *counters[n_half + this_thread]);

            delete counters[this_thread];
            delete counters[n_half + this_thread];
            counters[this_thread] = counter_tmp;
        }

        counters.resize(n_half);
    };
    //***********************************************************************
    cout<<"threads to counter_merged check\n";
    
    counter_merged = counters[0];
    size_t num_edge_new =(*counter_merged).size();
            
 
    std::vector<size_t> degree_new(num_node);
    // int edgenum_old = indices.size();
    // cout<<"old graph has "<<edgenum_old<<"edges\n";

    float weigh = (1.0)/(round);
       cout<<"start to for check\n";
    for (auto& pair : *counter_merged) { 
           
        pair.second = pair.second * weigh;
        degree_new[pair.first.first]++;
            // degree_new[pair.first.second]++;
            // num_edge_new=num_edge_new+2;
        
         
    }
    cout<<"finish to for check\n";
   
    cout<<"num_edge_new:"<<num_edge_new<<'\n';
    std::vector<int> indices_new(num_edge_new, 0);
    cout<<"indices check\n";
    std::vector<size_t> indptr_new(num_node + 1  , 0);
    cout<<"indptr check\n";
    std::vector<float> data_new( num_edge_new , 0);
    cout<<"data check\n";
    std::partial_sum(degree_new.begin(), degree_new.end(), indptr_new.begin() + 1);
    cout<<"last of indptr_new 1:"<<indptr_new.back()<<'\n';
    cout<<"turn_to_csr start check\n";
    turn_to_csr(indices_new,indptr_new,data_new,degree_new,num_edge_new,counter_merged);
    cout<<"turn_to_csr end check\n";
    // cout << "length of indices is " << indices_new.size() << '\n';
    // cout << "length of indptr is " << indptr_new.size() << '\n';
    // cout << "length of data is " << data_new.size() << '\n';
    // for (const auto &p : indptr_new)
    // {
    //         cout << p<<"\n";
    // }             
    cout<<"last of indptr_new 2 :"<<indptr_new.back()<<'\n';
    auto sampled_matrix = csr_matrix(indices_new,indptr_new,data_new);
   
    return sampled_matrix;
}

float BinaryGraphWalker::merge(const std::vector<ValuedVertexPair>& counter, 
        std::vector<ValuedVertexPair>& tmp,
        std::vector<VertexPair>& sampled_pairs) {
    float max_val = 0;
    std::sort(sampled_pairs.begin(), sampled_pairs.end());  //

    std::vector<ValuedVertexPair>::const_iterator iter = counter.cbegin();
    for (size_t i = 0, j = 0; i < sampled_pairs.size(); i = j) {  //
        for (j = i + 1; j < sampled_pairs.size() && sampled_pairs[j] == sampled_pairs[i]; ++j);  //
        for (;iter != counter.end() && iter->first < sampled_pairs[i]; ++iter) {  //
            max_val = std::max(max_val, iter->second);
            tmp.push_back(*iter);
        }
        if (iter != counter.end() && iter->first == sampled_pairs[i]) {
            max_val = std::max(max_val, j - i + iter->second);
            tmp.push_back(
                    std::make_pair(iter->first, j - i + iter->second));
            ++iter;
        } else {
            max_val = std::max(max_val, float(j - i));
            tmp.push_back(std::make_pair(sampled_pairs[i], float(j - i)));
        }
    }
    for (;iter != counter.end(); ++iter) {
        max_val = std::max(max_val, iter->second);
        tmp.push_back(*iter);
    }
    return max_val;
}


std::vector<ValuedVertexPair>* BinaryGraphWalker::merge_counters(const std::vector<ValuedVertexPair>& counter,
        const std::vector<ValuedVertexPair>& counter_other) { //
    std::vector<ValuedVertexPair>::const_iterator iter1 = counter.cbegin();
    std::vector<ValuedVertexPair>::const_iterator iter2 = counter_other.cbegin(); //
    std::vector<ValuedVertexPair> *counter_tmp = new std::vector<ValuedVertexPair>;

    while (iter1 != counter.cend() && iter2 != counter_other.cend()) { //
        if (iter1->first < iter2->first) {    //
            counter_tmp->push_back(*(iter1++));
        } else if (iter1->first > iter2->first) {
            counter_tmp->push_back(*(iter2++));
        } else {
            counter_tmp->push_back(
                    std::make_pair(iter1->first, iter1->second + iter2->second));
            ++iter1;
            ++iter2;
        }
    }

    for (;iter1 != counter.cend(); ++iter1) {
        counter_tmp->push_back(*iter1);
    }

    for (;iter2 != counter_other.cend(); ++iter2) {
        counter_tmp->push_back(*iter2);
    }
    return counter_tmp;
}

csr_matrix BinaryGraphWalker::run(const std::vector<VertexId>& indices_,
        const std::vector<VertexId>& indptr_, int num_node_,int num_round, int window_size, double w,double p) {
            int num_work=40;
            int num_check=2;
            auto graph  =  new BinaryGraphWalker(indices_, indptr_, num_node_);
            cout<<" new BinaryGraphWalker check\n";
            auto sampled_graph = graph->sampling(num_round,num_work,num_check,window_size,w,p);
            cout<<"graph->sampling check\n";
            return sampled_graph;
        }

