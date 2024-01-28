#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using namespace std;

// class StepSelector {
// private:
//     std::vector<int> steps;
//     std::discrete_distribution<> dist;

// public:
//     StepSelector(std::vector<int>& steps_list, std::vector<double>& probs)
//         : steps(steps_list), dist(probs.begin(), probs.end()) {}

//     int select_step() {
//         std::random_device rd;
//         std::mt19937 gen(rd());

//         return steps[dist(gen)];
//     }
// };

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

int cnt[6];

int main() {
    // 定义不同步长和对应概率的列表
    std::vector<int> step_values = {1, 2, 3, 4, 5};
    std::vector<double> probabilities = {0.1, 0.2, 0.3, 0.1, 0.3}; // 与步长列表对应

    StepSelector selector(step_values, probabilities);

    // 选择步长示例
    int c = 1e8;

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < c; ++i) {
        int selected_step = selector.select_step();
        cnt[selected_step]++;
        // std::cout << "Selected step: " << selected_step << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    std::cout << "程序运行时间： " << elapsed_time.count() << " 秒\n";

    for (int i = 1; i <= 5; i++) cout << i << ": " << 1.0 * cnt[i] / c << '\n';

    return 0;
}