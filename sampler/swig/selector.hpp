#include <vector>
#include <random>
#include <cstdio>

class StepSelector
{
private:
    std::vector<int> steps;
    std::vector<double> probabilities;
    std::random_device rd;
    std::mt19937 gen;
    std::discrete_distribution<> dist;

public:
    StepSelector(std::vector<int> &steps_list, std::vector<double> &probs)
        : steps(steps_list), probabilities(probs), gen(rd()), dist(probabilities.begin(), probabilities.end()) {}

    int select_step()
    {
        return steps[dist(gen)];
    }
};

void run(int c)
{
    int cnt[6] = {0};
    std::vector<int> step_values = {1, 2, 3, 4, 5};
    std::vector<double> probabilities = {0.1, 0.2, 0.3, 0.1, 0.3}; // 与步长列表对应

    StepSelector selector(step_values, probabilities);

    for (int i = 0; i < c; ++i)
    {
        int selected_step = selector.select_step();
        cnt[selected_step]++;
    }

    for (int i = 1; i <= 5; i++)
        printf("%d: %.5lf\n", i, 1.0 * cnt[i] / c);
}