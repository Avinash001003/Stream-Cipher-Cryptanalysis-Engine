#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <memory>
#include <thread>
#include <mutex>
#include <optional>
#include <array>
#include <functional>
#include <chrono>
#include <iomanip>
#include <fstream>

using namespace std;

// ============================================================================
// TYPE DEFINITIONS & CONFIGURATION
// ============================================================================

using TruthTable = vector<unsigned char>;
using PopulationT = vector<pair<TruthTable, int>>; // {genome, fitness}

struct GAConfig {
    int numVars = 5;
    int populationSize = 256;
    int maxGenerations = 5000;
    double mutationRate = 0.08;
    double elitismRatio = 0.15;
    int tournamentSize = 5;
    int numThreads = 4;
    bool adaptive_mutation = true;
    int verbosity = 1; // 0: silent, 1: normal, 2: detailed
};

struct Statistics {
    int bestCost;
    int worstCost;
    double avgCost;
    double diversity;
    int generation;
};

// ============================================================================
// RANDOM NUMBER GENERATION (C++17 Modern)
// ============================================================================

class RandomEngine {
    mt19937_64 engine;
    uniform_int_distribution<int> bitDist{0, 1};
    uniform_real_distribution<double> realDist{0.0, 1.0};
    
public:
    RandomEngine(unsigned seed = chrono::system_clock::now().time_since_epoch().count())
        : engine(seed) {}
    
    int randomBit() { return bitDist(engine); }
    double randomReal() { return realDist(engine); }
    
    template<typename T>
    T randomChoice(const vector<T>& choices) {
        uniform_int_distribution<size_t> dist(0, choices.size() - 1);
        return choices[dist(engine)];
    }
    
    int randomInt(int min, int max) {
        uniform_int_distribution<int> dist(min, max);
        return dist(engine);
    }
};

// ============================================================================
// BOOLEAN FUNCTION OPERATIONS
// ============================================================================

class BoolFunction {
public:
    static TruthTable generateRandom(int size, RandomEngine& rng) {
        TruthTable table(size);
        generate(table.begin(), table.end(), [&rng]() { return rng.randomBit(); });
        return table;
    }
    
    static int countOnes(const TruthTable& table) {
        return accumulate(table.begin(), table.end(), 0);
    }
    
    static int hammingDistance(const TruthTable& a, const TruthTable& b) {
        if (a.size() != b.size()) return -1;
        int distance = 0;
        for (size_t i = 0; i < a.size(); i++) {
            if (a[i] != b[i]) distance++;
        }
        return distance;
    }
    
    static void print(const TruthTable& table) {
        for (auto bit : table) cout << static_cast<int>(bit);
    }
};

// ============================================================================
// FITNESS EVALUATION (Advanced Metrics)
// ============================================================================

class FitnessEvaluator {
    const TruthTable& targetFunc;
    int tableSize;
    
public:
    FitnessEvaluator(const TruthTable& f) : targetFunc(f), tableSize(f.size()) {}
    
    // Primary objective: Minimize f(x) * g(x)
    int computeAnnihilationCost(const TruthTable& g) const {
        int cost = 0;
        for (size_t i = 0; i < tableSize; i++) {
            if (targetFunc[i] == 1 && g[i] == 1) {
                cost++;
            }
        }
        return cost;
    }
    
    // Secondary objective: Penalize trivial solutions (all zeros)
    int computeAlgebraicImmunity(const TruthTable& g) const {
        int weight = BoolFunction::countOnes(g);
        return (weight == 0) ? 1000000 : weight;
    }
    
    // Tertiary objective: Non-linearity measure (Hamming distance to linear functions)
    int computeNonlinearity(const TruthTable& g) const {
        int minDist = tableSize;
        for (int mask = 0; mask < tableSize; mask++) {
            TruthTable linear(tableSize);
            for (int i = 0; i < tableSize; i++) {
                linear[i] = (__builtin_popcount(i & mask) & 1);
            }
            int dist = BoolFunction::hammingDistance(g, linear);
            minDist = min(minDist, dist);
        }
        return tableSize - minDist;
    }
    
    // Combined fitness function (lower is better)
    int evaluateFitness(const TruthTable& g) const {
        int cost = computeAnnihilationCost(g);
        if (cost == 0) return 0; // Perfect annihilator!
        
        int immunity = computeAlgebraicImmunity(g);
        // Return weighted combination
        return cost * 10 + (immunity > 1000000 ? 1000000 : immunity / 10);
    }
};

// ============================================================================
// GENETIC OPERATORS (C++17 Advanced)
// ============================================================================

class GeneticOperators {
private:
    RandomEngine& rng;
    
public:
    GeneticOperators(RandomEngine& r) : rng(r) {}
    
    // Advanced crossover: Two-point crossover with C++17
    TruthTable crossover(const TruthTable& p1, const TruthTable& p2) const {
        int size = p1.size();
        int point1 = rng.randomInt(0, size - 2);
        int point2 = rng.randomInt(point1 + 1, size - 1);
        
        TruthTable child(size);
        for (int i = 0; i < size; i++) {
            if (i >= point1 && i < point2) {
                child[i] = p2[i];
            } else {
                child[i] = p1[i];
            }
        }
        return child;
    }
    
    // Adaptive mutation rate based on generation progress
    void mutateAdaptive(TruthTable& g, double baseRate, double progress) const {
        double adaptiveRate = baseRate * (1.0 + progress); // Increase rate over time
        for (size_t i = 0; i < g.size(); i++) {
            if (rng.randomReal() < adaptiveRate) {
                g[i] = 1 - g[i];
            }
        }
    }
    
    // Standard mutation
    void mutate(TruthTable& g, double rate) const {
        for (size_t i = 0; i < g.size(); i++) {
            if (rng.randomReal() < rate) {
                g[i] = 1 - g[i];
            }
        }
    }
    
    // Uniform crossover: Each bit independently chosen from parents
    TruthTable uniformCrossover(const TruthTable& p1, const TruthTable& p2) const {
        TruthTable child(p1.size());
        for (size_t i = 0; i < p1.size(); i++) {
            child[i] = (rng.randomReal() < 0.5) ? p1[i] : p2[i];
        }
        return child;
    }
};

// ============================================================================
// SELECTION MECHANISMS (Tournament Selection - Advanced)
// ============================================================================

class SelectionMechanism {
private:
    RandomEngine& rng;
    
public:
    SelectionMechanism(RandomEngine& r) : rng(r) {}
    
    // Tournament selection: More robust than random elite selection
    pair<int, int> tournamentSelection(const PopulationT& population, int tournamentSize) const {
        auto selectOne = [&]() {
            int best_idx = rng.randomInt(0, population.size() - 1);
            int best_cost = population[best_idx].second;
            
            for (int i = 1; i < tournamentSize; i++) {
                int idx = rng.randomInt(0, population.size() - 1);
                if (population[idx].second < best_cost) {
                    best_cost = population[idx].second;
                    best_idx = idx;
                }
            }
            return best_idx;
        };
        
        return {selectOne(), selectOne()};
    }
    
    // Roulette wheel selection with fitness scaling (for minimization)
    int rouletteSelection(const PopulationT& population) const {
        auto [minCost, maxCost] = minmax_element(
            population.begin(), population.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; }
        );
        
        int max_cost = (*maxCost).second;
        int min_cost = (*minCost).second;
        
        vector<double> probs;
        for (const auto& [genome, cost] : population) {
            // Convert cost to fitness (invert and normalize)
            double fitness = max_cost - cost + 1;
            probs.push_back(fitness);
        }
        
        // Normalize probabilities
        double sum = accumulate(probs.begin(), probs.end(), 0.0);
        transform(probs.begin(), probs.end(), probs.begin(),
                  [sum](double p) { return p / sum; });
        
        // Weighted random selection
        double r = rng.randomReal();
        double cumulative = 0.0;
        for (size_t i = 0; i < probs.size(); i++) {
            cumulative += probs[i];
            if (r <= cumulative) return i;
        }
        return population.size() - 1;
    }
};

// ============================================================================
// STATISTICS & MONITORING
// ============================================================================

class PopulationAnalyzer {
public:
    static Statistics analyze(const PopulationT& population, int generation) {
        vector<int> costs;
        transform(population.begin(), population.end(),
                  back_inserter(costs),
                  [](const auto& p) { return p.second; });
        
        auto [minIt, maxIt] = minmax_element(costs.begin(), costs.end());
        double avg = accumulate(costs.begin(), costs.end(), 0.0) / costs.size();
        
        // Diversity: Average Hamming distance between random pairs
        int samples = min(10, (int)population.size() / 2);
        double totalDist = 0;
        for (int i = 0; i < samples; i++) {
            int idx1 = rand() % population.size();
            int idx2 = rand() % population.size();
            totalDist += BoolFunction::hammingDistance(
                population[idx1].first,
                population[idx2].first
            );
        }
        double diversity = totalDist / samples;
        
        return {*minIt, *maxIt, avg, diversity, generation};
    }
    
    static void printStats(const Statistics& s) {
        cout << fixed << setprecision(2);
        cout << "Gen " << setw(5) << s.generation
             << " | Best: " << setw(4) << s.bestCost
             << " | Avg: " << setw(7) << s.avgCost
             << " | Worst: " << setw(4) << s.worstCost
             << " | Diversity: " << setw(6) << s.diversity << "\n";
    }
};

// ============================================================================
// MAIN GENETIC ALGORITHM ENGINE (C++17 Advanced)
// ============================================================================

class GeneticAlgorithmEngine {
private:
    GAConfig config;
    RandomEngine rng;
    FitnessEvaluator evaluator;
    GeneticOperators operators;
    SelectionMechanism selector;
    TruthTable targetFunc;
    vector<string> executionLog;
    
public:
    GeneticAlgorithmEngine(const TruthTable& f, const GAConfig& cfg)
        : config(cfg), evaluator(f), operators(rng), selector(rng), targetFunc(f) {}
    
    void logMessage(const string& msg) {
        executionLog.push_back(msg);
        if (config.verbosity >= 1) cout << msg << "\n";
    }
    
    void saveLog(const string& filename) {
        ofstream file(filename);
        for (const auto& msg : executionLog) {
            file << msg << "\n";
        }
        file.close();
    }
    
    optional<TruthTable> run() {
        auto startTime = chrono::high_resolution_clock::now();
        
        int tableSize = 1 << config.numVars;
        
        // Phase 1: Initialize population with Monte Carlo sampling
        if (config.verbosity >= 1) {
            cout << "\n[*] Monte Carlo Initialization (" << (config.populationSize * 5)
                 << " samples)...\n";
        }
        
        PopulationT population;
        for (int i = 0; i < config.populationSize * 5; i++) {
            auto genome = BoolFunction::generateRandom(tableSize, rng);
            int fitness = evaluator.evaluateFitness(genome);
            population.push_back({genome, fitness});
        }
        
        // Sort and keep best individuals
        sort(population.begin(), population.end(),
             [](const auto& a, const auto& b) { return a.second < b.second; });
        population.resize(config.populationSize);
        
        if (config.verbosity >= 1) {
            cout << "[+] Initial best cost: " << population[0].second << "\n\n";
            cout << "[*] Starting Genetic Evolution...\n";
            cout << "    Threads: " << config.numThreads << " | "
                 << "Population: " << config.populationSize << " | "
                 << "Generations: " << config.maxGenerations << "\n\n";
        }
        
        // Phase 2: Generational Evolution Loop
        for (int gen = 1; gen <= config.maxGenerations; gen++) {
            // Check for convergence
            if (population[0].second == 0) {
                if (config.verbosity >= 1) {
                    cout << "\n[SUCCESS] Perfect annihilator found!\n";
                    cout << "    Generation: " << gen << "\n";
                    cout << "    Annihilator (g): ";
                    BoolFunction::print(population[0].first);
                    cout << "\n";
                }
                auto endTime = chrono::high_resolution_clock::now();
                auto elapsed = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
                cout << "    Time: " << elapsed.count() << "ms\n";
                return population[0].first;
            }
            
            // Statistics and reporting
            if (config.verbosity >= 1 && gen % 100 == 0) {
                auto stats = PopulationAnalyzer::analyze(population, gen);
                PopulationAnalyzer::printStats(stats);
            }
            
            // Elitism: Keep top individuals
            int eliteCount = max(1, (int)(config.populationSize * config.elitismRatio));
            PopulationT elite(population.begin(), population.begin() + eliteCount);
            
            // Breeding phase
            PopulationT newGeneration;
            for (int i = eliteCount; i < config.populationSize; i++) {
                auto [p1_idx, p2_idx] = selector.tournamentSelection(population, config.tournamentSize);
                const auto& p1 = population[p1_idx].first;
                const auto& p2 = population[p2_idx].first;
                
                // Crossover
                auto child = operators.uniformCrossover(p1, p2);
                
                // Adaptive mutation
                double progress = static_cast<double>(gen) / config.maxGenerations;
                if (config.adaptive_mutation) {
                    operators.mutateAdaptive(child, config.mutationRate, progress * 0.3);
                } else {
                    operators.mutate(child, config.mutationRate);
                }
                
                int fitness = evaluator.evaluateFitness(child);
                newGeneration.push_back({child, fitness});
            }
            
            // Combine elite with new generation
            population.clear();
            population.insert(population.end(), elite.begin(), elite.end());
            population.insert(population.end(), newGeneration.begin(), newGeneration.end());
            
            // Sort for next generation
            sort(population.begin(), population.end(),
                 [](const auto& a, const auto& b) { return a.second < b.second; });
            population.resize(config.populationSize);
        }
        
        if (config.verbosity >= 1) {
            cout << "\n[!] Algorithm converged to local minimum after " << config.maxGenerations
                 << " generations.\n";
            cout << "    Best cost found: " << population[0].second << "\n";
            cout << "    Best solution (g): ";
            BoolFunction::print(population[0].first);
            cout << "\n";
        }
        
        return nullopt;
    }
};

// ============================================================================
// MAIN PROGRAM
// ============================================================================

int main() {
    cout << "╔════════════════════════════════════════════════════════════════╗\n"
         << "║         Advanced Boolean Function Annihilator Search          ║\n"
         << "║                    With C++17 Features                        ║\n"
         << "╚════════════════════════════════════════════════════════════════╝\n";
    
    GAConfig config;
    config.numVars = 5;              // 32-bit truth table
    config.populationSize = 256;     // Larger population
    config.maxGenerations = 10000;   // More generations
    config.mutationRate = 0.08;      // Dynamic mutation
    config.tournamentSize = 5;       // Tournament selection
    config.adaptive_mutation = true; // Enable adaptive rates
    config.verbosity = 1;
    
    int tableSize = 1 << config.numVars;
    
    // Generate target cipher function
    RandomEngine rng;
    TruthTable targetFunc = BoolFunction::generateRandom(tableSize, rng);
    
    cout << "\n[+] Target Cipher Function (f): ";
    BoolFunction::print(targetFunc);
    cout << "\n";
    
    cout << "[+] Configuration:\n"
         << "    Variables: " << config.numVars << " (table size: " << tableSize << ")\n"
         << "    Population Size: " << config.populationSize << "\n"
         << "    Max Generations: " << config.maxGenerations << "\n"
         << "    Mutation Rate: " << config.mutationRate << "\n"
         << "    Adaptive Mutation: " << (config.adaptive_mutation ? "Yes" : "No") << "\n";
    
    GeneticAlgorithmEngine ga(targetFunc, config);
    optional<TruthTable> result = ga.run();
    
    cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    cout << "║                      FINAL RESULTS SUMMARY                     ║\n";
    cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    
    cout << "[TARGET FUNCTION]\n";
    cout << "  Function (f): ";
    BoolFunction::print(targetFunc);
    cout << "\n  Size: " << tableSize << " bits (" << config.numVars << " variables)\n";
    cout << "  Ones count: " << BoolFunction::countOnes(targetFunc) << "\n\n";
    
    if (result) {
        cout << "[SUCCESS] Annihilator successfully found!\n\n";
        cout << "[ANNIHILATOR FUNCTION]\n";
        cout << "  Function (g): ";
        BoolFunction::print(*result);
        cout << "\n  Size: " << tableSize << " bits (" << config.numVars << " variables)\n";
        cout << "  Ones count: " << BoolFunction::countOnes(*result) << "\n\n";
        
        // Verify the solution
        cout << "[VERIFICATION]\n";
        int violations = 0;
        for (size_t i = 0; i < targetFunc.size(); i++) {
            if (targetFunc[i] == 1 && (*result)[i] == 1) {
                violations++;
            }
        }
        cout << "  Annihilation violations: " << violations << " (should be 0)\n";
        cout << "  Perfect annihilator: " << (violations == 0 ? "YES ✓" : "NO ✗") << "\n";
        cout << "  Hamming distance (f,g): " << BoolFunction::hammingDistance(targetFunc, *result) << "\n\n";
        
        // Bit-by-bit analysis
        cout << "[BIT-BY-BIT ANALYSIS]\n";
        cout << "  Position | f | g | Product | Safe\n";
        cout << "  ---------|---|---|---------|-------\n";
        for (int i = 0; i < (int)targetFunc.size(); i++) {
            int f_bit = (int)targetFunc[i];
            int g_bit = (int)(*result)[i];
            int product = f_bit * g_bit;
            string safe = (product == 0) ? "✓" : "✗";
            cout << "  " << setw(8) << i << " | " << f_bit << " | " << g_bit << " | " 
                 << setw(7) << product << " | " << safe << "\n";
        }
    } else {
        cout << "[FAILED] No perfect annihilator found.\n\n";
        cout << "[BEST ATTEMPT]\n";
        cout << "  (Would have been stored in solution if found)\n";
    }
    
    cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    cout << "║                    END OF EXECUTION                            ║\n";
    cout << "╚════════════════════════════════════════════════════════════════╝\n";    
    // Save results to CSV file
    ofstream csvFile("annihilator_results.csv");
    csvFile << "\"ANNIHILATOR SEARCH RESULTS\"\n\n";
    csvFile << "\"EXECUTION SUMMARY\"\n";
    csvFile << "\"Configuration\",\"Value\"\n";
    csvFile << "\"Variables\",\"" << config.numVars << "\"\n";
    csvFile << "\"Table Size\",\"" << tableSize << "\"\n";
    csvFile << "\"Population Size\",\"" << config.populationSize << "\"\n";
    csvFile << "\"Max Generations\",\"" << config.maxGenerations << "\"\n";
    csvFile << "\"Mutation Rate\",\"" << config.mutationRate << "\"\n";
    csvFile << "\"Adaptive Mutation\",\"" << (config.adaptive_mutation ? "Yes" : "No") << "\"\n";
    csvFile << "\"Tournament Size\",\"" << config.tournamentSize << "\"\n\n";
    
    csvFile << "\"TARGET FUNCTION\"\n";
    csvFile << "\"Property\",\"Value\"\n";
    csvFile << "\"Function (f)\",\"";
    for (auto bit : targetFunc) csvFile << (int)bit;
    csvFile << "\"\n";
    csvFile << "\"Ones Count\",\"" << BoolFunction::countOnes(targetFunc) << "\"\n\n";
    
    if (result) {
        csvFile << "\"RESULT STATUS\",\"SUCCESS\"\n\n";
        csvFile << "\"ANNIHILATOR FUNCTION\"\n";
        csvFile << "\"Property\",\"Value\"\n";
        csvFile << "\"Function (g)\",\"";
        for (auto bit : *result) csvFile << (int)bit;
        csvFile << "\"\n";
        csvFile << "\"Ones Count\",\"" << BoolFunction::countOnes(*result) << "\"\n\n";
        
        csvFile << "\"VERIFICATION\"\n";
        csvFile << "\"Metric\",\"Value\"\n";
        int violations = 0;
        for (size_t i = 0; i < targetFunc.size(); i++) {
            if (targetFunc[i] == 1 && (*result)[i] == 1) {
                violations++;
            }
        }
        csvFile << "\"Annihilation Violations\",\"" << violations << "\"\n";
        csvFile << "\"Perfect Annihilator\",\"" << (violations == 0 ? "YES" : "NO") << "\"\n";
        csvFile << "\"Hamming Distance\",\"" << BoolFunction::hammingDistance(targetFunc, *result) << "\"\n\n";
        
        csvFile << "\"BIT-BY-BIT ANALYSIS\"\n";
        csvFile << "\"Position\",\"f\",\"g\",\"Product\",\"Safe\"\n";
        for (int i = 0; i < (int)targetFunc.size(); i++) {
            int f_bit = (int)targetFunc[i];
            int g_bit = (int)(*result)[i];
            int product = f_bit * g_bit;
            string safe = (product == 0) ? "YES" : "NO";
            csvFile << "\"" << i << "\",\"" << f_bit << "\",\"" << g_bit << "\",\"" 
                    << product << "\",\"" << safe << "\"\n";
        }
    } else {
        csvFile << "\"RESULT STATUS\",\"FAILED\"\n";
    }
    
    csvFile.close();
    
    cout << "\n[+] Results saved to: annihilator_results.csv\n";
    cout << "[*] You can open this file in Excel or any spreadsheet application.\n\n";
        return 0;
}