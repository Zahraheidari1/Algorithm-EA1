# Evolutionary Computation — Homework 1

> Implementation and analysis of an Evolutionary Algorithm (EA1) across four classic optimization problems, exploring the effect of population size, crossover probability, mutation probability, and selection strategy on algorithm performance.

---

## Table of Contents

- [Overview](#overview)
- [Algorithm Framework (EA1)](#algorithm-framework-ea1)
- [Problems Implemented](#problems-implemented)
  - [1. Binary Benchmark Problems](#1-binary-benchmark-problems)
  - [2. Graph Coloring](#2-graph-coloring)
  - [3. Traveling Salesman Problem (TSP)](#3-traveling-salesman-problem-tsp)
  - [4. Ackley Function Optimization](#4-ackley-function-optimization)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Key Results & Findings](#key-results--findings)
- [Technologies Used](#technologies-used)

---

## Overview

This project implements a unified Evolutionary Algorithm (EA1) framework and applies it to four fundamentally different optimization problems. The study systematically investigates how key EA parameters — selection method, recombination operator, mutation operator, population size, and crossover/mutation probabilities — affect convergence speed, solution quality, and population diversity. Each problem uses a representation and operators tailored to its structure.

---

## Algorithm Framework (EA1)

The core EA1 framework is defined as follows:

```
Algorithm: EA1
 1. pop       = Generate popSize initial candidate solutions of size problemSize
 2. popFit    = Evaluate pop using f(x)
 3. While not Terminate():
 4.   parentsPool  = Select popSize solutions from pop using popFit
 5.   parentPairs  = Shuffle parentsPool and randomly pair solutions
 6.   offspr       = Perform Recombination on parentPairs with Pc
 7.   offspr       = Perform Mutation on offspr with Pm
 8.   offsprFit    = Evaluate offspr using f(x)
 9.   [pop, popFit] = Select best popSize solutions from [pop + offspr]
10. Return best solution in pop
```

**Termination condition:** either the optimal solution is found, or the number of generations reaches **300**.

**Survivor selection:** elitist — the top `popSize` individuals from the combined parent + offspring pool are kept.

---

## Problems Implemented

### 1. Binary Benchmark Problems

**Representation:** Binary strings (bit vectors)  
**Recombination:** Uniform crossover  
**Mutation:** Bit-flip mutation

Three pseudo-Boolean fitness functions `f: {0,1}ⁿ → ℝ` are studied:

| Function | Formula | Optimal Solution |
|----------|---------|-----------------|
| **OneMax** | `Σ xᵢ` | All-ones string |
| **Peak** | `Π xᵢ` | All-ones string |
| **Trap** | `3 × problemSize × Π xᵢ − Σ xᵢ` | All-zeros or all-ones |

**Experiments conducted:**
- Evolution curves (best / worst / average fitness per generation)
- Statistical analysis over ≥5 independent runs (mean ± std dev)
- Scalability analysis: problem sizes `{10, 30, 50, 100}`
- Population size sweep: `{50, 100, 200, 300}`
- Crossover probability sweep: `{0.5, 0.7, 0.9, 1.0}`
- Mutation probability sweep: `{0.05, 0.1, 0.3, 0.5}`

---

### 2. Graph Coloring

**Representation:** Integer vector (each gene = color assigned to a vertex)  
**Recombination:** Two-point crossover  
**Mutation:** Random-value mutation  
**Dataset:** Queen graph instances from DIMACS (`queen5_5`, `queen6_6`, `queen7_7`, `queen8_12`)

The objective is to minimize the number of colors used while ensuring no two adjacent vertices share the same color.

**Fitness function:** Balances a hard constraint (no adjacent same-color vertices) against a soft constraint (minimize total colors used).

| Instance | Vertices | Chromatic Number |
|----------|----------|-----------------|
| queen5_5 | 25 | 5 |
| queen6_6 | 36 | 7 |
| queen7_7 | 49 | 7 |
| queen8_12 | 96 | 12 |

**Library used:** [`networkx`](https://networkx.org/)

---

### 3. Traveling Salesman Problem (TSP)

**Representation:** Permutation encoding (city tour order)  
**Recombination:** Order crossover (OX)  
**Mutation:** Swap mutation  
**Dataset:** TSPLIB instances (`burma14`, `bayg29`, `dantzig42`, `gr120`)

The goal is to find the minimum-cost Hamiltonian cycle visiting all cities exactly once.

| Instance | Cities | Best Known Cost |
|----------|--------|----------------|
| burma14 | 14 | 3323 |
| bayg29 | 29 | 1610 |
| dantzig42 | 42 | 699 |
| gr120 | 120 | 6942 |

**Library used:** [`tsplib95`](https://tsplib95.readthedocs.io/) (for reading `.tsp` files only; all search is done by EA1)

---

### 4. Ackley Function Optimization

**Representation:** Real-valued vectors  
**Recombination:** Simulated Binary Crossover (SBX)  
**Mutation:** Polynomial mutation (non-uniform)  
**Search space:** `[-5, 5]ᵈ`

The Ackley function is a well-known non-convex benchmark with a single global minimum at the origin and many local minima, making it challenging for optimization algorithms:

```
f(x) = -a·exp(-b·√(1/d · Σxᵢ²)) - exp(1/d · Σcos(cxᵢ)) + a + exp(1)
```

where `a = 20`, `b = 0.2`, `c = 2π`.

**Parameter sweep:**
- Population size: `{50, 100, 200, 300}`
- Crossover probability: `{0.3, 0.5, 0.7, 0.9}`
- Mutation step size (η): `{0.1, 0.5, 1, 2, 4, 20}`

**Bonus — Fitness Sharing:** A diversity-preservation mechanism is implemented to locate multiple optima simultaneously. Shared fitness is computed as:

```
f'(i) = f(i) / Σⱼ sh(d(i,j))

       ⎧ 1 - (d/σ)^α    if d ≤ σ
sh(d) = ⎨
       ⎩ 0               otherwise
```

Applied to multimodal benchmarks: **Himmelblau's function**, **Ackley function**, and **Holder Table function**.

---

## Project Structure

```
HW1/
├── src/
│   └── genetic.py            # Core EA1 implementation (selection, crossover, mutation)
├── documents/
│   ├── binrary.ipynb         # Binary benchmark experiments (OneMax, Peak, Trap)
│   ├── graph_coloring.ipynb  # Graph coloring experiments
│   ├── tsp.ipynb             # TSP experiments
│   └── ackley.ipynb          # Ackley function & fitness sharing experiments
├── data/
│   ├── bayg29.tsp            # TSP instance: 29 cities
│   ├── burma14.tsp           # TSP instance: 14 cities
│   ├── dantzig42.tsp         # TSP instance: 42 cities
│   ├── gr120.tsp             # TSP instance: 120 cities
│   ├── queen5_5.col          # Graph coloring: 25 vertices
│   ├── queen6_6.col          # Graph coloring: 36 vertices
│   ├── queen7_7.col          # Graph coloring: 49 vertices
│   └── queen8_12.col         # Graph coloring: 96 vertices
└── EC_HW1_Zahra_Heidari.pdf  # Full report with results and analysis
```

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/Zahraheidari1/Algorithm-EA1.git
cd <Algorithm-EA1>

# Install dependencies
pip install numpy matplotlib networkx tsplib95 jupyter
```

---

## Usage

Run each experiment notebook from the `documents/` directory:

```bash
jupyter notebook documents/binrary.ipynb        # Binary problems
jupyter notebook documents/graph_coloring.ipynb  # Graph coloring
jupyter notebook documents/tsp.ipynb             # TSP
jupyter notebook documents/ackley.ipynb          # Ackley + fitness sharing
```

Or import the core EA directly:

```python
from src.genetic import EA1

best_solution = EA1(
    pop_size=100,
    problem_size=30,
    crossover_prob=0.9,
    mutation_prob=0.1,
    fitness_fn=your_fitness_function,
    max_generations=300
)
```

---

## Key Results & Findings

- **Selection pressure** strongly affects convergence speed; fitness-proportionate selection risks premature convergence on deceptive functions (e.g., Trap).
- **Population size** trades off exploration breadth vs. computational cost — larger populations benefit high-dimensional TSP instances more than simple binary problems.
- **Mutation probability** has a sweet spot: too low causes stagnation, too high destroys useful building blocks.
- **Crossover operator choice** is problem-dependent: OX preserves relative city ordering in TSP; SBX enables fine-grained real-valued exploration in Ackley.
- **Fitness sharing** successfully maintains population diversity on multimodal functions, recovering multiple global/local optima in a single run.

---

## Technologies Used

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c)
![NetworkX](https://img.shields.io/badge/NetworkX-graph--library-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

- **Python 3.11+**
- **NumPy** — vectorized fitness evaluation and array operations
- **Matplotlib** — evolution curves, scatter plots, convergence analysis
- **NetworkX** — graph construction and manipulation (Graph Coloring & TSP)
- **tsplib95** — parsing standard TSPLIB benchmark files
- **Jupyter Notebook** — reproducible experiment notebooks

---

*Course: Evolutionary Computation — Isfahan University of Technology, AI & Robotics Group, Fall 2024*  
*Instructor: Dr. Hossein Karshenas | TA: Mehrnoush Alipour*
