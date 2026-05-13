import numpy as np
import random
import tsplib95
import networkx as nx
import matplotlib.pyplot as plt

def fitness_function(x):
    return np.sum(x)

def fitness_function_peak(x):
    return np.prod(x)

def fitness_function_trap(x):
    return 3 * x.size * np.prod(x) - np.sum(x)

def evaluate_population(pop,fitness_type = 'OneMax'):
    if fitness_type == 'Peak':
        return np.array([fitness_function_peak(ind) for ind in pop])
    elif fitness_type == 'Trap':
        return np.array([fitness_function_trap(ind) for ind in pop])
    
    return np.array([fitness_function(ind) for ind in pop])

def generate_population(pop_size, problem_size):
    return np.random.randint(2, size=(pop_size, problem_size))

def roulette_wheel_selection(pop, pop_fit, select_highest=True):
    if select_highest:
        total = np.sum(pop_fit - np.min(pop_fit))
        if total == 0:
            probability = np.ones(len(pop)) / len(pop)
        else:
            probability = (pop_fit - np.min(pop_fit)) / total
    else:
        total = np.sum(np.max(pop_fit) - pop_fit)  
        if total == 0:
            probability = np.ones(len(pop)) / len(pop)
        else:
            probability = (np.max(pop_fit) - pop_fit) / total
    
    selected_indices = np.random.choice(range(len(pop)), size=len(pop), p=probability, replace=True)
    return pop[selected_indices]

def tournament_selection(pop, pop_fit, tournament_size=2, maximize=True):
    selected_parents = []
    
    for _ in range(len(pop)):
        competitors_idx = np.random.choice(range(len(pop)), tournament_size, replace=False)
        competitors_fit = pop_fit[competitors_idx]
        
        if maximize:
            winner_idx = competitors_idx[np.argmax(competitors_fit)]  
        else:
            winner_idx = competitors_idx[np.argmin(competitors_fit)] 
        
        selected_parents.append(pop[winner_idx])
    
    return np.array(selected_parents)

def crossover_uniform(parents, crossover_prob):
    offspring = []
    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        parent2 = parents[(i + 1) % len(parents)]
        if random.random() < crossover_prob:
            child1 = parent1.copy()
            child2 = parent2.copy()
            for j in range(len(parent1)):
                if random.random() < 0.5:  
                    child1[j] = parent2[j]
                    child2[j] = parent1[j]
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        offspring.extend([child1, child2])
    return np.array(offspring)

def crossover_uniform_population(population, crossover_prob):
    num_parents, genome_length = population.shape
    offspring = population.copy()
    
    for j in range(genome_length):
        if random.random() < crossover_prob:
            for i in range(0, num_parents, 2):
                if i + 1 < num_parents:
                    if random.random() < 0.5:
                        offspring[i, j], offspring[i + 1, j] = offspring[i + 1, j], offspring[i, j]
    
    return offspring

def crossover_two_point(parents, crossover_prob):
    offspring = []
    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        parent2 = parents[(i + 1) % len(parents)]
        
        if random.random() < crossover_prob:
            point1 = random.randint(1, len(parent1) - 2)
            point2 = random.randint(point1 + 1, len(parent1) - 1) 
            
            child1 = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
            child2 = np.concatenate((parent2[:point1], parent1[point1:point2], parent2[point2:]))
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        
        offspring.extend([child1, child2]) 
    
    return np.array(offspring)

def mutation_change_bit(offspring, mutation_prob):
    for i in range(len(offspring)):
        for j in range(len(offspring[i])):
            if random.random() < mutation_prob:
                offspring[i][j] = 1 - offspring[i][j]
    return offspring

def mu_plus_lambda_selection(pop, offspring, pop_fit, offspring_fit, pop_size, select_highest=True):
    combined_pop = np.vstack((pop, offspring))
    combined_fit = np.concatenate((pop_fit, offspring_fit))
    
    if select_highest:
        best_indices = np.argsort(combined_fit)[-pop_size:] 
    else:
        best_indices = np.argsort(combined_fit)[:pop_size]   
    
    next_pop = combined_pop[best_indices]
    next_pop_fit = combined_fit[best_indices]
    
    return next_pop, next_pop_fit

def genetic_binrary(pop_size, problem_size, crossover_prob, mutation_prob, max_generations, solfitness = None, fitness_type = 'OneMax', selection_type = 'Roulette', crossover_type = "Uniform"):
    pop = generate_population(pop_size, problem_size)
    pop_fit = evaluate_population(pop,fitness_type)
    
    best_fitness_values = []
    worst_fitness_values = []
    avg_fitness_values = []

    best_solution = pop[np.argmax(pop_fit)]
    best_generation = 0

    best_fitness = np.max(pop_fit)
    worst_fitness = np.min(pop_fit)
    avg_fitness = np.mean(pop_fit)
    best_fitness_values.append(best_fitness)
    worst_fitness_values.append(worst_fitness)
    avg_fitness_values.append(avg_fitness)

    generation = 0
    

    while generation < max_generations and (solfitness is None or (solfitness is not None and best_fitness < solfitness) ):
        parents = []

        if selection_type == 'Tournament':
            parents = tournament_selection(pop, pop_fit)
        else:
            parents = roulette_wheel_selection(pop, pop_fit)

        np.random.shuffle(parents)
        
        if crossover_type == "PopulationWiseUniform":
            offspring = crossover_uniform_population(parents,crossover_prob)
        else:
            offspring = crossover_uniform(parents,crossover_prob)

        offspring = mutation_change_bit(offspring, mutation_prob)
        offspring_fit = evaluate_population(offspring,fitness_type)

        pop,pop_fit = mu_plus_lambda_selection(pop,offspring,pop_fit,offspring_fit,pop_size)
        
        current_best_fitness = np.max(pop_fit)
        worst_fitness = np.min(pop_fit)
        avg_fitness = np.mean(pop_fit)
        
        best_fitness_values.append(current_best_fitness)
        worst_fitness_values.append(worst_fitness)
        avg_fitness_values.append(avg_fitness)
        
    
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = pop[np.argmax(pop_fit)]
            best_generation = generation + 1
        
        generation += 1
    
    return best_fitness_values, worst_fitness_values, avg_fitness_values, best_solution, best_generation

def load_tsp_data(file_path):
    problem = tsplib95.load(file_path)
    nodes = list(problem.get_nodes())

    node_coords = problem.node_coords

    if node_coords is None or len(node_coords) == 0 or node_coords ==[] or node_coords == {}:
        node_coords = {}
        with open(file_path, 'r') as file:
            lines = file.readlines()
            capture = False
            for line in lines:
                if line.strip() == "NODE_COORD_SECTION" or line.strip() =="DISPLAY_DATA_SECTION":
                    capture = True
                elif line.strip() == "EOF":
                    capture = False
                    break
                elif capture:
                    parts = line.split()
                    node_id = int(parts[0])
                    x_coord = float(parts[1])
                    y_coord = float(parts[2])
                    node_coords[node_id] = (x_coord, y_coord)

    has_weights = problem.edge_weights is not None

    num_nodes = len(nodes)
    distance_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if has_weights:
                distance = problem.get_weight(nodes[i], nodes[j])
            else:
                coord1 = node_coords[nodes[i]]
                coord2 = node_coords[nodes[j]]
                distance = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

    return distance_matrix, node_coords

def create_tsp_graph(best_solution, distances, node_coords):
    G = nx.Graph()

    for node_id in best_solution:
        coord = node_coords[node_id + 1] 
        G.add_node(node_id, pos=(coord[0], coord[1]))

    for i in range(len(best_solution)):
        start = best_solution[i]
        end = best_solution[(i + 1) % len(best_solution)]
        distance = distances[start][end]
        G.add_edge(start, end, weight=distance)

    return G

def draw_tsp_graph(G):
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Best TSP Solution Path")
    plt.show()

def fitness_function_tsp(route, distances):
    return np.sum([distances[route[i - 1], route[i]] for i in range(len(route))])

def evaluate_population_tsp(pop, distances):
    return np.array([fitness_function_tsp(ind, distances) for ind in pop])

def generate_population_tsp(pop_size, num_cities):
    return np.array([random.sample(range(num_cities), num_cities) for _ in range(pop_size)])

def crossover_ordered(parent1, parent2):
    size = len(parent1)
    child = [-1] * size

    start, end = sorted(random.sample(range(size), 2))
    child[start:end] = parent1[start:end]

    idx = end
    for gene in parent2:
        if gene not in child:
            if idx >= size:
                idx = 0
            child[idx] = gene
            idx += 1

    return np.array(child)

def mutation_swap(route, mutation_prob):
    for i in range(len(route)):
        if random.random() < mutation_prob:
            j = random.randint(0, len(route) - 1)
            route[i], route[j] = route[j], route[i]
    return route

def genetic_tsp(filename, pop_size, crossover_prob, mutation_prob, max_generations, solfitness = None, selection_type = 'Roulette'):
    distances, nodes = load_tsp_data(filename)
    num_cities = len(nodes)
    pop = generate_population_tsp(pop_size, num_cities)
    pop_fit = evaluate_population_tsp(pop, distances)

    best_fitness_values = []
    worst_fitness_values = []
    avg_fitness_values = []

    best_fitness = np.min(pop_fit)
    best_solution = pop[np.argmin(pop_fit)]
    best_generation = 0  

    best_fitness_values.append(best_fitness)
    worst_fitness_values.append(np.max(pop_fit))
    avg_fitness_values.append(np.mean(pop_fit))

    generation = 0

    while generation < max_generations and (solfitness is None or (solfitness is not None and best_fitness > solfitness) ):
        parents=[]
        if selection_type == 'Tournament':
            parents = tournament_selection(pop,pop_fit,maximize=False)
        else:
            parents = roulette_wheel_selection(pop, pop_fit,select_highest=False)
        np.random.shuffle(pop)
        offspring = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[(i + 1) % len(parents)]
            
            if random.random() < crossover_prob:
                child1 = crossover_ordered(parent1, parent2)
                child2 = crossover_ordered(parent2, parent1)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1.copy(), parent2.copy()])
        offspring = np.array(offspring)

        offspring = np.array([mutation_swap(ind, mutation_prob) for ind in offspring])
        offspring_fit = evaluate_population_tsp(offspring, distances)

        pop,pop_fit = mu_plus_lambda_selection(pop,offspring,pop_fit,offspring_fit,pop_size,select_highest=False)

        current_best_fitness = np.min(pop_fit)
        worst_fitness_values.append(np.max(pop_fit))
        avg_fitness_values.append(np.mean(pop_fit))
        best_fitness_values.append(current_best_fitness)

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = pop[np.argmin(pop_fit)]
            best_generation = generation + 1

        generation += 1

    G = create_tsp_graph(best_solution,distances,nodes)

    return best_fitness_values, worst_fitness_values, avg_fitness_values, best_solution, best_generation, G

def read_dimacs_graph(file_path):
    """
    Reads a DIMACS formatted graph from a file and returns it as an adjacency list.
    
    Args:
        file_path (str): The path to the DIMACS file.
        
    Returns:
        dict: A dictionary representing the adjacency list of the graph.
    """
    graph = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # Skip comments
            if line.startswith('c'):
                continue
            # Read the problem line
            if line.startswith('p'):
                parts = line.split()
                num_vertices = int(parts[2])
                # Initialize the adjacency list
                graph = {i: [] for i in range(1, num_vertices + 1)}
            # Read edges
            elif line.startswith('e'):
                parts = line.split()
                u, v = int(parts[1]), int(parts[2])
                graph[u].append(v)

    return graph

def fitness_function_coloring(solution, G):
    num_colors = len(set(solution))
    conflicts = sum(1 for u, v in G.edges if solution[u - 1] == solution[v - 1])
    return num_colors + ( conflicts ** 2 ) * 10

def fitness_function_coloring2(solution, G):
    num_colors = len(set(solution))
    conflicts = sum(1 for u, v in G.edges if solution[u - 1] == solution[v - 1])
    return num_colors + conflicts * 10

def fitness_function_coloring3(solution, G):
    num_colors = len(set(solution))
    conflicts = sum(1 for u, v in G.edges if solution[u - 1] == solution[v - 1])
    return num_colors*0.1 + conflicts * 10

def fitness_function_coloring4(solution, G):
    num_colors = len(set(solution))
    conflicts = sum(1 for u, v in G.edges if solution[u - 1] == solution[v - 1])
    return (num_colors**4)*0.1  + ( conflicts ** 4 ) * 20

def evaluate_population_coloring(pop, G):
    return np.array([fitness_function_coloring4(ind, G) for ind in pop])

def generate_population_coloring(pop_size, num_vertices, num_colors):
    return np.array([np.random.randint(num_colors, size=num_vertices) for _ in range(pop_size)])

def mutate_coloring(offspring, mutation_prob, num_colors):
    for i in range(len(offspring)):
        for j in range(len(offspring[i])):
            if random.random() < mutation_prob:
                offspring[i][j] = random.randint(0, num_colors - 1)
    return offspring

def genetic_coloring(filename, pop_size, num_colors, crossover_prob, mutation_prob, max_generations, solfitness = None, selection_type = 'Roulette'):
    graph = read_dimacs_graph(filename)
    G = nx.Graph()
    G.add_nodes_from(graph.keys())
    count = 0
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            G.add_edge(node,neighbor)
            count+=1

    num_vertices = G.number_of_nodes()
    pop = generate_population_coloring(pop_size, num_vertices, num_colors)
    pop_fit = evaluate_population_coloring(pop, G)
    
    best_fitness_values, worst_fitness_values, avg_fitness_values = [], [], []
    best_solution = pop[np.argmin(pop_fit)]
    best_fitness = np.min(pop_fit)
    best_generation = 0
    best_fitness_values.append(best_fitness)
    worst_fitness_values.append(np.max(pop_fit))
    avg_fitness_values.append(np.mean(pop_fit))

    generation = 0

    while generation < max_generations and (solfitness is None or (solfitness is not None and best_fitness > solfitness) ):
        if selection_type == 'Tournament':
            parents = tournament_selection(pop,pop_fit,maximize=False)
        else:
            parents = roulette_wheel_selection(pop, pop_fit,select_highest=False)
        np.random.shuffle(parents)
        offspring = crossover_two_point(parents, crossover_prob)
        offspring = mutate_coloring(offspring, mutation_prob, num_colors)
        offspring_fit = evaluate_population_coloring(offspring, G)
        
        pop,pop_fit = mu_plus_lambda_selection(pop,offspring,pop_fit,offspring_fit,pop_size,select_highest=False)
        
        current_best_fitness = np.min(pop_fit)
        best_fitness_values.append(current_best_fitness)
        worst_fitness_values.append(np.max(pop_fit))
        avg_fitness_values.append(np.mean(pop_fit))

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = pop[np.argmin(pop_fit)]
            best_generation = generation + 1

        generation += 1

    return best_fitness_values, worst_fitness_values, avg_fitness_values, best_solution, best_generation, G

def plot_colored_graph(G, colors):
    pos = nx.spring_layout(G) 
    color_map = [colors[node - 1] for node in G.nodes()]
    nx.draw(G, pos, node_color=color_map, with_labels=True, node_size=500, cmap=plt.cm.tab20)
    plt.show()

def ackley_function(x, a=20, b=0.2, c=2 * np.pi):
    d = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)

def initialize_population_ackley(pop_size, dim, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, (pop_size, dim))

def simulated_binary_crossover(parent1, parent2, crossover_prob, lower_bound, upper_bound, beta=None):
    if np.random.rand() > crossover_prob:
        return parent1, parent2
    if beta is None:
        beta = np.random.beta(2, 2)
    child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
    child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
    return np.clip(child1, lower_bound, upper_bound), np.clip(child2, lower_bound, upper_bound)

def non_uniform_mutation(individual, mutation_prob, mutation_step_size, lower_bound, upper_bound):
    for i in range(len(individual)):
        if np.random.rand() < mutation_prob:
            individual[i] += mutation_step_size * np.random.normal()
            individual[i] = np.clip(individual[i], lower_bound, upper_bound)
    return individual

def genetic_ackley(dim, pop_size=300, crossover_prob=0.9, mutation_prob=1, mutation_step_size=20, max_generations=1000, solfitness = None, selection_type = 'Roulette'):
    lower_bound, upper_bound = -5, 5
    population = initialize_population_ackley(pop_size, dim, lower_bound, upper_bound)
    fitness = np.array([ackley_function(ind) for ind in population])

    best_fitness_values, worst_fitness_values, avg_fitness_values = [], [], []
    best_solution = population[np.argmin(fitness)]
    best_fitness = np.min(fitness)
    best_generation = 0
    best_fitness_values.append(best_fitness)
    worst_fitness_values.append(np.max(fitness))
    avg_fitness_values.append(np.mean(fitness))

    generation = 0

    while generation < max_generations and (solfitness is None or (solfitness is not None and best_fitness > solfitness) ):
        if selection_type == 'Tournament':
            parents = tournament_selection(population,fitness,maximize=False)
        else:
            parents = roulette_wheel_selection(population, fitness,select_highest=False)
        np.random.shuffle(parents)
        offspring = []
        for i in range(0, pop_size, 2):
            parent1, parent2 = parents[i], parents[min(i + 1, pop_size - 1)]
            child1, child2 = simulated_binary_crossover(parent1, parent2, crossover_prob, lower_bound, upper_bound)
            child1 = non_uniform_mutation(child1, mutation_prob, mutation_step_size, lower_bound, upper_bound)
            child2 = non_uniform_mutation(child2, mutation_prob, mutation_step_size, lower_bound, upper_bound)
            offspring.extend([child1, child2])

        offspring = np.array(offspring[:pop_size])
        offspring_fitness = np.array([ackley_function(ind) for ind in offspring])

        population,fitness = mu_plus_lambda_selection(population,offspring,fitness,offspring_fitness,pop_size,select_highest=False)

        current_best_fitness = np.min(fitness)
        best_fitness_values.append(current_best_fitness)
        worst_fitness_values.append(np.max(fitness))
        avg_fitness_values.append(np.mean(fitness))

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[np.argmin(fitness)]
            best_generation = generation + 1

        generation += 1


    return best_fitness_values, worst_fitness_values, avg_fitness_values , best_solution, best_generation

def plot_ackley_with_solution(best_solution, lower_bound=-5, upper_bound=5):
    x = np.linspace(lower_bound, upper_bound, 100)
    y = np.linspace(lower_bound, upper_bound, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([ackley_function(np.array([x_val, y_val])) for x_val, y_val in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

    best_x, best_y = best_solution[0], best_solution[1]
    best_z = ackley_function(best_solution)
    ax.scatter(best_x, best_y, best_z, color='red', s=50, label="Best Solution", marker='o')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Ackley Function Value")
    plt.title("Ackley Function with Best Solution Point")
    plt.legend()
    plt.show()