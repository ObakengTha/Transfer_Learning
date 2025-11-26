import pandas as pd
import random
import numpy as np
import copy
import math
import pickle

random_seed = 80
random.seed(random_seed)


df = pd.read_excel("197Dataset.xlsx")

ColumnsToFix = ['lread','ppgin','lwrite','pgfree','atch','pgout','pgscan','pgout','pgin']

for col in ColumnsToFix:
    df[col] = df[col].replace(0, np.nan)
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

df = df.drop_duplicates().reset_index(drop=True)

test_ratio = 0.2
test_size = int(len(df) * test_ratio)

Feature_cols = df.columns.drop("target")
X_Transfer = df[Feature_cols].to_numpy()
y_Transfer = df["target"].to_numpy()

trainSet = df.iloc[:-test_size]
testSet = df.iloc[-test_size:]

X_train = trainSet[Feature_cols].to_numpy()
y_train = trainSet["target"].to_numpy()

X_test = testSet[Feature_cols].to_numpy()
y_test = testSet["target"].to_numpy()

# GP Parameters
PopulationSize = 500
Generations = 75
MutationRate = 0.05
CrossOverRate = 0.8
Max_Depth = 5

Functions = ['+', '-', '*', '/', 'sin', 'log', 'exp']
Terminals = [f"X{i}" for i in range(X_Transfer.shape[1])] + [str(round(random.uniform(-1, 1), 2)) for _ in range(5)]

# zero_counts = (df==0).sum()
# zero_percent = (zero_counts/len(df))*100

# zero_summary = pd.DataFrame({
#     'Zero Count': zero_counts,
#     'Zero %': zero_percent
# })

# print(zero_summary.sort_values(by= 'Zero %', ascending=False))

class BinaryTreeNode:

    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None

class BinaryTree:
    def __init__(self, root=None):
        self.root = root

    def evaluate(self, *inputs):
        return self._evaluate_node(self.root, inputs)

    def _evaluate_node(self, node, inputs):
        if node is None:
            return 0

        if node.is_leaf():
            if node.data.startswith("X"):
                idx = int(node.data[1:])
                return inputs[idx]
            else:
                try:
                    return float(node.data)
                except ValueError:
                    return 0

        left_val = self._evaluate_node(node.left, inputs)
        right_val = self._evaluate_node(node.right, inputs)

        try:
            if node.data == "+":
                return left_val + right_val
            elif node.data == "-":
                return left_val - right_val
            elif node.data == "*":
                return left_val * right_val
            elif node.data == "/":
                return left_val / (right_val + 1e-6)
            elif node.data == "sin":
                return math.sin(left_val)
            elif node.data == "log":
                return math.log(abs(left_val) + 1e-16)
            elif node.data == "exp":
                return math.exp(min(left_val, 50))
        except Exception:
            return 0
        return 0

with open("best_tree.pk5","rb") as f:
    best_tree = pickle.load(f)

# Grow Method
def generate_random_tree(depth):
    if depth <= 1 or random.random() < 0.5:
        return BinaryTreeNode(random.choice(Terminals))
    else:
        op = random.choice(Functions)
        if op in ['sin', 'log', 'exp']:
            left_child = generate_random_tree(depth - 1)
            return BinaryTreeNode(op, left_child, None)
        else:
            left_child = generate_random_tree(depth - 1)
            right_child = generate_random_tree(depth - 1)
            return BinaryTreeNode(op, left_child, right_child)

# Fitness Function
def GetTree_Size(node):
    if node is None:
        return 0
    return 1 + GetTree_Size(node.left) + GetTree_Size(node.right)

def calculate_tree_mse(tree, X, y):
    predictions = [tree.evaluate(*x_row) for x_row in X]
    mse = np.mean((y - np.array(predictions)) ** 2)
    penalty = 0.0001 * GetTree_Size(tree.root)
    return mse + penalty

# Tree Operations
def get_all_nodes(node, parent=None, is_left=None):
    if node is None:
        return []
    nodes = [(node, parent, is_left)]
    nodes += get_all_nodes(node.left, node, True)
    nodes += get_all_nodes(node.right, node, False)
    return nodes

def subtree_crossover(tree1, tree2):
    tree1_copy = copy.deepcopy(tree1)
    tree2_copy = copy.deepcopy(tree2)

    nodes1 = get_all_nodes(tree1_copy.root)
    nodes2 = get_all_nodes(tree2_copy.root)

    node1, parent1, is_left1 = random.choice(nodes1)
    node2, parent2, is_left2 = random.choice(nodes2)

    if parent1 is None:
        tree1_copy.root = node2
    else:
        if is_left1:
            parent1.left = node2
        else:
            parent1.right = node2

    if parent2 is None:
        tree2_copy.root = node1
    else:
        if is_left2:
            parent2.left = node1
        else:
            parent2.right = node1

    return tree1_copy, tree2_copy

def mutate_tree(node, mutation_rate, max_depth):
    if node is None:
        return None

    if random.random() < mutation_rate:
        return generate_random_tree(random.randint(1, max_depth))
    else:
        node.left = mutate_tree(node.left, mutation_rate, max_depth)
        node.right = mutate_tree(node.right, mutation_rate, max_depth)
        return node

def mutate(tree, mutation_rate, max_depth):
    tree_copy = copy.deepcopy(tree)
    tree_copy.root = mutate_tree(tree_copy.root, mutation_rate, max_depth)
    return tree_copy

def tournament_selection(population, X, y, k=3):
    selected = random.sample(population, k)
    selected.sort(key=lambda t: calculate_tree_mse(t, X, y))
    return selected[0]

# Initialize population
population = [BinaryTree(generate_random_tree(Max_Depth)) for _ in range(PopulationSize)]

# Evolution Process
best_fitnessoverall = float('inf')
notimproving = 0
earlystop = 10

population_transfer = [copy.deepcopy(best_tree)] + [BinaryTree(generate_random_tree(Max_Depth)) for _ in range(PopulationSize - 1)]

for generation in range(Generations):
    fitnesses = [calculate_tree_mse(ind, X_train, y_train) for ind in population_transfer]
    best_fitness = min(fitnesses)
    print(f"Tranfer Generation {generation + 1}, Best MSE: {best_fitness:.4f}")

    if best_fitness < best_fitnessoverall:
        best_fitnessoverall = best_fitness
        notimproving = 0
    else:
        notimproving += 1

    if notimproving >= earlystop:
        print(f"No improvement for {earlystop} generations. Early stopping at generation {generation + 1}.")
        break

    sorted_population = [ind for _, ind in sorted(zip(fitnesses, population_transfer), key=lambda x: x[0])]
    new_population = sorted_population[:5]

    while len(new_population) < PopulationSize:
        parent1 = tournament_selection(population_transfer, X_train, y_train)
        parent2 = tournament_selection(population_transfer, X_train, y_train)

        if random.random() < CrossOverRate:
            child1, child2 = subtree_crossover(parent1, parent2)
        else:
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)

        child1 = mutate(child1, MutationRate, Max_Depth)
        child2 = mutate(child2, MutationRate, Max_Depth)

        new_population.extend([child1, child2])

    population_transfer = new_population[:PopulationSize]

# Final evaluation
best_transfer_tree = min(population_transfer, key=lambda t: calculate_tree_mse(t, X_train, y_train))
# with open("best_tree.pk1","wb") as f:
#     pickle.dump(best_transfer_tree, f)
test_mse = calculate_tree_mse(best_transfer_tree, X_test, y_test)
print("Final Test MSE:", test_mse)

