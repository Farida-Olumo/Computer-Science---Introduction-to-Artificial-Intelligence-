import pandas as pd
import numpy as np
import scipy.stats

class Decision_Tree:
    def __init__(self, attribute, branches, label):
        self.attribute = attribute
        self.branches = branches
        self.label = label

    @staticmethod
    def make_leaf(label):
        return Decision_Tree('class', {}, label)
    
    @staticmethod
    def make_node(attribute, branches):
        return Decision_Tree(attribute, branches, None)

    def __repr__(self):
        return self.string_repr(0)
        
    def string_repr(self, indent):
        indentation = '\t' * indent
        if self.attribute == 'class':
            return f'\n{indentation}class = {self.label}'
        else:
            representation = ''
            for value in self.branches:
                representation += f'\n{indentation}{self.attribute} = {value}:'
                representation += self.branches[value].string_repr(indent+1)
            return representation

    def classify(self, data_point):
        if self.attribute == 'class':
            return self.label
        else:
            val = data_point.get(self.attribute, 'NA')
            if val not in self.branches:
                return "Low" 
            return self.branches[val].classify(data_point)

def entropy(data_frame):
    if data_frame.empty: return 0
    counts = data_frame['class'].value_counts(normalize=True)
    return scipy.stats.entropy(counts, base=2)

def info_gain(data_frame, attribute, domain):
    total_entropy = entropy(data_frame)
    weighted_entropy = 0
    for value in domain:
        subset = data_frame[data_frame[attribute] == value]
        if not subset.empty:
            weighted_entropy += (len(subset) / len(data_frame)) * entropy(subset)
    return total_entropy - weighted_entropy

def choose_attribute(attributes_with_domains, data_frame):
    best_score = -1
    best_attribute = list(attributes_with_domains.keys())[0]
    for attr, domain in attributes_with_domains.items():
        score = info_gain(data_frame, attr, domain)
        if score > best_score:
            best_score, best_attribute = score, attr
    return best_score, best_attribute

def train_decision_tree(data_frame, attributes_with_domains, default_class, threshold):
    if data_frame.empty:
        return Decision_Tree.make_leaf(default_class)
    maj_class = data_frame['class'].mode()[0]
    if len(data_frame['class'].unique()) == 1 or len(data_frame) < threshold or not attributes_with_domains:
        return Decision_Tree.make_leaf(maj_class)
    score, best_attr = choose_attribute(attributes_with_domains, data_frame)
    branches = {}
    remaining_attrs = {k: v for k, v in attributes_with_domains.items() if k != best_attr}
    for value in attributes_with_domains[best_attr]:
        subset = data_frame[data_frame[best_attr] == value]
        branches[value] = train_decision_tree(subset, remaining_attrs, maj_class, threshold)
    return Decision_Tree.make_node(best_attr, branches)

def eval_decision_tree(tree, data_frame):
    correct = sum(tree.classify(row) == row['class'] for _, row in data_frame.iterrows())
    return correct / len(data_frame)

def cross_validation(train_data, test_data, attributes_with_domains, k, threshold_list):
    best_threshold = threshold_list[0]
    max_val_acc = -1
    fold_size = len(train_data) // k
    print("--- 10-Fold Cross Validation ---")
    for t in threshold_list:
        accs = []
        for i in range(k):
            val = train_data.iloc[i*fold_size:(i+1)*fold_size]
            train = train_data.drop(val.index)
            tree = train_decision_tree(train, attributes_with_domains, 'Low', t)
            accs.append(eval_decision_tree(tree, val))
        avg_acc = np.mean(accs)
        print(f"Threshold {t}: Avg Val Acc = {avg_acc:.4f}")
        if avg_acc > max_val_acc:
            max_val_acc, best_threshold = avg_acc, t
    final_tree = train_decision_tree(train_data, attributes_with_domains, 'Low', best_threshold)
    test_acc = eval_decision_tree(final_tree, test_data)
    return best_threshold, test_acc, final_tree

if __name__ == "__main__":
    data_frame = pd.read_csv("categorical_real_estate.csv").fillna('NA')
    attributes = [c for c in data_frame.columns if c != 'class']
    attr_domains = {attr: set(data_frame[attr]) for attr in attributes}
    train_data = data_frame.iloc[0:1000]
    test_data = data_frame.iloc[1000:]
    best_t, accuracy, tree = cross_validation(train_data, test_data, attr_domains, 10, [10, 20, 40, 80, 160])
    print(f"\nBest Threshold: {best_t}\nTest Accuracy: {accuracy:.4f}\nTree Structure:\n{tree}")