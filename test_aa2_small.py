# import numpy as np
# from itertools import combinations
# from collections import defaultdict
# from scipy.spatial.distance import cdist


# def find_indistinguishable_classes(
#     predicate_matrix, class_to_idx=None, predicates_name_to_idx=None
# ):
#     """
#     Find classes that cannot be uniquely identified by the given predicates.

#     Parameters:
#     -----------
#     predicate_matrix : numpy.ndarray
#         Binary matrix of shape (num_classes, num_predicates) where each row represents
#         the predicate values for a class.
#     class_to_idx : dict, optional
#         A dictionary mapping class names to their indices in the predicate matrix.
#     predicates_name_to_idx : dict, optional
#         A dictionary mapping predicate names to their indices in the predicate matrix.

#     Returns:
#     --------
#     dict
#         A dictionary where keys are tuples representing unique predicate signatures
#         and values are lists of class indices that share that signature
#     """
#     num_classes = predicate_matrix.shape[0]

#     # Create reverse mapping from index to class name
#     idx_to_class = None
#     if class_to_idx is not None:
#         idx_to_class = {v: k for k, v in class_to_idx.items()}

#     # Convert each row to a tuple for hashability
#     row_tuples = [tuple(row) for row in predicate_matrix]

#     # Group classes by their predicate signatures
#     class_signatures = defaultdict(list)
#     for i, signature in enumerate(row_tuples):
#         class_signatures[signature].append(i)

#     # Count unique signatures
#     unique_signatures = len(class_signatures)

#     # Find which classes have identical predicate sets
#     indistinguishable_groups = {
#         sig: indices for sig, indices in class_signatures.items() if len(indices) > 1
#     }

#     print(f"Analysis of Class Distinguishability:")
#     print(f"Total number of classes: {num_classes}")
#     print(f"Number of unique predicate signatures: {unique_signatures}")
#     print(
#         f"Number of classes that cannot be uniquely identified: {num_classes - unique_signatures}"
#     )
#     print(f"Number of indistinguishable groups: {len(indistinguishable_groups)}")

#     print("\nIndistinguishable Class Groups:")
#     for signature, indices in indistinguishable_groups.items():
#         if idx_to_class:
#             class_names = [idx_to_class[idx] for idx in indices]
#             print(f"Group with signature {signature}:")
#             print(f"  Classes: {', '.join(class_names)}")
#         else:
#             print(f"Group with signature {signature}:")
#             print(f"  Classes with indices: {indices}")

#     return indistinguishable_groups


# def analyze_predicate_contribution(
#     predicate_matrix, class_to_idx=None, predicates_name_to_idx=None
# ):
#     """
#     Analyze how each predicate contributes to class distinguishability.

#     Parameters:
#     -----------
#     predicate_matrix : numpy.ndarray
#         Binary matrix of shape (num_classes, num_predicates)
#     class_to_idx : dict, optional
#         A dictionary mapping class names to their indices in the predicate matrix.
#     predicates_name_to_idx : dict, optional
#         A dictionary mapping predicate names to their indices in the predicate matrix.
#     """
#     num_classes, num_predicates = predicate_matrix.shape

#     # Create reverse mapping from predicate index to name
#     predicates_idx_to_name = None
#     if predicates_name_to_idx is not None:
#         predicates_idx_to_name = {v: k for k, v in predicates_name_to_idx.items()}

#     # Baseline - how many unique signatures do we have with all predicates
#     full_tuples = [tuple(row) for row in predicate_matrix]
#     baseline_unique = len(set(full_tuples))

#     print("\nPredicate Contribution Analysis:")
#     print(
#         f"With all {num_predicates} predicates: {baseline_unique} unique class signatures"
#     )

#     # Test removing each predicate
#     for p in range(num_predicates):
#         # Create a matrix without this predicate
#         reduced_matrix = np.delete(predicate_matrix, p, axis=1)

#         # Count unique signatures
#         reduced_tuples = [tuple(row) for row in reduced_matrix]
#         reduced_unique = len(set(reduced_tuples))

#         # Calculate the impact of removing this predicate
#         impact = baseline_unique - reduced_unique

#         predicate_name = (
#             predicates_idx_to_name.get(p, f"Predicate_{p}")
#             if predicates_idx_to_name
#             else f"Predicate_{p}"
#         )
#         print(
#             f"Removing {predicate_name}: {reduced_unique} unique signatures (loss of {impact} distinctions)"
#         )


# def print_predicate_signatures(indistinguishable_groups, predicates_idx_to_name):
#     """
#     Print the predicate signatures for each indistinguishable group in a more readable format.

#     Parameters:
#     -----------
#     indistinguishable_groups : dict
#         Dictionary where keys are tuples representing predicate signatures and values are lists of class indices
#     predicates_idx_to_name : dict
#         Dictionary mapping predicate indices to predicate names
#     """
#     print("\nDetailed Predicate Signatures for Indistinguishable Groups:")

#     for signature, indices in indistinguishable_groups.items():
#         class_names = [idx_to_class[idx] for idx in indices]
#         print(f"\nGroup with signature - Classes: {', '.join(class_names)}")
#         for i, value in enumerate(signature):
#             predicate_name = predicates_idx_to_name.get(i, f"Predicate_{i}")
#             print(f"  {predicate_name}: {value}")


# def reduce_to_specified_predicates(
#     original_matrix, original_predicates_name_to_idx, specified_predicates
# ):
#     """
#     Reduce a predicate matrix to only use the specified predicates.

#     Parameters:
#     -----------
#     original_matrix : numpy.ndarray
#         The original predicate matrix
#     original_predicates_name_to_idx : dict
#         Dictionary mapping original predicate names to indices
#     specified_predicates : list
#         List of predicate names to keep

#     Returns:
#     --------
#     tuple
#         (reduced_matrix, new_predicates_name_to_idx) where:
#         - reduced_matrix is the matrix with only the specified predicates
#         - new_predicates_name_to_idx is the updated mapping
#     """
#     # Create a mapping from the original predicate indices to the new indices
#     original_to_new_indices = {}

#     # Create new predicate name to index mapping
#     new_predicates_name_to_idx = {}

#     # Find indices of specified predicates in the original matrix
#     for new_idx, predicate_name in enumerate(specified_predicates):
#         if predicate_name in original_predicates_name_to_idx:
#             original_idx = original_predicates_name_to_idx[predicate_name]
#             original_to_new_indices[original_idx] = new_idx
#             new_predicates_name_to_idx[predicate_name] = new_idx
#         else:
#             print(
#                 f"Warning: Predicate '{predicate_name}' not found in the original dataset"
#             )

#     # Determine which columns to keep
#     columns_to_keep = list(original_to_new_indices.keys())

#     # Create the reduced matrix
#     reduced_matrix = original_matrix[:, columns_to_keep]

#     return reduced_matrix, new_predicates_name_to_idx


# # Main execution
# if __name__ == "__main__":
#     import numpy as np
#     from itertools import combinations
#     from collections import defaultdict
#     from torchvision import transforms
#     from datasets.aa2 import AA2

#     # Load the dataset as in the original code
#     data_dir = "/data/Datasets/"
#     normalize = transforms.Normalize(
#         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#     )

#     transform_train = transforms.Compose(
#         [
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ]
#     )
#     transform_test = transforms.Compose(
#         [
#             transforms.ToTensor(),
#             normalize,
#         ]
#     )
#     train_dataset = AA2(
#         root=data_dir,
#         split="train",
#         transform=transform_train,
#     )

#     # Get the actual data
#     predicate_matrix = train_dataset.predicate_matrix
#     class_to_idx = train_dataset.animals_class_to_idx
#     predicates_name_to_idx = train_dataset.predicates_name_to_idx

#     # Create reverse mapping for class names
#     idx_to_class = {v: k for k, v in class_to_idx.items()}

#     # List of reduced predicates as specified
#     reduced_predicates = [
#         "vegetation",
#         "forager",
#         "white",
#         "toughskin",
#         "solitary",
#         "small",
#         "fierce",
#         "slow",
#         "meat",
#         "plains",
#         "nestspot",
#     ]

#     # Reduce the predicate matrix to only the specified predicates
#     reduced_matrix, reduced_predicates_name_to_idx = reduce_to_specified_predicates(
#         predicate_matrix, predicates_name_to_idx, reduced_predicates
#     )

#     print(f"Original predicate matrix shape: {predicate_matrix.shape}")
#     print(f"Reduced predicate matrix shape: {reduced_matrix.shape}")
#     print(f"Reduced predicates: {list(reduced_predicates_name_to_idx.keys())}")

#     # Analyze indistinguishable classes with reduced predicates
#     indistinguishable_groups = find_indistinguishable_classes(
#         reduced_matrix, class_to_idx, reduced_predicates_name_to_idx
#     )

#     # Analyze predicate contribution
#     analyze_predicate_contribution(
#         reduced_matrix, class_to_idx, reduced_predicates_name_to_idx
#     )

#     # Print detailed signatures for indistinguishable groups
#     reduced_predicates_idx_to_name = {
#         v: k for k, v in reduced_predicates_name_to_idx.items()
#     }
#     print_predicate_signatures(indistinguishable_groups, reduced_predicates_idx_to_name)


import numpy as np
from itertools import combinations
from collections import defaultdict
from torchvision import transforms
from datasets.aa2 import AA2


def count_indistinguishable_classes(predicate_matrix):
    """
    Count the number of classes that cannot be uniquely identified.

    Parameters:
    -----------
    predicate_matrix : numpy.ndarray
        Binary matrix of shape (num_classes, num_predicates)

    Returns:
    --------
    int
        Number of classes that cannot be uniquely identified
    """
    num_classes = predicate_matrix.shape[0]

    # Convert each row to a tuple for hashability
    row_tuples = [tuple(row) for row in predicate_matrix]

    # Count unique signatures
    unique_signatures = len(set(row_tuples))

    # The number of classes that cannot be uniquely identified
    return num_classes - unique_signatures


def evaluate_predicate_subset(original_matrix, subset_indices):
    """
    Evaluate a subset of predicates by counting indistinguishable classes.

    Parameters:
    -----------
    original_matrix : numpy.ndarray
        The original predicate matrix
    subset_indices : list
        List of indices of predicates to include in the subset

    Returns:
    --------
    int
        Number of classes that cannot be uniquely identified with this subset
    """
    # Create a matrix with only the selected predicates
    subset_matrix = original_matrix[:, subset_indices]

    # Count indistinguishable classes
    return count_indistinguishable_classes(subset_matrix)


def find_optimal_predicate_subset(
    original_matrix, original_predicates_name_to_idx, subset_size=6
):
    """
    Find the optimal subset of predicates that minimizes indistinguishable classes.

    Parameters:
    -----------
    original_matrix : numpy.ndarray
        The original predicate matrix
    original_predicates_name_to_idx : dict
        Dictionary mapping original predicate names to indices
    subset_size : int
        Size of the predicate subset to find

    Returns:
    --------
    tuple
        (best_indices, best_names, best_score, reduced_matrix) where:
        - best_indices is the list of predicate indices in the optimal subset
        - best_names is the list of predicate names in the optimal subset
        - best_score is the number of indistinguishable classes with the optimal subset
        - reduced_matrix is the predicate matrix with only the optimal subset of predicates
    """
    num_predicates = original_matrix.shape[1]

    # Create reverse mapping from index to predicate name
    predicates_idx_to_name = {v: k for k, v in original_predicates_name_to_idx.items()}

    # Generate all possible combinations of predicates
    all_combinations = list(combinations(range(num_predicates), subset_size))

    # Initialize with worst possible score
    best_score = float("inf")
    best_indices = None

    print(
        f"Evaluating {len(all_combinations)} possible combinations of {subset_size} predicates..."
    )

    # Evaluate each combination
    for i, subset_indices in enumerate(all_combinations):
        # Print progress every 100 combinations
        if i % 100 == 0:
            print(f"Evaluated {i}/{len(all_combinations)} combinations...")

        # Count indistinguishable classes with this subset
        score = evaluate_predicate_subset(original_matrix, subset_indices)

        # Update best score if this combination is better
        if score < best_score:
            best_score = score
            best_indices = subset_indices

    # Get the names of the best predicates
    best_names = [
        predicates_idx_to_name.get(idx, f"Predicate_{idx}") for idx in best_indices
    ]

    # Create the reduced matrix with the best subset
    reduced_matrix = original_matrix[:, best_indices]

    return best_indices, best_names, best_score, reduced_matrix


def analyze_indistinguishable_groups(predicate_matrix, class_to_idx):
    """
    Analyze the indistinguishable groups of classes.

    Parameters:
    -----------
    predicate_matrix : numpy.ndarray
        Binary matrix of shape (num_classes, num_predicates)
    class_to_idx : dict
        A dictionary mapping class names to their indices in the predicate matrix.

    Returns:
    --------
    dict
        A dictionary where keys are tuples representing unique predicate signatures
        and values are lists of class indices that share that signature
    """
    num_classes = predicate_matrix.shape[0]

    # Create reverse mapping from index to class name
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Convert each row to a tuple for hashability
    row_tuples = [tuple(row) for row in predicate_matrix]

    # Group classes by their predicate signatures
    class_signatures = defaultdict(list)
    for i, signature in enumerate(row_tuples):
        class_signatures[signature].append(i)

    # Count unique signatures
    unique_signatures = len(class_signatures)

    # Find which classes have identical predicate sets
    indistinguishable_groups = {
        sig: indices for sig, indices in class_signatures.items() if len(indices) > 1
    }

    print(f"\nAnalysis of Class Distinguishability:")
    print(f"Total number of classes: {num_classes}")
    print(f"Number of unique predicate signatures: {unique_signatures}")
    print(
        f"Number of classes that cannot be uniquely identified: {num_classes - unique_signatures}"
    )
    print(f"Number of indistinguishable groups: {len(indistinguishable_groups)}")

    print("\nIndistinguishable Class Groups:")
    for signature, indices in indistinguishable_groups.items():
        class_names = [idx_to_class[idx] for idx in indices]
        print(f"Group with signature {signature}:")
        print(f"  Classes: {', '.join(class_names)}")

    return indistinguishable_groups


def print_predicate_signatures(
    indistinguishable_groups,
    reduced_predicates_idx_to_name,
    reduced_predicates_name_to_idx,
    idx_to_class,
):
    """
    Print the predicate signatures for each indistinguishable group in a more readable format.

    Parameters:
    -----------
    indistinguishable_groups : dict
        Dictionary where keys are tuples representing predicate signatures and values are lists of class indices
    reduced_predicates_idx_to_name : dict
        Dictionary mapping predicate indices to predicate names
    """
    print("\nDetailed Predicate Signatures for Indistinguishable Groups:")

    for signature, indices in indistinguishable_groups.items():
        class_names = [idx_to_class[idx] for idx in indices]
        print(f"\nGroup with signature - Classes: {', '.join(class_names)}")
        for i, value in enumerate(signature):
            predicate_name = reduced_predicates_idx_to_name.get(i, f"Predicate_{i}")
            predicate_idx = reduced_predicates_name_to_idx.get(predicate_name, i)
            print(f"  {predicate_name} (index {predicate_idx}): {value}")


if __name__ == "__main__":
    # Load the dataset
    data_dir = "/data/Datasets/"
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dataset = AA2(
        root=data_dir,
        split="train",
        transform=transform_train,
    )

    # Get the actual data
    predicate_matrix = train_dataset.predicate_matrix
    class_to_idx = train_dataset.animals_class_to_idx
    predicates_name_to_idx = train_dataset.predicates_name_to_idx

    # Create reverse mapping for class names
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Original 11 predicates
    original_predicates = [
        "vegetation",
        "forager",
        "white",
        "toughskin",
        "solitary",
        "small",
        "fierce",
        "slow",
        "meat",
        "plains",
        "nestspot",
    ]

    # Create a mapping from the original predicate names to their indices
    original_indices = []
    subset_predicates_name_to_idx = {}

    for predicate_name in original_predicates:
        if predicate_name in predicates_name_to_idx:
            original_idx = predicates_name_to_idx[predicate_name]
            original_indices.append(original_idx)
            subset_predicates_name_to_idx[predicate_name] = len(
                subset_predicates_name_to_idx
            )

    # Create the original reduced matrix with the specified 11 predicates
    original_reduced_matrix = predicate_matrix[:, original_indices]

    print(f"Original predicate matrix shape: {predicate_matrix.shape}")
    print(
        f"Reduced predicate matrix with 11 predicates shape: {original_reduced_matrix.shape}"
    )

    # Analyze baseline - how many classes are indistinguishable with all 11 predicates
    baseline_score = count_indistinguishable_classes(original_reduced_matrix)
    print(
        f"\nBaseline: With all 11 predicates, {baseline_score} classes cannot be uniquely identified"
    )

    # Find the optimal subset of 6 predicates
    best_indices, best_names, best_score, best_matrix = find_optimal_predicate_subset(
        original_reduced_matrix, subset_predicates_name_to_idx, subset_size=6
    )

    print(f"\nOptimal 6-predicate subset found:")
    print(f"Predicates: {', '.join(best_names)}")
    print(f"Indices (in the reduced 11-predicate matrix): {best_indices}")
    print(
        f"With these 6 predicates, {best_score} classes cannot be uniquely identified"
    )

    # Create mapping for the best predicates
    best_predicates_name_to_idx = {name: i for i, name in enumerate(best_names)}
    best_predicates_idx_to_name = {i: name for i, name in enumerate(best_names)}

    # Analyze indistinguishable groups with the best 6 predicates
    indistinguishable_groups = analyze_indistinguishable_groups(
        best_matrix, class_to_idx
    )

    # Print detailed signatures for indistinguishable groups
    print_predicate_signatures(
        indistinguishable_groups,
        best_predicates_idx_to_name,
        best_predicates_name_to_idx,
        idx_to_class,
    )
