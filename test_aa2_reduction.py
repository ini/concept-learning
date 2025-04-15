import numpy as np
from itertools import combinations
from collections import defaultdict
from torchvision import transforms
from datasets.aa2 import AA2


import numpy as np
from itertools import combinations
from collections import defaultdict
from scipy.spatial.distance import cdist


def find_minimal_predicates(
    predicate_matrix, class_to_idx=None, predicates_name_to_idx=None
):
    """
    Find the minimal set of predicates needed to uniquely identify each class.

    Parameters:
    -----------
    predicate_matrix : numpy.ndarray
        Binary matrix of shape (num_classes, num_predicates) where each row represents
        the predicate values for a class.
    class_to_idx : dict, optional
        A dictionary mapping class names to their indices in the predicate matrix.
    predicates_name_to_idx : dict, optional
        A dictionary mapping predicate names to their indices in the predicate matrix.

    Returns:
    --------
    tuple
        (minimal_indices, minimal_names, reduced_matrix) where:
        - minimal_indices is a list of predicate indices that are necessary
        - minimal_names is a list of predicate names corresponding to those indices
        - reduced_matrix is the predicate matrix with only the necessary columns
    """
    num_classes, num_predicates = predicate_matrix.shape

    # Create reverse mapping from predicate index to name
    predicates_idx_to_name = None
    if predicates_name_to_idx is not None:
        predicates_idx_to_name = {v: k for k, v in predicates_name_to_idx.items()}

    # Function to check if a set of predicates uniquely identifies all classes
    def is_unique_identifier(predicate_indices):
        # Extract the submatrix with only the selected predicates
        submatrix = predicate_matrix[:, predicate_indices]

        # Convert each row to a tuple for hashability
        row_tuples = [tuple(row) for row in submatrix]

        # If the number of unique tuples equals the number of classes,
        # then the predicates uniquely identify each class
        return len(set(row_tuples)) == num_classes

    # Start with a greedy approach: add predicates one by one based on how many class pairs they distinguish

    # First, identify which classes are identical with the full set of predicates
    full_tuples = [tuple(row) for row in predicate_matrix]
    unique_classes = len(set(full_tuples))

    if unique_classes < num_classes:
        print(
            f"Warning: Even with all predicates, only {unique_classes} unique class signatures exist."
        )

        # Find which classes have identical predicate sets
        class_signatures = defaultdict(list)
        for i, tup in enumerate(full_tuples):
            class_signatures[tup].append(i)

        # Print the classes with identical signatures
        print("Classes with identical predicate signatures:")
        for signature, indices in class_signatures.items():
            if len(indices) > 1:
                if class_to_idx:
                    class_names = [idx_to_class[idx] for idx in indices]
                    print(f"  {', '.join(class_names)}")
                else:
                    print(f"  Classes with indices: {indices}")

    # Initialize the set of selected predicates
    selected_indices = []

    # Keep track of pairs of classes that are still identical
    class_pairs = list(combinations(range(num_classes), 2))
    indistinguishable_pairs = []

    for i, j in class_pairs:
        if np.array_equal(predicate_matrix[i], predicate_matrix[j]):
            indistinguishable_pairs.append((i, j))

    if indistinguishable_pairs:
        print(
            f"Warning: {len(indistinguishable_pairs)} pairs of classes are identical with all predicates."
        )
        return selected_indices, [], predicate_matrix[:, selected_indices]

    # For each predicate, count how many class pairs it helps distinguish
    predicate_scores = []
    for p in range(num_predicates):
        distinguishable_count = 0
        for i, j in class_pairs:
            if predicate_matrix[i, p] != predicate_matrix[j, p]:
                distinguishable_count += 1
        predicate_scores.append((p, distinguishable_count))

    # Sort predicates by their score (descending)
    predicate_scores.sort(key=lambda x: x[1], reverse=True)

    # Add predicates one by one until all classes are uniquely identifiable
    candidate_indices = [p for p, _ in predicate_scores]

    # Try a greedy approach first
    for idx in candidate_indices:
        selected_indices.append(idx)
        if is_unique_identifier(selected_indices):
            break

    # If we've selected all predicates but still don't have uniqueness,
    # then it's not possible with the given predicates
    if len(selected_indices) == num_predicates and not is_unique_identifier(
        selected_indices
    ):
        print(
            "Warning: It's not possible to uniquely identify all classes with the given predicates."
        )

    # Now try to minimize further by removing predicates one by one if possible
    minimal_indices = selected_indices.copy()

    for idx in selected_indices:
        test_indices = [i for i in minimal_indices if i != idx]
        if is_unique_identifier(test_indices):
            minimal_indices = test_indices

    # Get the names of the minimal predicates if available
    minimal_names = []
    if predicates_idx_to_name:
        minimal_names = [
            predicates_idx_to_name.get(idx, f"Predicate_{idx}")
            for idx in minimal_indices
        ]

    # Extract the reduced matrix
    reduced_matrix = predicate_matrix[:, minimal_indices]

    return minimal_indices, minimal_names, reduced_matrix


def identify_predicates_to_swap(
    class1_idx, class2_idx, predicate_matrix, predicate_indices, predicates_idx_to_name
):
    """
    Identify which specific predicates need to be swapped to transform class1 into class2.

    Parameters:
    -----------
    class1_idx : int
        Index of the source class in the predicate matrix
    class2_idx : int
        Index of the target class in the predicate matrix
    predicate_matrix : numpy.ndarray
        Binary matrix of shape (num_classes, num_predicates)
    predicate_indices : list
        List of indices of the predicates to consider
    predicates_idx_to_name : dict
        A dictionary mapping predicate indices to their names

    Returns:
    --------
    list of tuples
        Each tuple contains (predicate_idx, predicate_name, value_in_class1, value_in_class2)
    """
    # Get the predicate vectors for both classes (only for selected predicates)
    class1_predicates = predicate_matrix[class1_idx, predicate_indices]
    class2_predicates = predicate_matrix[class2_idx, predicate_indices]

    # Find indices where the predicates differ
    diff_indices = np.where(class1_predicates != class2_predicates)[0]

    # Create a list of predicates that need to be swapped
    predicates_to_swap = []
    for i in diff_indices:
        original_idx = predicate_indices[i]
        predicate_name = predicates_idx_to_name.get(
            original_idx, f"Predicate_{original_idx}"
        )
        predicates_to_swap.append(
            (
                original_idx,
                predicate_name,
                predicate_matrix[class1_idx, original_idx],
                predicate_matrix[class2_idx, original_idx],
            )
        )

    return predicates_to_swap


def get_closest_classes_with_minimal_predicates(
    predicate_matrix, minimal_indices, class_to_idx=None, predicates_name_to_idx=None
):
    """
    Find unique pairs of closest classes using only the minimal set of predicates.

    Parameters:
    -----------
    predicate_matrix : numpy.ndarray
        Binary matrix of shape (num_classes, num_predicates)
    minimal_indices : list
        Indices of the minimal set of predicates
    class_to_idx : dict, optional
        A dictionary mapping class names to their indices in the predicate matrix.
    predicates_name_to_idx : dict, optional
        A dictionary mapping predicate names to their indices in the predicate matrix.

    Returns:
    --------
    list of tuples
        Each tuple contains (class1_idx, class2_idx, num_swaps, class1_name, class2_name, predicates_to_swap)
        Only unique pairs are included (no symmetric duplicates)
    """
    # Create a reduced matrix with only the minimal set of predicates
    reduced_matrix = predicate_matrix[:, minimal_indices]
    num_classes = predicate_matrix.shape[0]

    # Create reverse mapping from predicate index to name
    predicates_idx_to_name = None
    if predicates_name_to_idx is not None:
        predicates_idx_to_name = {v: k for k, v in predicates_name_to_idx.items()}

    # Create reverse mapping from index to class name
    idx_to_class = None
    if class_to_idx is not None:
        idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Calculate Hamming distance between each pair of classes in the reduced matrix
    hamming_distances = cdist(reduced_matrix, reduced_matrix, metric="hamming")
    # Convert from fraction to count
    swap_matrix = np.round(hamming_distances * len(minimal_indices)).astype(int)

    # Track pairs we've already seen to avoid duplicates
    seen_pairs = set()
    unique_closest_pairs = []

    # Create a mask to exclude self-comparisons
    np.fill_diagonal(swap_matrix, len(minimal_indices) + 1)

    # For each class, find its closest class
    for i in range(num_classes):
        closest_idx = np.argmin(swap_matrix[i])
        min_swaps = swap_matrix[i, closest_idx]

        # Create a canonical representation of the pair (always order by smaller index first)
        pair = tuple(sorted([i, closest_idx]))

        # Skip if we've already seen this pair
        if pair in seen_pairs:
            continue

        seen_pairs.add(pair)

        # Get class names if available
        class1_name = idx_to_class[pair[0]] if idx_to_class else None
        class2_name = idx_to_class[pair[1]] if idx_to_class else None

        # Identify which predicates need to be swapped
        predicates_to_swap = None
        if predicates_idx_to_name is not None:
            predicates_to_swap = identify_predicates_to_swap(
                pair[0],
                pair[1],
                predicate_matrix,
                minimal_indices,
                predicates_idx_to_name,
            )

        unique_closest_pairs.append(
            (pair[0], pair[1], min_swaps, class1_name, class2_name, predicates_to_swap)
        )

    # Sort by number of swaps needed (ascending)
    unique_closest_pairs.sort(key=lambda x: x[2])

    return unique_closest_pairs


def analyze_minimal_predicates_and_closest_classes(
    predicate_matrix, class_to_idx=None, predicates_name_to_idx=None
):
    """
    Find the minimal set of predicates and analyze the closest classes using that set.

    Parameters:
    -----------
    predicate_matrix : numpy.ndarray
        Binary matrix of shape (num_classes, num_predicates)
    class_to_idx : dict, optional
        A dictionary mapping class names to their indices in the predicate matrix.
    predicates_name_to_idx : dict, optional
        A dictionary mapping predicate names to their indices in the predicate matrix.
    """
    # First, find the minimal set of predicates
    minimal_indices, minimal_names, reduced_matrix = find_minimal_predicates(
        predicate_matrix, class_to_idx, predicates_name_to_idx
    )

    print(f"\nMinimal Predicate Set Analysis:")
    print(f"Original number of predicates: {predicate_matrix.shape[1]}")
    print(f"Minimal number of predicates: {len(minimal_indices)}")
    print(
        f"Reduction: {predicate_matrix.shape[1] - len(minimal_indices)} predicates removed "
        f"({(1 - len(minimal_indices)/predicate_matrix.shape[1])*100:.2f}%)"
    )

    if minimal_names:
        print("\nMinimal predicate set:")
        for i, name in enumerate(minimal_names):
            print(f"{i+1}. {name}")

    # Now find the closest classes using only the minimal predicates
    closest_pairs = get_closest_classes_with_minimal_predicates(
        predicate_matrix, minimal_indices, class_to_idx, predicates_name_to_idx
    )

    print("\nAnalysis of Closest Classes Using Minimal Predicates:\n")
    print(
        f"{'Class 1':<20} | {'Class 2':<20} | {'Swaps Required':<15} | {'Predicates to Swap'}"
    )
    print("-" * 100)

    for result in closest_pairs:
        class1_idx, class2_idx, swaps, class1_name, class2_name, predicates_to_swap = (
            result
        )

        # Print basic info about the class pair
        print(f"{class1_name:<20} | {class2_name:<20} | {swaps:<15} | ", end="")

        # Print details about the predicates that need to be swapped
        if predicates_to_swap:
            predicate_details = []
            for idx, name, val1, val2 in predicates_to_swap:
                # Show the change direction
                change = f"{val1}->{val2}"
                predicate_details.append(f"{name} ({change})")

            # Print first predicate on the same line
            if predicate_details:
                print(predicate_details[0])
            else:
                print("")

            # Print remaining predicates on new lines with proper indentation
            for detail in predicate_details[1:]:
                print(f"{'':<60} | {detail}")
        else:
            print("No predicate details available")

    # Find all class-to-class mappings with minimum swaps
    reduced_matrix = predicate_matrix[:, minimal_indices]
    hamming_distances = cdist(reduced_matrix, reduced_matrix, metric="hamming")
    swap_matrix = np.round(hamming_distances * len(minimal_indices)).astype(int)

    # Create a mask for the diagonal (we don't want to consider self-comparisons)
    np.fill_diagonal(swap_matrix, len(minimal_indices) + 1)
    min_swaps = np.min(swap_matrix)

    # Find all pairs with the minimum number of swaps
    min_pairs = []
    seen_min_pairs = set()

    min_indices = np.argwhere(swap_matrix == min_swaps)
    for i, j in min_indices:
        # Create canonical representation of the pair
        pair = tuple(sorted([i, j]))
        if pair in seen_min_pairs:
            continue
        seen_min_pairs.add(pair)
        min_pairs.append((i, j))

    # Create reverse mapping from index to class name
    idx_to_class = None
    if class_to_idx is not None:
        idx_to_class = {v: k for k, v in class_to_idx.items()}

    if min_pairs:
        print("\nOverall Minimum Swaps Required (Using Minimal Predicates):")
        for i, j in min_pairs:
            class1 = idx_to_class[i] if idx_to_class else f"Class_{i}"
            class2 = idx_to_class[j] if idx_to_class else f"Class_{j}"
            print(f"{class1} <-> {class2}: {min_swaps} swaps")

            # Print the predicates that need to be swapped
            if predicates_name_to_idx is not None:
                predicates_idx_to_name = {
                    v: k for k, v in predicates_name_to_idx.items()
                }
                predicates_to_swap = identify_predicates_to_swap(
                    i, j, predicate_matrix, minimal_indices, predicates_idx_to_name
                )
                print("  Predicates to swap:")
                for idx, name, val1, val2 in predicates_to_swap:
                    print(f"  - {name}: {val1} -> {val2}")


# Example usage:
if __name__ == "__main__":
    # In a real scenario, use your actual data
    np.random.seed(42)

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
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dataset = AA2(
        root=data_dir,
        split="train",
        transform=transform_train,
    )

    # In your actual code, replace with:
    predicate_matrix = train_dataset.predicate_matrix
    class_to_idx = train_dataset.animals_class_to_idx
    predicates_name_to_idx = train_dataset.predicates_name_to_idx
    # In your actual code, use:
    # predicate_matrix = train_dataset.predicate_matrix
    # class_to_idx = train_dataset.animals_class_to_idx
    # predicates_name_to_idx = train_dataset.predicates_name_to_idx

    analyze_minimal_predicates_and_closest_classes(
        predicate_matrix, class_to_idx, predicates_name_to_idx
    )
