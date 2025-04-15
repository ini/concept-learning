from torchvision import transforms
from datasets.aa2 import AA2

import numpy as np
from scipy.spatial.distance import cdist


def find_min_swaps_between_classes(predicate_matrix, class_to_idx=None):
    """
    Find the fewest number of predicates that need to be swapped to transform one class into another.

    Parameters:
    -----------
    predicate_matrix : numpy.ndarray
        Binary matrix of shape (num_classes, num_predicates) where each row represents
        the predicate values for a class.
    class_to_idx : dict, optional
        A dictionary mapping class names to their indices in the predicate matrix.

    Returns:
    --------
    tuple
        (swap_matrix, idx_to_class) where:
        - swap_matrix is a matrix of shape (num_classes, num_classes) containing the number of
          predicates that need to be swapped
        - idx_to_class is a dictionary mapping indices to class names (if class_to_idx is provided)
    """
    # Ensure we're working with a binary matrix
    if not np.all(np.logical_or(predicate_matrix == 0, predicate_matrix == 1)):
        raise ValueError("Predicate matrix must contain only binary values (0 or 1)")

    # Calculate Hamming distance between each pair of classes
    num_classes = predicate_matrix.shape[0]

    # Using scipy.spatial.distance.cdist for efficient computation
    hamming_distances = cdist(predicate_matrix, predicate_matrix, metric="hamming")
    # Convert from fraction to count (hamming distance in scipy returns a fraction)
    swap_matrix = np.round(hamming_distances * predicate_matrix.shape[1]).astype(int)

    # Create a reverse mapping from index to class name if class_to_idx is provided
    idx_to_class = None
    if class_to_idx is not None:
        idx_to_class = {v: k for k, v in class_to_idx.items()}

    return swap_matrix, idx_to_class


def identify_predicates_to_swap(
    class1_idx, class2_idx, predicate_matrix, predicates_idx_to_name
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
    predicates_idx_to_name : dict
        A dictionary mapping predicate indices to their names

    Returns:
    --------
    list of tuples
        Each tuple contains (predicate_idx, predicate_name, value_in_class1, value_in_class2)
    """
    # Get the predicate vectors for both classes
    class1_predicates = predicate_matrix[class1_idx]
    class2_predicates = predicate_matrix[class2_idx]

    # Find indices where the predicates differ
    diff_indices = np.where(class1_predicates != class2_predicates)[0]

    # Create a list of predicates that need to be swapped
    predicates_to_swap = []
    for idx in diff_indices:
        predicate_name = predicates_idx_to_name.get(idx, f"Predicate_{idx}")
        predicates_to_swap.append(
            (idx, predicate_name, class1_predicates[idx], class2_predicates[idx])
        )

    return predicates_to_swap


def get_closest_classes_unique_pairs(
    predicate_matrix, class_to_idx=None, predicates_name_to_idx=None
):
    """
    Find unique pairs of closest classes, avoiding symmetric duplicates.

    Parameters:
    -----------
    predicate_matrix : numpy.ndarray
        Binary matrix of shape (num_classes, num_predicates)
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
    swap_matrix, idx_to_class = find_min_swaps_between_classes(
        predicate_matrix, class_to_idx
    )
    num_classes = swap_matrix.shape[0]

    # Create reverse mapping from predicate index to name
    predicates_idx_to_name = None
    if predicates_name_to_idx is not None:
        predicates_idx_to_name = {v: k for k, v in predicates_name_to_idx.items()}

    # Track pairs we've already seen to avoid duplicates
    seen_pairs = set()
    unique_closest_pairs = []

    # Create a mask to exclude self-comparisons
    np.fill_diagonal(swap_matrix, predicate_matrix.shape[1] + 1)

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
                pair[0], pair[1], predicate_matrix, predicates_idx_to_name
            )

        unique_closest_pairs.append(
            (pair[0], pair[1], min_swaps, class1_name, class2_name, predicates_to_swap)
        )

    # Sort by number of swaps needed (ascending)
    unique_closest_pairs.sort(key=lambda x: x[2])

    return unique_closest_pairs


def analyze_and_print_unique_pairs(
    predicate_matrix, class_to_idx, predicates_name_to_idx=None
):
    """
    Analyze the predicate matrix and print unique pairs of closest classes.

    Parameters:
    -----------
    predicate_matrix : numpy.ndarray
        Binary matrix of shape (num_classes, num_predicates)
    class_to_idx : dict
        A dictionary mapping class names to their indices in the predicate matrix.
    predicates_name_to_idx : dict, optional
        A dictionary mapping predicate names to their indices in the predicate matrix.
    """
    # Find all unique closest pairs
    unique_closest_pairs = get_closest_classes_unique_pairs(
        predicate_matrix, class_to_idx, predicates_name_to_idx
    )

    print("Analysis of Closest Classes by Predicate Swaps (Unique Pairs):\n")
    print(
        f"{'Class 1':<20} | {'Class 2':<20} | {'Swaps Required':<15} | {'Predicates to Swap'}"
    )
    print("-" * 100)

    for result in unique_closest_pairs:
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
    swap_matrix, idx_to_class = find_min_swaps_between_classes(
        predicate_matrix, class_to_idx
    )

    # Create a mask for the diagonal (we don't want to consider self-comparisons)
    np.fill_diagonal(swap_matrix, predicate_matrix.shape[1] + 1)
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

    if min_pairs:
        print("\nOverall Minimum Swaps Required:")
        for i, j in min_pairs:
            class1 = idx_to_class[i]
            class2 = idx_to_class[j]
            print(f"{class1} <-> {class2}: {min_swaps} swaps")

            # Print the predicates that need to be swapped
            if predicates_name_to_idx is not None:
                predicates_idx_to_name = {
                    v: k for k, v in predicates_name_to_idx.items()
                }
                predicates_to_swap = identify_predicates_to_swap(
                    i, j, predicate_matrix, predicates_idx_to_name
                )
                print("  Predicates to swap:")
                for idx, name, val1, val2 in predicates_to_swap:
                    print(f"  - {name}: {val1} -> {val2}")


# Example usage:
if __name__ == "__main__":
    # In a real scenario, use your actual predicate_matrix and class_to_idx
    # Example data for demonstration:
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
    # predicates_name_to_idx = train_dataset.predicates_name_to_idx
    predicates_name_to_idx = {
        v: k for k, v in enumerate(train_dataset.selected_predicates)
    }

    analyze_and_print_unique_pairs(
        predicate_matrix, class_to_idx, predicates_name_to_idx
    )
