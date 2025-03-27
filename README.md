# README.md

## Search Algorithms Visualization and Performance Analysis

This project implements and visualizes several search algorithms, comparing their performance across different list sizes. It includes both linear and binary search techniques, as well as advanced variations such as adaptive binary search and exponential search. The program also provides detailed scatter plots to visualize the search process.

---

## Features

- **Search Algorithms**:
  - Linear Search
  - Linear Search with Sentinel
  - Ordered Linear Search
  - Binary Search (Iterative and Recursive)
  - Adaptive Binary Search
  - Binary Search Tree (BST)
  - Exponential Search combined with Binary Search
  
- **Visualization**:
  - Scatter plots dynamically updated during the search process.
  - Highlights the current position and target value for better understanding.

- **Performance Measurement**:
  - Measures execution time for each algorithm in microseconds.
  - Compares performance across list sizes (10, 100, 1000, 5000).

- **Binary Search Tree Construction**:
  - Builds a BST iteratively to avoid recursion depth issues.

---

## Installation

1. Clone the repository:

```
git clone https://github.com/ComparingSearchAlgorithms.git
cd ComparingSearchAlgorithms
```

2. Install dependencies:
```
pip install -r requirements.txt
```


---

## Usage

### Run the Program

Execute the script directly to visualize and compare the performance of search algorithms:

```
py algorithms.py
```

### Configuration Options

- **List Sizes**: Modify `list_sizes` in the script to test different dataset sizes.
- **Target Value**: Change `target` to specify the value being searched.
- **Number of Executions**: Adjust `n_executions` to control the number of repetitions for performance measurement.

---

## Code Overview

### Key Components

1. **Search Algorithms**:
   - Implemented as static methods in the `Searchers` class.
   - Includes visualization logic using scatter plots.

2. **Binary Search Tree**:
   - Constructed using `build_bst()` method.
   - Supports efficient searching through recursive traversal.

3. **Performance Measurement**:
   - `run_and_measure()` function calculates average execution time for each algorithm.

4. **Visualization**:
   - Scatter plots highlight key points during the search process.
   - Final comparative graph shows performance across algorithms and list sizes.

---

## Example Output

### Visualization During Search
Scatter plots dynamically update to show progress during each algorithm's execution.

### Performance Comparison Graph
A log-log scale graph comparing average execution times for each algorithm across different list sizes.

---

## Dependencies

- Python >= 3.7
- Matplotlib
- NumPy

---

## Contributing

Feel free to submit issues or pull requests to improve this project. Contributions are welcome!

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.


