import sys
from scipy.io import mmread, mmwrite
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
import os

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <input_file.mtx>")
    sys.exit(1)

input_file = sys.argv[1]

# Load the matrix (dense or sparse)
matrix = mmread(input_file)

# If dense, convert to COO with explicit zeros included
if not hasattr(matrix, "tocoo"):
    # matrix is dense numpy array
    rows, cols = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]), indexing='ij')
    data = matrix.flatten()
    # Construct COO with all entries (including zeros)
    matrix = coo_matrix((data, (rows.flatten(), cols.flatten())), shape=matrix.shape)
else:
    # If sparse, convert to COO and explicitly keep zeros
    coo = matrix.tocoo()
    # Extract all entries including zeros (if zeros are not stored, they won't appear)
    # So to keep zeros, we can convert to dense and reconstruct COO with all elements
    dense = matrix.toarray()
    rows, cols = np.meshgrid(np.arange(dense.shape[0]), np.arange(dense.shape[1]), indexing='ij')
    data = dense.flatten()
    matrix = coo_matrix((data, (rows.flatten(), cols.flatten())), shape=dense.shape)

# Generate output filename: "<name>_coordinate.mtx"
stem = os.path.splitext(os.path.basename(input_file))[0]
output_file = f"{stem}_coordinate.mtx"

# Write as coordinate format
mmwrite(output_file, matrix, comment='Converted to coordinate format (with zeros)')

print(f"Wrote coordinate format to: {output_file}")