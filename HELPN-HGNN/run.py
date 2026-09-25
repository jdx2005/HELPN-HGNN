import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

import warnings
warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False",
    category=UserWarning
)

from train import main

if __name__ == '__main__':
    main()