from surprise import SVD
from pathlib import Path
from surprise.model_selection import cross_validate

PATH = Path('../../')
DATA= PATH/'data'
CODE= PATH/'code'
PRODUCTS=PATH/'products'
ratings = pd.read_csv(WORKING_DATA/'ratings_small.csv')


# Load the movielens-100k dataset (download it if needed).
data = ratings

# Use the famous SVD algorithm.
algo = SVD()

# Run 5-fold cross-validation and print results.
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
