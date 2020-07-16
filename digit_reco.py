import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import datasets
from sklearn import manifold

#%matplotlib inline

data = datasets.fetch_openml(
                'mnist_784',
                version=1,
                return_X_y=True
)
pixel_values, targets = data
targets = targets.astype(int)

print('Printing an image\n')
print(pixel_values[0,:])
print(targets[0])

single_image = pixel_values[0,:].reshape(28,28)
plt.imshow(single_image, cmap='gray')
# for printing an image
plt.show()

tsne = manifold.TSNE(n_components=2, random_state=42)
transformed_data = tsne.fit_transform(pixel_values[:3000, :])

tsne_df = pd.DataFrame(
    np.column_stack((transformed_data, targets[:3000])),
    columns=["x", "y", "targets"]
)

tsne_df.loc[:, "targets"] = tsne_df.targets.astype(int)

grid = sns.FacetGrid(tsne_df, hue ="targets", size=8)

grid.map(plt.scatter, "x", "y").add_legend()
