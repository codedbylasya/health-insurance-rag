import numpy as np
embeddings = np.load(r"C:\Projects\PortfolioProject\health_Doc_Summarizer\data\embeddings.npy")
print(embeddings.shape)
print(embeddings.dtype)
print(embeddings[0][:5])  # peek at first 5 values of the first vector