#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


# In[5]:


#2: Load the datasets
customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")


# In[7]:


# Step 3: Merge the datasets
data = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")


# In[10]:


# Step 4: Aggregate data to create customer-level features
customer_features = data.groupby("CustomerID").agg({
    "TotalValue": "sum",  # Total transaction value
    "Quantity": "sum",    # Total quantity purchased
}).reset_index()


# In[15]:


# Create a pivot table for customer-product interactions
customer_product_features = data.pivot_table(
    index="CustomerID", columns="ProductID", values="Quantity", fill_value=0
)


# In[14]:


# Combine aggregated features with customer-product features
combined_features = pd.concat([customer_features.set_index("CustomerID"), customer_product_features], axis=1)


# In[16]:


# Step 5: Normalize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(combined_features)


# In[18]:


# Step 6: Compute similarity matrix
similarity_matrix = cosine_similarity(scaled_features)
similarity_df = pd.DataFrame(similarity_matrix, index=combined_features.index, columns=combined_features.index)


# In[19]:


# Step 7: Generate top 3 lookalikes for customers C0001 to C0020
lookalikes = {}


# In[22]:


for customer_id in combined_features.index[:20]:  # First 20 customers based on the dataset
    similar_customers = similarity_df.loc[customer_id].nlargest(4)  # Get top 4 (including self)
    similar_customers = similar_customers.iloc[1:]  # Exclude the customer itself
    lookalikes[customer_id] = list(zip(similar_customers.index, similar_customers.values))


# In[24]:


# Step 8: Format results as required for Lookalike.csv
lookalike_data = [{"cust_id": cust_id, "similar_customers": [{"cust_id": sim[0], "score": sim[1]} for sim in sims]}
                  for cust_id, sims in lookalikes.items()]


# In[25]:


lookalike_df = pd.DataFrame(lookalike_data)


# In[29]:


# Step 9: Save to CSV
lookalike_df.to_csv("Lookalike.csv", index=False)


# In[27]:


# Display the results
print(lookalike_df.head())


# In[30]:


from IPython.display import FileLink

# Save the file and create a download link
lookalike_df.to_csv("Lookalike.csv", index=False)
FileLink("Lookalike.csv")


# In[ ]:




