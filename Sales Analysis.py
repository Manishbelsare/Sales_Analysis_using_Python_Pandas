#!/usr/bin/env python
# coding: utf-8

# # Import Necessary Libraries

# In[22]:


import pandas as pd
import os


# # Task #1: Merge the 12 months of sales data into a single Csv file

# In[23]:


df = pd.read_csv("./Sales_Data/Sales_April_2019.csv")

files = [file for file in os.listdir('./Sales_Data')]

all_months_data = pd.DataFrame()

for file in files:
    df = pd.read_csv("./Sales_Data/"+file)
    all_months_data = pd.concat([all_months_data, df])
    
all_months_data.to_csv("all_data.csv", index=False)  


# Read updated dataframe

# In[24]:


all_data = pd.read_csv("all_data.csv")
all_data.head()


# # Clean up the data

# Drop rows of NaN

# In[25]:


nan_df = all_data[all_data.isna().any(axis=1)]
nan_df.head()

all_data = all_data.dropna(how='all')
all_data.head()


# Find 'or' and delete it

# In[26]:


all_data = all_data[all_data['Order Date'].str[0:2] != 'Or']


# convert columns to the correct type

# In[27]:


all_data['Quantity Ordered'] =pd.to_numeric(all_data['Quantity Ordered']) # Make int
all_data['Price Each'] = pd.to_numeric(all_data['Price Each']) #Make float

all_data.head()


# In[ ]:





# # Augment data with additional columns

# ### Task 2: Add month Column

# In[28]:


all_data['Month'] = all_data['Order Date'].str[0:2]
all_data['Month'] = all_data['Month'].astype('int32')
all_data.head()


# # Task 3: Add a Sales column

# In[29]:


all_data['Sales'] = all_data['Quantity Ordered'] * all_data['Price Each']
all_data.head()


# ## Task 4: Add a city column

# In[30]:


# Use .apply()
def get_city(address):
    return address.split(',')[1]

def get_state(address):
    return address.split(',')[2].split(' ')[1]

all_data['city'] = all_data['Purchase Address'].apply(lambda x: f"{get_city(x)} ({get_state(x)})")

all_data.head()


# In[ ]:





# Question 1: What was the best month for sales? How much was earned that month?

# In[31]:


results = all_data.groupby('Month').sum(numeric_only=True)


# In[32]:


import matplotlib.pyplot as plt

months = range(1,13)

plt.bar(months,results['Sales'])
plt.xticks(months)
plt.ylabel('Sales in USD ($)')
plt.xlabel('Month number')
plt.show()


# # Question 2: What city had the highest number of sales?

# In[33]:


results = all_data.groupby('city').sum(numeric_only=True)
results


# In[34]:


import matplotlib.pyplot as plt

cities = [city for city, df in all_data.groupby('city')]

plt.bar(cities, results['Sales'])
plt.xticks(cities, rotation='vertical',size=8)
plt.ylabel('Sales in usd ($)')
plt.xlabel('city name')
plt.show()


# # Question 3: What time should we display advertisements to maximize likelihood of customer's buying product?

# In[35]:


all_data['Order Date'] = pd.to_datetime(all_data['Order Date'])


# In[ ]:


all_data['Hour'] = all_data['Order Date'].dt.hour
all_data['Minute'] = all_data['Order Date'].dt.minute
all_data['count'] = 1
all_data.head()


# In[39]:


hours = [hour for hour, df in all_data.groupby('Hour')]

plt.plot(hours, all_data.groupby(['Hour']).count())
plt.xticks(hours)
plt.xlabel('Hour')
plt.ylabel('Number of Orders')
plt.grid()
plt.show()

# My recommendation is around 11am (11) or 7pm (19)


# # Question 4: What products are most often sold together?

# In[44]:


df = all_data[all_data['Order ID'].duplicated(keep=False)]

df.loc[:, 'Grouped'] = df.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))

df = df[['Order ID', 'Grouped']].drop_duplicates()

df.head()


# In[68]:


# Referenced: https://stackoverflow.com/questions/52195887/counting-unique-parts-of-numbers-into-a-python-dictionary
from itertools import combinations
from collections import Counter

count = Counter()

for row in df['Grouped']:
    row_list = row.split(',')
    count.update(Counter(combinations(row_list, 2)))
    
for key, value in count.most_common(10):
    print(key, value)


# # Question 5: What product sold the most? why do you think it sold the most?

# In[72]:


product_group = all_data.groupby('Product')

product_group.sum(numeric_only=True)


# In[76]:


product_group = all_data.groupby('Product')
quantity_ordered = product_group.sum(numeric_only = True)['Quantity Ordered']

products = [product for product, df in product_group]

plt.bar(products, quantity_ordered)
plt.ylabel('Quantity Ordered')
plt.xlabel('Product')
plt.xticks(products, rotation='vertical',size=8)
plt.show()


# In[85]:


prices = all_data.groupby('Product').mean(numeric_only=True)['Price Each']

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.bar(products, quantity_ordered, color='g')
ax2.plot(products, prices, 'b-')

ax1.set_xlabel('Product Name')
ax1.set_ylabel('Quantity Ordered', color = 'g')
ax2.set_ylabel('Price ($)', color='b')
ax1.set_xticklabels(products, rotation= 'vertical', size = 8)

plt.show()


# In[ ]:




