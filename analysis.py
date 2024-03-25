import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

data: pd.DataFrame = pd.read_csv("ecommerce_customer_data.csv")
print(data.head())

# Numeric column summary breakdown: User_ID, Age, Product_Browsing_Time, Total_Pages_Viewed, Items_Added_to_Cart, Total_Purchases
numeric_summary: pd.DataFrame = data.describe()
print(numeric_summary)


# Non-numeric columns summary breakdown: Gender, Location, Device_Type
categorical_summary: pd.DataFrame = data.describe(include='object')
print(categorical_summary)


# Age distribution
f1 = plt.figure(1)
plt.hist(data['Age'])
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribution of Age')
plt.xticks(np.linspace(18,34,9))

# Gender distribution
# 261 Men vs 239 Women
# 1.16% more men than women
f2 = plt.figure(2)
gender_counts = data['Gender'].value_counts().reset_index()
plt.bar(gender_counts['Gender'], gender_counts['count'])
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Distribution of Gender')


#Product Browsing Time and Total Pages Viewed
#Scatter plot to represent data points. Use Ordinary Least Squares regression trendline to demonstrate linear relationship
#This doesnt show a particularly strong relationship between the two data points
f3 = plt.figure(3)

trendline = np.polyfit(data['Product_Browsing_Time'], data['Total_Pages_Viewed'], 1)
func = np.poly1d(trendline)

plt.scatter(data['Product_Browsing_Time'], data['Total_Pages_Viewed'])
plt.plot(data['Product_Browsing_Time'], func(data['Product_Browsing_Time']), color='red')
plt.xlabel('Product Browsing Time')
plt.ylabel('Total_Pages_Viewed')
plt.title("Relationship Between Product Browsing Time and Total Pages Viewed")

#Average Total Pages Viewed vs Gender
# Women view on average: 27.577 pages
# Men view on average: 26.820 pages
# Women on average view ~1.02% more pages than men
f4 = plt.figure(4)

gender_groups = data.groupby('Gender')['Total_Pages_Viewed'].mean().reset_index()
gender_groups.columns = ['Gender', 'Average_Total_Pages_Viewed']

plt.bar(gender_groups['Gender'], gender_groups['Average_Total_Pages_Viewed'])
plt.xlabel('Gender')
plt.ylabel('Average Total Pages Viewed')
plt.title('Relationship Between Gender and Average Total Pages Viewed')

#Average Total Pages Viewed By Devices
# Desktop: 26 pages on Average
# Mobile: 27.792 pages on Average
# Tablet: 27.669 pages on Average
# Conclusion: Target mobile users more -> more likely to view pages
f5 = plt.figure(5)
devices_groups = data.groupby('Device_Type')['Total_Pages_Viewed'].mean().reset_index()
devices_groups.columns = ['Device_Type', 'Average_Total_Pages_Viewed']

plt.bar(devices_groups['Device_Type'], devices_groups['Average_Total_Pages_Viewed'])
plt.xlabel('Device Type')
plt.ylabel('Average_Total_Pages_Viewed')
plt.title('Relationship Between Device Type and Average Total Pages Viewed')


#Customer Lifetime Value
# Get CLV as proportion of total pages viewed and total purchases to age
f6 = plt.figure(6)
data['CLV'] = (data['Total_Purchases'] * data['Total_Pages_Viewed'])/data['Age']

data['Segment'] = pd.cut(data['CLV'], bins=[1,2.5,5, float('inf')], labels=['Low Value', 'Medium Value', 'High Value'])

segment_counts = data['Segment'].value_counts().reset_index()
segment_counts.columns = ['Segment', 'Count']

# Bar chart visualises customer segments
plt.bar(segment_counts['Segment'], segment_counts['Count'])

plt.xlabel('Value Segment')
plt.ylabel('Count')
plt.title('Customer Segmentation by CLV')
plt.show()
