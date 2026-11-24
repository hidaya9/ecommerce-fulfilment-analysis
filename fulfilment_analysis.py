"""
E-Commerce Fulfilment Analysis

Explores courier performance, shipping efficiency and refund-related
losses to support data-driven operational recommendations.

The original datasets are not included in this repository.
Synthetic sample files are provided to illustrate the data structure
and are not intended to reproduce the reported findings.
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
orders = pd.read_csv('orders.csv')
refunds = pd.read_csv('refunds.csv')
shipments = pd.read_csv('shipments.csv')


print("\n=== EXPLORATORY AND STATISTICAL ANALYSIS ===")

# 1. Parcel Volume by Courier

# Each row in the shipments dataset represents one parcel, so the number
# of shipment records for each courier gives the total parcels handled.
total_parcels = (shipments['courier'].value_counts())
max_courier = total_parcels.idxmax()

print("\n1. Parcel Volume by Courier")
print(f"Courier handling the most parcels: {max_courier}")


# 2. Overall Average Shipping Time

# Shipping time is calculated using delivered parcels only, as returned and
# lost parcels do not have a completed delivery time.
# Round shipping days to 1dp.
avg_shipping_days = round(
    shipments[shipments['status'] == 'Delivered']['shipping_days'].mean(), 1
    )

print("\n2. Overall Average Shipping Time")
print(f"Average shipping time: {avg_shipping_days} days")


#3. Average Shipping Time by Region

# Only delivered parcels are included because returned and lost parcels
# do not have a completed shipping time.
# The shipments dataset does not contain region, so it is joined to orders
# using order_id to add the region associated with each parcel.
delivered_shipments = shipments[shipments["status"] == "Delivered"]
joined_df = pd.merge(
    left=orders,
    right=delivered_shipments,
    on='order_id',
    how='inner'
)

# Calculate the mean shipping time of delivered parcels within each region.
avg_shipping_per_region = joined_df.groupby('region')['shipping_days'].mean()
max_region = avg_shipping_per_region.idxmax()

print("\n3. Average Shipping Time by Region")
print(f"Region with longest average shipping time: {max_region}")


#4. Total Order Value by Product Category

# order_value is recorded at order level, so no join to shipment or refund
# data is required. Order values are summed within each product category.
category_totals = orders.groupby('category')['order_value'].sum()
max_category = category_totals.idxmax()

print("\n4. Total Order Value by Product Category")
print(f"Category with highest total order value: {max_category}")


# 5. Refund Rates by Product Category

# The refunds dataset contains one row per refunded order. It is joined to
# orders using order_id so each refunded order can be assigned a category.
orders_refunds = pd.merge(
    left=orders,
    right=refunds,
    on='order_id',
    how='inner'
)

# Count refunded orders within each category.
refunds_per_category = orders_refunds.groupby('category')['refund_id'].count()

# The denominator is all orders in each category, as the question defines
# refund rate as refunded orders / all orders.
orders_per_category = orders['category'].value_counts()

refund_rate_per_category = (refunds_per_category / orders_per_category) * 100
worst_category = refund_rate_per_category.idxmax()

print("\n5. Refund Rates by Product Category")
print(refund_rate_per_category)
print(f"Category with highest refund rate: {worst_category}")


# 6. Statistical Significance of Refund Rate Differences

# worst_category already calculated above in 5.
best_category = refund_rate_per_category.idxmin()

# Get total and refunded order counts for the best and worst categories.
worst_orders = orders_per_category[worst_category]
best_orders = orders_per_category[best_category]

worst_refunds = refunds_per_category[worst_category]
best_refunds = refunds_per_category[best_category]

# Calculate the number of non-refunded orders in each category.
worst_not_refunded = worst_orders - worst_refunds
best_not_refunded = best_orders - best_refunds

# A chi-square test of independence is used to compare refund rates
# between the categories with the highest and lowest observed rates.
#
# H0: Refund status is independent of product category.
# H1: Refund status is associated with product category.
#
# The test uses a 2x2 contingency table of refunded and
# non-refunded orders for the two categories.

# Create a 2x2 contingency table of refund status by category.
contingency_table = [
    [worst_refunds, worst_not_refunded],
    [best_refunds, best_not_refunded]
]

contingency_df = pd.DataFrame(
    data=contingency_table,
    columns=['Refunded', 'Not Refunded'],
    index=[worst_category, best_category]
)

# Run a chi-square test of independence.
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

print("\n6. Statistical Significance of Refund Rate Differences")
# Displaying results
print(f"\nChi-Square Statistic: {chi2_stat:.4f}")
print(f"Degrees of Freedom: {dof}")
print("Expected Frequencies Table:\n", expected)
print(f"P-value: {p_value:.4f}")

# Significance level
alpha = 0.05

# Decision
if p_value < alpha:
    print("\nReject Null Hypothesis: There is a significant association between category and refund status.")
else:
    print("\nFail to Reject Null Hypothesis: Insufficient evidence of a statistically significant association between product category and refund status.")

# A p-value below 0.05 provides evidence of an association
# between product category and refund status.
# Otherwise, there is insufficient evidence to reject H0.


#7. Returned and Refunded Parcels by Courier

# Filter to returned parcels, then inner join to refunds so only parcels
# also belonging to refunded orders remain.
returned_shipments = shipments[shipments['status'] == 'Returned']

returned_and_refunded = pd.merge(
    left=returned_shipments,
    right=refunds,
    on='order_id',
    how='inner'
)
# Count returned-and-refunded parcels for each courier.
grouped_returns = returned_and_refunded.groupby('courier')['shipment_id'].count()

# Divide by all parcels handled by each courier.
# total_parcels was calculated earlier in 1.
percentage_per_courier = (grouped_returns / total_parcels) * 100 

lowest_courier = percentage_per_courier.idxmin()

print("\n7. Returned and Refunded Parcels by Courier")
print(f"Courier with lowest returned and refunded parcel rate: {lowest_courier}")


# 8. Net Revenue per Parcel by Courier and Category

# Net revenue is assumed as order value minus refund amount.

# Left join refunds to orders so non-refunded orders are retained.
orders_refunds_joined = pd.merge(
    left=orders,
    right=refunds,
    on='order_id',
    how='left'
)[['order_id', 'category', 'order_value', 'refund_amount']]

# Non-refunded orders have no refund record, so missing refund amounts
# are treated as zero.
orders_refunds_joined['refund_amount'] = (
    orders_refunds_joined['refund_amount'].fillna(0)
)

orders_refunds_joined['net_revenue'] = (
    orders_refunds_joined['order_value']
    - orders_refunds_joined['refund_amount']
)

# Revenue is recorded at order level, while courier data is at parcel level.
# Count the number of parcels in each order so order-level net revenue can
# be allocated across its parcels. 

parcels_per_order = (
    shipments.groupby('order_id')['shipment_id'].count().to_frame()
)

parcels_per_order = parcels_per_order.rename(
    columns={'shipment_id': 'parcel_count'}
)

orders_with_parcels = pd.merge(
    left=orders_refunds_joined,
    right=parcels_per_order,
    on='order_id',
    how='inner'
)

# Assumption: where an order contains multiple parcels, its net revenue is
# allocated equally across those parcels because no parcel-level revenue
# allocation is provided.
orders_with_parcels['revenue_per_parcel'] = (
    orders_with_parcels['net_revenue']
    / orders_with_parcels['parcel_count']
)

# Join the parcel-level revenue back to shipments to identify the courier
# responsible for each parcel.
parcel_revenues = pd.merge(
    left=shipments,
    right=orders_with_parcels,
    on='order_id',
    how='left'
)[
    ['shipment_id', 'order_id', 'courier', 'category', 'order_value',
     'refund_amount', 'net_revenue', 'parcel_count', 'revenue_per_parcel']
]

# Calculate average net revenue per parcel for each courier & category.
parcel_revenue_by_courier_category = (
    parcel_revenues
    .groupby(['courier', 'category'])['revenue_per_parcel']
    .mean()
)

highest_combo = parcel_revenue_by_courier_category.idxmax()

print("\n8. Net Revenue per Parcel by Courier and Category")
print(f"Highest courier & category combination: {highest_combo}")



print("\n=== BUSINESS ANALYSIS AND RECOMMENDATIONS ===")


# 1. Single Courier Performance Comparison

# Filter shipments to delivered parcels, then group them by courier.
# Compare delivery speed, delivery rate and return rate across couriers.
avg_shipping_by_courier = (
    shipments[shipments['status'] == 'Delivered']
    .groupby('courier')['shipping_days']
    .mean()
)

delivery_rate_by_courier = (
    shipments[shipments['status'] == 'Delivered']['courier'].value_counts()
    / total_parcels
) * 100

return_rate_by_courier = (
    shipments[shipments['status'] == 'Returned']['courier'].value_counts()
    / total_parcels
) * 100

# Use the parcel-level revenue calculated in Section 1, 8. to compare
# average net revenue per parcel across couriers.
revenue_per_parcel_by_courier = (
    parcel_revenues.groupby('courier')['revenue_per_parcel'].mean()
)

# Refunds are order-level while courier data is parcel-level, so orders
# handled by multiple couriers cannot be clearly attributed to one courier.
# Count the number of unique couriers used for each order.
couriers_per_order = shipments.groupby('order_id')['courier'].nunique()

print("\n1. Single Courier Performance Comparison")
print("Number of couriers used per order:")
print(couriers_per_order.value_counts())

# Filter the courier count series to orders where the number of unique
# couriers equals 1. .index returns the corresponding order_ids, which
# can then be used to filter the shipments dataset.
single_courier_order_ids = (
    couriers_per_order[couriers_per_order == 1].index
)

# Keep shipment records belonging to those single-courier orders.
single_courier_shipments = shipments[
    shipments['order_id'].isin(single_courier_order_ids)
]

# An order can contain multiple parcels, so drop duplicates to create
# one order-courier row per single-courier order.
single_courier_orders = (
    single_courier_shipments[['order_id', 'courier']]
    .drop_duplicates()
)

# Left join refunds so refunded and non-refunded orders are both retained.
single_courier_orders = pd.merge(
    single_courier_orders,
    refunds[['order_id', 'refund_id']],
    on='order_id',
    how='left'
)

# Create boolean column of whether each order was refunded.
single_courier_orders['refunded'] = (
    single_courier_orders['refund_id'].notna()
)

# The mean of the boolean column gives the proportion of orders refunded.
refund_rate_by_courier = (
    single_courier_orders.groupby('courier')['refunded'].mean() * 100
)

# Calculate average refund loss across all orders for each courier.
# Non-refunded orders are assigned a refund amount of £0.
single_courier_refunds = pd.merge(
    single_courier_orders[['order_id', 'courier']],
    refunds[['order_id', 'refund_amount']],
    on='order_id',
    how='left'
)

single_courier_refunds['refund_amount'] = (
    single_courier_refunds['refund_amount'].fillna(0)
)

refund_loss_per_order = (
    single_courier_refunds.groupby('courier')['refund_amount'].mean()
)

# Creating visuals
refund_rate_by_courier.plot(kind='bar')
plt.title('Refund Rate by Courier - Single-Courier Orders')
plt.xlabel('Courier')
plt.ylabel('Refund Rate (%)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

refund_loss_per_order.plot(kind='bar')
plt.title('Average Refund Loss per Order by Courier')
plt.xlabel('Courier')
plt.ylabel('Average Refund Loss per Order (£)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

courier_comparison = pd.DataFrame({
    'Avg Delivery Time (Days)': avg_shipping_by_courier,
    'Delivery Rate (%)': delivery_rate_by_courier,
    'Return Rate (%)': return_rate_by_courier,
    'Refund Rate* (%)': refund_rate_by_courier,
    'Avg Refund Loss per Order (£)': refund_loss_per_order,
    'Net Revenue per Parcel (£)': revenue_per_parcel_by_courier
}).round(2)

print(courier_comparison)



# 2. Regional and Category-Based Courier Analysis

# Category & courier analysis

# Add product category to each single-courier order so refund rates
# can be compared between couriers within each category.
category_courier_orders = pd.merge(
    single_courier_orders[['order_id', 'courier']],
    orders[['order_id', 'category']],
    on='order_id',
    how='left'
)

# Add refund_id using a left join so both refunded and non-refunded
# orders remain in the analysis.
category_courier_orders = pd.merge(
    category_courier_orders,
    refunds[['order_id', 'refund_id']],
    on='order_id',
    how='left'
)

# Boolean column showing whether each order was refunded.
category_courier_orders['refunded'] = (
    category_courier_orders['refund_id'].notna()
)

# Group by category and courier. The mean of the boolean refunded
# column gives the refund proportion.
refund_rate_category_courier = (
    category_courier_orders
    .groupby(['category', 'courier'])['refunded']
    .mean() * 100
)

# Reshape results so each courier is a column for easier comparison.
category_comparison = refund_rate_category_courier.unstack()

category_comparison.columns = [
    'Arrow Refund Rate (%)',
    'Bolt Refund Rate (%)',
    'Cyclone Refund Rate (%)'
]
category_comparison = category_comparison.round(2)

print("\n2. Regional and Category-Based Courier Analysis")
print("Refund rate by category and courier:")
print(category_comparison)


# Region & courier analysis

# Add region to each single-courier order so refund rates can also
# be compared between couriers within each region.
region_courier_orders = pd.merge(
    single_courier_orders[['order_id', 'courier']],
    orders[['order_id', 'region']],
    on='order_id',
    how='left'
)

# Add refund information while retaining non-refunded orders.
region_courier_orders = pd.merge(
    region_courier_orders,
    refunds[['order_id', 'refund_id']],
    on='order_id',
    how='left'
)

region_courier_orders['refunded'] = (
    region_courier_orders['refund_id'].notna()
)

# Calculate refund rate for each region & courier combination.
refund_rate_region_courier = (
    region_courier_orders
    .groupby(['region', 'courier'])['refunded']
    .mean() * 100
)

# Reshape the results into a region & courier table for comparison
# and use the same table to create visual.
region_refund_plot = (
    refund_rate_region_courier
    .unstack()
    .round(2)
)

region_refund_plot.plot(kind='bar')

plt.title('Refund Rate by Region and Courier')
plt.xlabel('Region')
plt.ylabel('Refund Rate (%)')
plt.xticks(rotation=0)
plt.legend(title='Courier')
plt.tight_layout()
plt.show()



# 3. Refund Reasons and Operational Improvements

# Count refunds by recorded reason to identify the main sources
# of refund-related losses.
refund_reason_counts = refunds['refund_reason'].value_counts()

# Convert the counts to percentages of all refunded orders.
refund_reason_percentages = (
    refund_reason_counts / len(refunds)
) * 100

refund_reason_summary = pd.DataFrame({
    'Refund Count': refund_reason_counts,
    'Percentage of Refunds (%)': refund_reason_percentages
}).round(2)

print("\n3. Refund Reasons and Operational Improvements")
print("Refund reasons:")
print(refund_reason_summary)

# Wrong Item and Damaged are combined because both can potentially
# be addressed through improvements to picking, packing and quality control.
fulfilment_error_refunds = (
    refund_reason_counts['Wrong Item']
    + refund_reason_counts['Damaged']
)

fulfilment_error_percentage = (
    fulfilment_error_refunds / len(refunds)
) * 100

print(
    f"\nWrong Item + Damaged: {fulfilment_error_refunds} refunds "
    f"({fulfilment_error_percentage:.1f}% of all refunds)"
)

# Late is assessed separately because it supports the recommendation
# to introduce proactive management of orders at risk of delay.
late_refunds = refund_reason_counts['Late']
late_refund_percentage = (late_refunds / len(refunds)) * 100

print(
    f"Late: {late_refunds} refunds "
    f"({late_refund_percentage:.1f}% of all refunds)"
)