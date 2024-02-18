# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 00:27:05 2024

@author: mahmoud.ali
"""

import pandas as pd
import pyodbc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
# Replace these variables with your actual SQL Server connection details
server = 'MAHMOUDALI-NB'
database = 'freezone'
username = 'sa'
password = 'Gladiator*1234'
driver = 'SQL Server'  # Adjust driver if necessary

# Create a connection string
conn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'

# Establish a connection
conn = pyodbc.connect(conn_str)

#Replace 'your_table' with the actual table name
table_name = 'Sales'
table_name1 = 'Masterdata'
table_name2='veh_spec'
# SQL query to retrieve data
data_query=f' select  spec_Tank_Capcity,spec_AVG_fuel_Consum,spec_Engine_cc,spec_Horsepower,spec_Number_of_doors,spec_Body_Style,spec_Fuel_type,spec_vehicle_type ,[Vehicle desc],  [Inv No#],[Sales Type],[Chassis],[brand],[Purchases] ,[New/Used],[Inv Date],[Location],Salesman,[Invoiced To],[Invoice Account],[Customer Name],Manufacturer,Model,[Variant Code],YEAR,m.VSB,[Ezz Options Cost],m.Profit ,Shipname,[Ship date],[Total Cost],[Total Item Cost],Sales*-1 as Sales ,[Export Charges],[GAFI Duty  0#1%],[GAFI DUTY  2%],[land Insurance],[Local Import Charges],[Local Marine Insurance],Storage,[Vehicle discounts] ,[Factory options cost of sales],Transportation,[PDI Elezz For Service],[Sea Freigh],[Other Expenses],GPS  from {table_name1} m join {table_name} s on m.VSB=s.VSB   join {table_name2}  v on m.Manufacturer=v.spec_brand  where v.spec_Vehicle_desc=m.[Vehicle desc]'
# Read data into a DataFrame

df = pd.read_sql(data_query, conn)
pd.set_option('display.max.row',None)
pd.set_option('display.max.column',None)

# count_model=f'''select count(*) as count_model,Model from {table_name1} where  Manufacturer ='VOLVO' group by Model'''
# c = pd.read_sql(count_model, conn)
# print(c)

# Close the connection
conn.close()
print(df.dtypes)
print(df['brand'].unique())
pd.set_option('display.float.format',lambda x :'%.2f' %x)
df['YEAR'].fillna('2023',inplace=True)
df['Y']=df['Inv Date'].str[6:]
df['Month']=df['Inv Date'].str[3:5]
# print(df['Y'].head(10))
plt.figure(figsize=(10,5))
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i],ha='center')
     
        
     
        
        


def graphs_total_profit_sales_cost_item_cost_allbrand_byyear(y):
    
    if __name__ == '__main__':
        measure_total=df.loc[df['Y']==y].groupby(['brand']).sum()[['Profit','Sales','Total Cost','Total Item Cost','Purchases']].sort_values(['Sales'], ascending=False)
        T_profit=pd.DataFrame(measure_total.reset_index())
        print(T_profit)   
        plt.figure(figsize=(10,5))
        plt.bar(T_profit['brand'],T_profit['Profit'])
        addlabels(T_profit['brand'],T_profit['Profit'])
        plt.title(f' Profit By brand {y}')
        plt.xlabel('brand')
        plt.ylabel('  Profit')
        plt.show()
        plt.figure(figsize=(10,5))
        plt.bar(T_profit['brand'],T_profit['Sales'])
        addlabels(T_profit['brand'],T_profit['Sales'])
        plt.title(f'  Sales By brand {y}')
        plt.xlabel('brand')
        plt.ylabel('  Sales')
        plt.show()
        plt.figure(figsize=(10,5))
        plt.bar(T_profit['brand'],T_profit['Purchases'])
        addlabels(T_profit['brand'],T_profit['Purchases'])
        plt.title(f'Purchases By brand {y}')
        plt.xlabel('brand')
        plt.ylabel('Purchases')
        plt.show()    
        plt.figure(figsize=(10,5))
        plt.bar(T_profit['brand'],T_profit['Total Cost'])
        addlabels(T_profit['brand'],T_profit['Total Cost'])
        plt.title(f'  Total Cost By brand {y}')
        plt.xlabel('brand')
        plt.ylabel(' Cost')
        plt.show()
        plt.figure(figsize=(10,5))
        plt.bar(T_profit['brand'],T_profit['Total Item Cost'])
        addlabels(T_profit['brand'],T_profit['Total Item Cost'])
        plt.title(f'  Item Cost By brand {y}')
        plt.xlabel('brand')
        plt.ylabel('  Item Cost')
        plt.show()      
        plt.figure(figsize=(20,10))
        plt.subplot(3,2,1)
        plt.bar(T_profit['brand'],T_profit['Profit'])
        plt.title(f'Profit By brand {y}')
        plt.xlabel('brand')
        plt.ylabel('Profit')
        plt.subplot(3,2,2)
        plt.bar(T_profit['brand'],T_profit['Sales'])
        plt.title(f'Sales By brand {y}')
        plt.xlabel('brand')
        plt.ylabel('Sales') 
        plt.subplot(3,2,3)
        
        plt.bar(T_profit['brand'],T_profit['Purchases'])
        plt.title(f'Purchases By brand {y}')
        plt.xlabel('brand')
        plt.ylabel('Purchases')  
        plt.subplot(3,2,4)
        plt.bar(T_profit['brand'],T_profit['Total Cost'])
        plt.title(f'  Cost By brand {y}')
        plt.xlabel('brand')
        plt.ylabel('  Cost')  
        plt.subplot(3,2,5)
        plt.title(f'  Item Cost By brand {y}')
        plt.bar(T_profit['brand'],T_profit['Total Item Cost'])
        plt.title('  Item Cost By brand')
        plt.xlabel('brand')
        plt.ylabel('  Item Cost')  
        plt.show()
# graphs_total_profit_sales_cost_item_cost_allbrand_byyear('2024')    
# graphs_total_profit_sales_cost_item_cost_allbrand_byyear('2023')    
    
def graphs_total_profit_sales_cost_item_cost_allbrand_allyears():
    
    if __name__ == '__main__':
        measure_total=df.groupby(['brand']).sum()[['Profit','Sales','Total Cost','Total Item Cost','Purchases']].sort_values(['Sales'], ascending=False)
        T_profit=pd.DataFrame(measure_total.reset_index())
        print(T_profit)   
        plt.figure(figsize=(10,5))
        plt.bar(T_profit['brand'],T_profit['Profit'])
        addlabels(T_profit['brand'],T_profit['Profit'])
        plt.title('  Profit By brand')
        plt.xlabel('brand')
        plt.ylabel('  Profit')
        plt.show()
        plt.figure(figsize=(10,5))
        plt.bar(T_profit['brand'],T_profit['Sales'])
        addlabels(T_profit['brand'],T_profit['Sales'])
        plt.title('  Sales By brand')
        plt.xlabel('brand')
        plt.ylabel('  Sales')
        plt.show()
        plt.figure(figsize=(10,5))
        plt.bar(T_profit['brand'],T_profit['Purchases'])
        addlabels(T_profit['brand'],T_profit['Purchases'])
        plt.title('Purchases By brand')
        plt.xlabel('brand')
        plt.ylabel('Purchases')
        plt.show()    
        plt.figure(figsize=(10,5))
        plt.bar(T_profit['brand'],T_profit['Total Cost'])
        addlabels(T_profit['brand'],T_profit['Total Cost'])
        plt.title('  Total Cost By brand')
        plt.xlabel('brand')
        plt.ylabel(' Cost')
        plt.show()
        plt.figure(figsize=(10,5))
        plt.bar(T_profit['brand'],T_profit['Total Item Cost'])
        addlabels(T_profit['brand'],T_profit['Total Item Cost'])
        plt.title('  Item Cost By brand')
        plt.xlabel('brand')
        plt.ylabel('  Item Cost')
        plt.show()      
        plt.figure(figsize=(20,10))
        plt.subplot(3,2,1)
        plt.bar(T_profit['brand'],T_profit['Profit'])
        plt.title('Profit By brand')
        plt.xlabel('brand')
        plt.ylabel('Profit')
        plt.subplot(3,2,2)
        plt.bar(T_profit['brand'],T_profit['Sales'])
        plt.title('Sales By brand')
        plt.xlabel('brand')
        plt.ylabel('Sales') 
        plt.subplot(3,2,3)
        plt.title('Purchases By brand')
        plt.bar(T_profit['brand'],T_profit['Purchases'])
        plt.title('Purchases By brand')
        plt.xlabel('brand')
        plt.ylabel('Purchases')  
        plt.subplot(3,2,4)
        plt.bar(T_profit['brand'],T_profit['Total Cost'])
        plt.title('  Cost By brand')
        plt.xlabel('brand')
        plt.ylabel('  Cost')  
        plt.subplot(3,2,5)
        plt.title('  Item Cost By brand')
        plt.bar(T_profit['brand'],T_profit['Total Item Cost'])
        plt.title('  Item Cost By brand')
        plt.xlabel('brand')
        plt.ylabel('  Item Cost')  
        plt.show()      
        
# graphs_total_profit_sales_cost_item_cost_allbrand_allyears()    
def graphs_total_profit_sales_cost_item_cost_bymodel(b,y):
    
    if __name__ == '__main__':
        measure_total=df.loc[(df['brand']==b) &(df['Y']==y) ].groupby(['Model']).sum()[['Profit','Sales','Total Cost','Total Item Cost']].sort_values(['Profit'], ascending=False)
        T_profit=pd.DataFrame(measure_total.reset_index())
        print(T_profit)   
        plt.figure(figsize=(10,5))
        plt.bar(T_profit['Model'],T_profit['Profit'])
        addlabels(T_profit['Model'],T_profit['Profit'])
        plt.title('  Profit By Model')
        plt.xticks(rotation='vertical',size=8)
        plt.xlabel('brand')
        plt.ylabel('  Profit')
        plt.show()
        plt.figure(figsize=(10,5))
        plt.bar(T_profit['Model'],T_profit['Sales'])
        addlabels(T_profit['Model'],T_profit['Sales'])
        plt.title('  Sales By Model')
        plt.xlabel('Model')
        plt.ylabel('  Sales')
        plt.show()
        plt.figure(figsize=(10,5))
        plt.bar(T_profit['Model'],T_profit['Total Cost'])
        addlabels(T_profit['Model'],T_profit['Total Cost'])
        plt.title('  Total Cost By Model')
        plt.xticks(rotation='vertical',size=8)
        plt.xlabel('Model')
        plt.ylabel(' Cost')
        plt.show()
        plt.figure(figsize=(10,5))
        plt.bar(T_profit['Model'],T_profit['Total Item Cost'])
        addlabels(T_profit['Model'],T_profit['Total Item Cost'])
        plt.title('  Item Cost By Model')
        plt.xticks(rotation='vertical',size=8)
        plt.xlabel('ModelItem Cost')
        plt.show()      
        plt.figure(figsize=(20,10))
        plt.subplot(2,2,1)
        plt.bar(T_profit['Model'],T_profit['Profit'])
        plt.title('Profit By Model')
        plt.xticks(rotation='vertical',size=8)
        plt.xticks(rotation='vertical',size=8)
        plt.xlabel('Model')
        plt.ylabel('Profit')
        plt.subplot(2,2,2)
        plt.bar(T_profit['Model'],T_profit['Sales'])
        plt.title('Sales By Model')
        plt.xticks(rotation='vertical',size=8)
        plt.xlabel('Model')
        plt.ylabel('Sales')  
        plt.subplot(2,2,3)
        plt.bar(T_profit['Model'],T_profit['Total Cost'])
        plt.title('  Cost By Model')
        plt.xticks(rotation='vertical',size=8)
        plt.xlabel('Model')
        plt.ylabel('  Cost')  
        plt.subplot(2,2,4)
        plt.title('  Item Cost By Model')
        plt.bar(T_profit['Model'],T_profit['Total Item Cost'])
        plt.title('  Item Cost By Model')
        plt.xticks(rotation='vertical',size=8)
        plt.xlabel('Model')
        plt.ylabel('  Item Cost')  
        plt.show()
        print('-'*70)
# graphs_total_profit_sales_cost_item_cost_bymodel('VOLVO','2024')
def graphs_profit_sales_cost_item_cost_bymont(y):
    
    if __name__ == '__main__':
        measure_total=df.loc[df['Y']==y].groupby(['Month']).sum()[['Profit','Sales','Total Cost','Total Item Cost']].sort_values(['Month'], ascending=True)
        T_profit=pd.DataFrame(measure_total.reset_index())
        print(measure_total)   
        plt.figure(figsize=(10,5))
        plt.bar(T_profit['Month'],T_profit['Profit'])
        addlabels(T_profit['Month'],T_profit['Profit'])
        plt.title('Profit By Month')
        plt.xlabel('Month')
        plt.ylabel('Profit')
        plt.show()
        plt.figure(figsize=(10,5))
        plt.bar(T_profit['Month'],T_profit['Sales'])
        addlabels(T_profit['Month'],T_profit['Sales'])
        plt.title('Sales By Month')
        plt.xlabel('Month')
        plt.ylabel('Sales')
        plt.show()
        plt.figure(figsize=(10,5))
        plt.bar(T_profit['Month'],T_profit['Total Cost'])
        addlabels(T_profit['Month'],T_profit['Total Cost'])
        plt.title('Total Cost By Month')
        plt.xlabel('Month')
        plt.ylabel('Total Cost')
        plt.show()
        plt.figure(figsize=(10,5))
        plt.bar(T_profit['Month'],T_profit['Total Item Cost'])
        addlabels(T_profit['Month'],T_profit['Total Item Cost'])
        plt.title('Total Item Cost By Month')
        plt.xlabel('Month')
        plt.ylabel('Total Item Cost')
        plt.show()      
        plt.figure(figsize=(20,10))
        plt.subplot(2,2,1)
        plt.bar(T_profit['Month'],T_profit['Profit'])
        plt.title('Profit By Month')
        plt.xlabel('Month')
        plt.ylabel('Profit')
        plt.subplot(2,2,2)
        plt.bar(T_profit['Month'],T_profit['Sales'])
        plt.title('Sales By Month')
        plt.xlabel('Month')
        plt.ylabel('Sales')  
        plt.subplot(2,2,3)
        plt.bar(T_profit['Month'],T_profit['Total Cost'])
        plt.title('Total Cost By Month')
        plt.xlabel('Month')
        plt.ylabel('Total Cost')  
        plt.subplot(2,2,4)
        plt.title('Total Item Cost By Month')
        plt.bar(T_profit['Month'],T_profit['Total Item Cost'])
        plt.title('Total Item Cost By Month')
        plt.xlabel('Month')
        plt.ylabel('Total Item Cost')  
        plt.show()
# graphs_profit_sales_cost_item_cost_bymont('2023')
def graphs_mean_profit_sales_cost_item_cost_allbrand(y):
    
    if __name__ == '__main__':
        measure_mean=df.loc[df['Y']==y].groupby(['brand']).mean()[['Profit','Sales','Total Cost','Total Item Cost']]
        mean_measure=pd.DataFrame(measure_mean.reset_index())
        print(mean_measure)   
        plt.figure(figsize=(10,5))
        plt.bar(mean_measure['brand'],mean_measure['Profit'])
        addlabels(mean_measure['brand'],mean_measure['Profit'])
        plt.title('AVG Profit By brand')
        plt.xlabel('brand')
        plt.ylabel('AVG Profit')
        plt.show()
        plt.figure(figsize=(10,5))
        plt.bar(mean_measure['brand'],mean_measure['Sales'])
        addlabels(mean_measure['brand'],mean_measure['Sales'])
        plt.title('AVG Sales By brand')
        plt.xlabel('Month')
        plt.ylabel('AVG Sales')
        plt.show()
        plt.figure(figsize=(10,5))
        plt.bar(mean_measure['brand'],mean_measure['Total Cost'])
        addlabels(mean_measure['brand'],mean_measure['Total Cost'])
        plt.title('AVG Total Cost By brand')
        plt.xlabel('Month')
        plt.ylabel('AVG Cost')
        plt.show()
        plt.figure(figsize=(10,5))
        plt.bar(mean_measure['brand'],mean_measure['Total Item Cost'])
        addlabels(mean_measure['brand'],mean_measure['Total Item Cost'])
        plt.title('AVG Item Cost By brand')
        plt.xlabel('brand')
        plt.ylabel('AVG Item Cost')
        plt.show()      
        plt.figure(figsize=(20,10))
        plt.subplot(2,2,1)
        plt.bar(mean_measure['brand'],mean_measure['Profit'])
        plt.title('AVG Profit By brand')
        plt.xlabel('brand')
        plt.ylabel('Profit')
        plt.subplot(2,2,2)
        plt.bar(mean_measure['brand'],mean_measure['Sales'])
        plt.title('Sales By brand')
        plt.xlabel('brand')
        plt.ylabel('Sales')  
        plt.subplot(2,2,3)
        plt.bar(mean_measure['brand'],mean_measure['Total Cost'])
        plt.title('AVG Cost By brand')
        plt.xlabel('brand')
        plt.ylabel('AVG Cost')  
        plt.subplot(2,2,4)
        plt.title('AVG Item Cost By brand')
        plt.bar(mean_measure['brand'],mean_measure['Total Item Cost'])
        plt.title('AVG Item Cost By Month')
        plt.xlabel('brand')
        plt.ylabel('AVG Item Cost')  
        plt.show()
# graphs_mean_profit_sales_cost_item_cost_allbrand('2024')
def graphs_mean_profit_sales_cost_item_cost_bymodel(b,y):
    
    if __name__ == '__main__':
        measure_mean=df.loc[(df['brand']==b) &(df['Y']==y)].groupby(['Model']).mean()[['Profit','Sales','Total Cost','Total Item Cost']]
        mean_measure=pd.DataFrame(measure_mean.reset_index())
        print(mean_measure)   
        plt.figure(figsize=(10,5))
        plt.bar(mean_measure['Model'],mean_measure['Profit'])
        addlabels(mean_measure['Model'],mean_measure['Profit'])
        plt.title('AVG Profit By Model')
        plt.xlabel('Model')
        plt.ylabel('AVG Profit')
        plt.show()
        plt.figure(figsize=(10,5))
        plt.bar(mean_measure['Model'],mean_measure['Sales'])
        addlabels(mean_measure['Model'],mean_measure['Sales'])
        plt.title('AVG Sales By Model')
        plt.xlabel('Model')
        plt.ylabel('AVG Sales')
        plt.show()
        plt.figure(figsize=(10,5))
        plt.bar(mean_measure['Model'],mean_measure['Total Cost'])
        addlabels(mean_measure['Model'],mean_measure['Total Cost'])
        plt.title('AVG Total Cost By Model')
        plt.xlabel('Model')
        plt.ylabel('AVG Cost')
        plt.show()
        plt.figure(figsize=(10,5))
        plt.bar(mean_measure['Model'],mean_measure['Total Item Cost'])
        addlabels(mean_measure['Model'],mean_measure['Total Item Cost'])
        plt.title('AVG Item Cost By Model')
        plt.xlabel('Model')
        plt.ylabel('AVG Item Cost')
        plt.show()      
        plt.figure(figsize=(20,10))
        plt.subplot(2,2,1)
        plt.bar(mean_measure['Model'],mean_measure['Profit'])
        plt.title('AVG Profit By Model')
        plt.xlabel('Model')
        plt.ylabel('Profit')
        plt.subplot(2,2,2)
        plt.bar(mean_measure['Model'],mean_measure['Sales'])
        plt.title('AVG Sales By Model')
        plt.xlabel('Model')
        plt.ylabel('Sales')  
        plt.subplot(2,2,3)
        plt.bar(mean_measure['Model'],mean_measure['Total Cost'])
        plt.title('AVG Cost By brand')
        plt.xlabel('Model')
        plt.ylabel('AVG Cost')  
        plt.subplot(2,2,4)
        plt.title('AVG Item Cost By Model')
        plt.bar(mean_measure['Model'],mean_measure['Total Item Cost'])
        plt.title('AVG Item Cost By Month')
        plt.xlabel('Model')
        plt.ylabel('AVG Item Cost')  
        plt.show()
# graphs_mean_profit_sales_cost_item_cost_bymodel('VOLVO','2023')
def mode_profit_sales_by_model(b,y):
    print(f'the most amounts profit and sales are  duplicated for brand {b}')
    print(df.loc[(df['brand']==b) &(df['Y']==y)].groupby(['Model'])[['Profit','Sales']].agg(pd.Series.mode))
    # print(df.loc[(df['brand']==b) &(df['Y']==y)].groupby(['Model'])[['Profit','Sales']].agg(pd.Series.median)
mode_profit_sales_by_model('DS','2023')  

def func_mode_for_models_by_brand(y):
  print('the most models saled   by brand ')
  print(df.loc[df['Y']==y].groupby(['brand'])['Model'].agg(pd.Series.mode))
# func_mode_for_models_by_brand('2023')

def func_mode_for_vehicle_desc_by_brand(y):
  print('the most vehicle desc saled   by model ')  
  print(df.loc[df['Y']==y].groupby(['brand'])['Vehicle desc'].agg(pd.Series.mode))
# func_mode_for_vehicle_desc_by_brand('2024')
  
def measure_variance_allbrands(measure,y):
 
    print(f'Standard deviation  for Measure {measure} by brands')
    print(df.loc[df['Y']==y].groupby(['brand'])[measure].std().sort_values(ascending=False))  
# measure_variance_allbrands('Profit','2024')

def measure_variance_bybrand(measure,b,y): 
    v_f=df.loc[(df['brand']==b) &(df['Y']==y)].groupby(['brand'])[measure].std().sort_values(ascending=False) 
    print(f'The Standard Variation  for Measure {measure} for brand {b} ---> {v_f.iloc[0]}')
    data=df.loc[df['brand']==b]
    brand_data=pd.DataFrame(data.reset_index())
    mean_brand=brand_data[measure].median()
    max_v=mean_brand+v_f
    min_v=mean_brand-v_f
    print(f'The mean {measure} for BRAND {b} ---> {mean_brand}')
    plt.figure(figsize=(10, 6))
    plt.scatter(brand_data['Chassis'],brand_data[measure])
    plt.title(f'Year {y} brand {b} any values upper {max_v.iloc[0]:.2f} or lower {min_v.iloc[0]:.2f} is high variation')
    plt.axhline(y=mean_brand, color='red', linestyle='--', label='Mean')
    plt.axhline(y=max_v.iloc[0], color='black', linestyle='--', label='Upper Variance')
    plt.axhline(y=min_v.iloc[0], color='blue', linestyle='--', label='Lower Variance') 
    plt.xticks([], [])
    plt.legend()
    plt.show()
# measure_variance_bybrand('Profit','VOLVO','2024')


def Outliers_bybrand(measure,b,y): 
    measure_total=df[['brand','Profit','Sales','Total Cost','Total Item Cost','Y']]
    T_profit=pd.DataFrame(measure_total.reset_index())
    Q1=T_profit[measure].loc[(T_profit['brand']==b) & (T_profit['Y']==y)].quantile(0.25)
    Q3=T_profit[measure].loc[(T_profit['brand']==b) & (T_profit['Y']==y)].quantile(0.75)
    IQR=Q3-Q1
    print(f'25% {measure} values  less than {Q1}')
    print(f'75% {measure} values less than {Q3}')
    print(f'range middle of 50% {measure} values is {IQR}')
    max_range=Q3+1.5*IQR
    min_range=Q1-1.5*IQR
    print('-'*70)
    print(f'Chassis and Variant Model and The Values of  {measure} that outliers values :-')
    print('-'*70)
    print(df[['Sales Type','Chassis','Vehicle desc','Profit','Sales','Purchases','Total Cost','Total Item Cost','Vehicle discounts']].loc[(df['brand']==b) & (df['Y']==y) & (df[measure] > max_range) ].sort_values(measure,ascending=True) )
    print(df[['Sales Type','Chassis','Vehicle desc','Profit','Sales','Purchases','Total Cost','Total Item Cost','Vehicle discounts']].loc[(df['brand']==b) & (df['Y']==y) & (df[measure] < min_range) ].sort_values(measure,ascending=True) )
    
    print(f'Chassis and Variant Model and The Values of  {measure} that not outliers values :-')
    print('-'*70)
    print(df[['Sales Type','Chassis','Vehicle desc','Profit','Sales','Purchases','Total Cost','Total Item Cost','Vehicle discounts']].loc[(df['brand']==b) & (df['Y']==y) & (df[measure] < max_range) & (df[measure] > min_range) ].head(1) )
    v_f=df.loc[(df['brand']==b)& (df['Y']==y)].groupby(['brand'])[measure].std().sort_values(ascending=False) 
    print(f'The Standard Variation  for Measure {measure} for brand {b} ---> {v_f.iloc[0]}')
    data=df.loc[(df['brand']==b)& (df['Y']==y)]
    brand_data=pd.DataFrame(data.reset_index())
    mean_brand=brand_data[measure].median()
    plt.figure(figsize=(15, 6))
    plt.scatter(brand_data['Chassis'],brand_data[measure])
    plt.title(f'The brand  {b} Year {y} {measure} values that outside the range between {max_range:.2f} and {min_range:.2f} is outliers')
    plt.axhline(y=mean_brand, color='black', linestyle='--', label='Mean')
    plt.axhline(y=max_range, color='red', linestyle='--', label='max_range')
    plt.axhline(y=min_range, color='green', linestyle='--', label='min_range') 
    plt.xticks([], [])
    plt.legend()
    plt.show()
    
# Outliers_bybrand('Sales','MERCEDES','2023')  



def measure_variance_allmodels_bybrand(measure,b,y):
    print(f'Standard Variation Meaure for  {measure} values by Models')
    print(df.loc[(df['brand']==b)& (df['Y']==y)].groupby(['Model'])[measure].std().sort_values(ascending=False)) 
# measure_variance_allmodels_bybrand('Sales','VOLVO','2024')
def measure_variance_byspecific_model(measure,m,y): 
    #print(f'measure show what is the variance value between all values the {measure}  about the mean')  
    v_f=df.loc[(df['Model']==m)& (df['Y']==y)].groupby(['brand'])[measure].std().sort_values(ascending=False) 
    print(f'The Standard Variation  for Measure {measure} model {m} {v_f}' )
    data=df.loc[(df['Model']==m)& (df['Y']==y)]
    brand_data=pd.DataFrame(data.reset_index())
    mean_brand=brand_data[measure].mean()
    max_v=mean_brand+v_f
    min_v=mean_brand-v_f
    plt.figure(figsize=(10, 6))
    plt.scatter(brand_data['Chassis'],brand_data[measure])
    plt.title(f'Model {m} Year {y} any values upper {max_v.iloc[0]:.2f} or lower {min_v.iloc[0]:.2f} is high variation')
    plt.axhline(y=mean_brand, color='black', linestyle='--', label='Mean')
    plt.axhline(y=max_v.iloc[0], color='red', linestyle='--', label='Upper Variance')
    plt.axhline(y=min_v.iloc[0], color='blue', linestyle='--', label='Lower Variance')     
    plt.xticks([], [])    
    plt.show()
# measure_variance_byspecific_model('Sales','XC90','2024')
# measure_variance_byspecific_model('Sales','XC90','2023')
def measure_variance_allmodel_bybrand(measure,b,y): 
    #print(f'measure show what is the variance value between all values the {measure}  about the mean')  
    model_data=df['Model'].loc[df['brand']==b].unique()
    for m in model_data:
        v_f=df.loc[(df['Model']==m)& (df['Y']==y)].groupby(['Model'])[measure].std().sort_values(ascending=False) 
        
        print(f'The Standard Variation  for Measure {measure} model {m} {v_f}' )
        data=df.loc[(df['Model']==m)& (df['Y']==y)]
        brand_data=pd.DataFrame(data.reset_index())
        mean_brand=brand_data[measure].median()
        max_v=mean_brand+v_f
        min_v=mean_brand-v_f
        m_data=df[measure].loc[df['Model']==m]
        for mm_data, mv in zip(m_data, min_v):
            if mm_data < mv:
                print(mm_data)
                print(f'the {measure} high variation are {mm_data}') 

        plt.figure(figsize=(10, 6))
        plt.scatter(brand_data['Chassis'],brand_data[measure])
        plt.title(f'Model {m} Year {y} any values upper {max_v.iloc[0]:.2f} or lower {min_v.iloc[0]:.2f} is high variation')
        plt.axhline(y=mean_brand, color='black', linestyle='--', label='Mean')
        plt.axhline(y=max_v.iloc[0], color='red', linestyle='--', label='Upper Variance')
        plt.axhline(y=min_v.iloc[0], color='blue', linestyle='--', label='Lower Variance')     
        plt.xticks([], [])    
        plt.show()
# measure_variance_allmodel_bybrand('Purchases','VOLVO','2024')

def kur_skew_bymodel(measure,b,y):
    model_data=df['Model'].loc[df['brand']==b].unique()
    for m in model_data:
        sk_ku=df[measure].loc[(df['Model']==m)& (df['Y']==y)]
        sk_ku=pd.DataFrame(sk_ku.reset_index())
        
        print(f'The describe {measure} Model {m} ----> {sk_ku[measure].describe()}')
        print(f'Skewness: {sk_ku[measure].skew():.2f}')
        print(f'kurtosis: {sk_ku[measure].kurtosis():.2f}')    
        plt.hist(sk_ku[measure], bins=10, alpha=0.7, color='blue', edgecolor='black')
        
        # Add labels and title
        plt.xlabel(measure)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {measure} Data Model {m} Year {y}')
        
        # Display skewness value

        plt.show()
# kur_skew_bymodel('Profit','VOLVO','2024') 

def Outliers_by_specific_model(measure,m,y): 
    measure_total=df[['Model','Profit','Sales','Total Cost','Total Item Cost','Y']]
    T_profit=pd.DataFrame(measure_total.reset_index())
    Q1=T_profit[measure].loc[(T_profit['Model']==m) & (T_profit['Y']==y)].quantile(0.25)
    Q3=T_profit[measure].loc[(T_profit['Model']==m) & (T_profit['Y']==y)].quantile(0.75)
    IQR=Q3-Q1
    print(f'25% {measure} values  less than {Q1}')
    print(f'75% {measure} values less than {Q3}')
    print(f'range middle of 50% {measure} values is {IQR}')
    max_range=Q3+1.5*IQR
    min_range=Q1-1.5*IQR
    v_f=df.loc[(df['Model']==m)& (df['Y']==y)].groupby(['Model'])[measure].std().sort_values(ascending=False) 
    print('-'*70)
    print(f'Chassis and Variant Model and The Values of  {measure} that outliers values :-')
    print('-'*70)
    print(df[['Sales Type','Chassis','Vehicle desc','Profit','Sales','Purchases','Total Cost','Total Item Cost','Vehicle discounts']].loc[(df['Model']==m) & (df['Y']==y) & (df[measure] > max_range) ].sort_values(measure,ascending=True) )
    print(df[['Sales Type','Chassis','Vehicle desc','Profit','Sales','Purchases','Total Cost','Total Item Cost','Vehicle discounts']].loc[(df['Model']==m) & (df['Y']==y) & (df[measure] < min_range) ].sort_values(measure,ascending=True) )
    print('-'*70)
    print(f'Chassis and Variant Model and The Values of  {measure} that not outliers values :-')
    print('-'*70)
    print(df[['Sales Type','Chassis','Vehicle desc','Profit','Sales','Purchases','Total Cost','Total Item Cost','Vehicle discounts']].loc[(df['Model']==m) & (df['Y']==y) & (df[measure] < max_range) & (df[measure] > min_range) ].head(1) )
    print(f'The Standard Variation  for Measure {measure} for Model {m} ---> {v_f.iloc[0]}')
    data=df.loc[(df['Model']==m)& (df['Y']==y)]
    brand_data=pd.DataFrame(data.reset_index())
    mean_brand=brand_data[measure].median()
    plt.figure(figsize=(15, 6))
    plt.scatter(brand_data['Chassis'],brand_data[measure])
    plt.title(f'The Model  {m} Year {y} {measure} values that outside the range between {max_range:.2f} and {min_range:.2f} is outliers')
    plt.axhline(y=mean_brand, color='black', linestyle='--', label='Mean')
    plt.axhline(y=max_range, color='red', linestyle='--', label='Mean')
    plt.axhline(y=min_range, color='green', linestyle='--', label='Mean') 
    plt.xticks([], [])
    plt.legend()
    plt.show()
# Outliers_by_specific_model('Profit','XC60','2023')    
    
def Outliers_allmodel_bybrand(measure,b,y): 
    model_data=df['Model'].loc[df['brand']==b].unique()
    for m in model_data:    
        measure_total=df[['Model','Profit','Sales','Total Cost','Total Item Cost','Y']]
        T_profit=pd.DataFrame(measure_total.reset_index())
        Q1=T_profit[measure].loc[(T_profit['Model']==m) & (T_profit['Y']==y)].quantile(0.25)
        Q3=T_profit[measure].loc[(T_profit['Model']==m) & (T_profit['Y']==y)].quantile(0.75)
        IQR=Q3-Q1
        print(f'25% {measure} values  less than {Q1}')
        print(f'75% {measure} values less than {Q3}')
        print(f'range middle of 50% {measure} values is {IQR}')
        max_range=Q3+1.5*IQR
        min_range=Q1-1.5*IQR
        v_f=df.loc[(df['Model']==m)& (df['Y']==y)].groupby(['Model'])[measure].std().sort_values(ascending=False) 
        if v_f.empty:
            continue
        else:
            print('-'*70)
            print(f'Chassis and Variant Model and The Values of  {measure} that outliers values :-')
            print('-'*70)
            print(df[['Sales Type','Chassis','Vehicle desc','Profit','Sales','Purchases','Total Cost','Total Item Cost','Vehicle discounts']].loc[(df['Model']==m) & (df['Y']==y) & (df[measure] > max_range) ].sort_values(measure,ascending=True) )
            print(df[['Sales Type','Chassis','Vehicle desc','Profit','Sales','Purchases','Total Cost','Total Item Cost','Vehicle discounts']].loc[(df['Model']==m) & (df['Y']==y) & (df[measure] < min_range) ].sort_values(measure,ascending=True) )
            print('-'*70)
            print(f'Chassis and Variant Model and The Values of  {measure} that not outliers values :-')
            print('-'*70)
            print(df[['Sales Type','Chassis','Vehicle desc','Profit','Sales','Purchases','Total Cost','Total Item Cost','Vehicle discounts']].loc[(df['Model']==m) & (df['Y']==y) & (df[measure] < max_range) & (df[measure] > min_range) ].head(1) )
     
            print(f'The Standard Variation  for Measure {measure} for Model {m} ---> {v_f.iloc[0]}')
            data=df.loc[(df['Model']==m)& (df['Y']==y)]
            brand_data=pd.DataFrame(data.reset_index())
            mean_brand=brand_data[measure].median()
            plt.figure(figsize=(15, 6))
            plt.scatter(brand_data['Chassis'],brand_data[measure])
            plt.title(f'The Model  {m} {measure} values that outside the range between {max_range:.2f} and {min_range:.2f} is outliers')
            plt.axhline(y=mean_brand, color='black', linestyle='--', label='Mean')
            plt.axhline(y=max_range, color='red', linestyle='--', label='Mean')
            plt.axhline(y=min_range, color='green', linestyle='--', label='Mean') 
            plt.xticks([], [])
            plt.legend()
            plt.show()
            df.to_csv(r"C:\Users\mahmoud.ali\Downloads\output\.csv")
          
# Outliers_allmodel_bybrand('Profit','VOLVO','2023')
def measure_variance_bybrand_allvehicle_desc(measure,b,y): 
    print(f'Standard Variation Meaure for  {measure} values for brand {b} by Vehicle description')    
    print(df.loc[(df['brand']==b)&(df['Y']==y) ].groupby(['Vehicle desc'])[measure].std().sort_values(ascending=False))  
# measure_variance_bybrand_allvehicle_desc('Profit','VOLVO','2024')
  
def measure_variance_byvehicle_desc(measure,v,y): 
    v_f=df.loc[(df['Vehicle desc']==v)&(df['Y']==y)].groupby(['brand'])[measure].std().sort_values(ascending=False) 
    print(f'The Variance  for Measure {measure} Vehicle desc {v} {v_f}' )
    data=df.loc[(df['Vehicle desc']==v)&(df['Y']==y)]
    brand_data=pd.DataFrame(data.reset_index())
    mean_brand=brand_data[measure].mean()
    max_v=mean_brand+v_f
    min_v=mean_brand-v_f    
    plt.figure(figsize=(10, 6))
    plt.scatter(brand_data['Chassis'],brand_data[measure])
    plt.title(f'any values upper {max_v.iloc[0]:.2f} or lower {min_v.iloc[0]:.2f} is high variation')
    plt.axhline(y=mean_brand, color='black', linestyle='--', label='Mean')
    plt.axhline(y=max_v.iloc[0], color='red', linestyle='--', label='Upper Variance')
    plt.axhline(y=min_v.iloc[0], color='blue', linestyle='--', label='Lower Variance')    
    plt.xticks([], [])  
    plt.legend()
    plt.show()
# measure_variance_byvehicle_desc('Profit','Volvo XC40 Ulimate Dark','2024') 
#empty
def measure_variance_allvehicle_desc_bymodel(measure,m,y): 
    if __name__ == '__main__':
        variant=df['Vehicle desc'].loc[df['Model']==m].unique()
        for v in variant:
            v_f=df.loc[(df['Vehicle desc']==v)&(df['Y']==y)].groupby(['brand'])[measure].std().sort_values(ascending=False) 
            if v_f.empty:
                continue
            else:      
                print(f'The Variance  for Measure {measure} Vehicle desc {v} {v_f}' )
                data=df.loc[(df['Vehicle desc']==v)&(df['Y']==y)]
                brand_data=pd.DataFrame(data.reset_index())
                mean_brand=brand_data[measure].mean()
                max_v=mean_brand+v_f
                min_v=mean_brand-v_f    
                plt.figure(figsize=(15, 10))
                plt.scatter(brand_data['Chassis'],brand_data[measure])
                
                plt.title(f'{v} any values upper {max_v.iloc[0]:.2f} or lower {min_v.iloc[0]:.2f} is high variation')
                plt.axhline(y=mean_brand, color='black', linestyle='--', label='Mean')
                plt.axhline(y=max_v.iloc[0], color='red', linestyle='--', label='Upper Variance')
                plt.axhline(y=min_v.iloc[0], color='blue', linestyle='--', label='Lower Variance')   
                addlabels(brand_data['Chassis'],brand_data[measure])
                plt.xticks([], [])    
                plt.legend()
                plt.show()
            
# measure_variance_allvehicle_desc_bymodel('Sales','XC60','2023')   
# measure_variance_allvehicle_desc_bymodel('Purchases','XC60','2023')      
# measure_variance_allvehicle_desc_bymodel('Profit','XC40') 
def Outliers_byvehicledesc(measure,v,y): 
    measure_total=df[['Vehicle desc','Profit','Sales','Total Cost','Total Item Cost','Y']]
    T_profit=pd.DataFrame(measure_total.reset_index())
    Q1=T_profit[measure].loc[(T_profit['Vehicle desc']==v)&(T_profit['Y']==y)].quantile(0.25)
    Q3=T_profit[measure].loc[(T_profit['Vehicle desc']==v)&(T_profit['Y']==y)].quantile(0.75)
    IQR=Q3-Q1
    print(f'25% {measure} values  less than {Q1}')
    print(f'75% {measure} values less than {Q3}')
    print(f'range middle of 50% {measure} values is {IQR}')
    max_range=Q3+1.5*IQR
    min_range=Q1-1.5*IQR
    v_f=df.loc[(df['Vehicle desc']==v)&(df['Y']==y)].groupby(['Vehicle desc'])[measure].std().sort_values(ascending=False) 
    if v_f.empty:
        print('No Data')
    else:  
        print('-'*70)
        print(f'Chassis and Variant Model and The Values of  {measure} that outliers values :-')
        print('-'*70)
        print(df[['Sales Type','Chassis','Vehicle desc','Profit','Sales','Purchases','Total Cost','Total Item Cost','Vehicle discounts']].loc[(df['Vehicle desc']==v) & (df['Y']==y) & (df[measure] > max_range) ].sort_values(measure,ascending=True) )
        print(df[['Sales Type','Chassis','Vehicle desc','Profit','Sales','Purchases','Total Cost','Total Item Cost','Vehicle discounts']].loc[(df['Vehicle desc']==v) & (df['Y']==y) & (df[measure] < min_range) ].sort_values(measure,ascending=True) )
        print('-'*70)
        print(f'Chassis and Variant Model and The Values of  {measure} that not outliers values :-')
        print('-'*70)
        print(df[['Sales Type','Chassis','Vehicle desc','Profit','Sales','Purchases','Total Cost','Total Item Cost','Vehicle discounts']].loc[(df['Vehicle desc']==v) & (df['Y']==y) & (df[measure] < max_range) & (df[measure] > min_range) ].head(1) )
            
        print(f'The Standard Variation  for Measure {measure} for Vehicle desc {v} Year {y} ---> {v_f.iloc[0]}')
        data=df.loc[(df['Vehicle desc']==v)&(df['Y']==y)]
        brand_data=pd.DataFrame(data.reset_index())
        mean_brand=brand_data[measure].median()
        plt.figure(figsize=(10, 6))
        plt.scatter(brand_data['Chassis'],brand_data[measure])
        plt.title(f'The Vehicle desc  {v} Year {y} {measure} values that outside the range between {max_range:.2f} and {min_range:.2f} is outliers')
        plt.axhline(y=mean_brand, color='black', linestyle='--', label='Mean')
        plt.axhline(y=max_range, color='red', linestyle='--', label='max range')
        plt.axhline(y=min_range, color='green', linestyle='--', label='min range') 
        plt.xticks([], [])
        plt.legend()
        plt.show()
# Outliers_byvehicledesc('Sales','Volvo XC60 Ultimate Dark','2023')
# print(df['Model'].loc[df['brand']=='ALFA'].unique())

def Outliers_allvehicledesc_bymodel(measure,m,y): 
    veh_desc=df['Vehicle desc'].loc[df['Model']==m].unique()
    for v in veh_desc:
        measure_total=df[['Vehicle desc','Profit','Sales','Total Cost','Total Item Cost','Y']]
        T_profit=pd.DataFrame(measure_total.reset_index())
        Q1=T_profit[measure].loc[(T_profit['Vehicle desc']==v)&(T_profit['Y']==y)].quantile(0.25)
        Q3=T_profit[measure].loc[(T_profit['Vehicle desc']==v)&(T_profit['Y']==y)].quantile(0.75)
        IQR=Q3-Q1
        print(f'25% {measure} values  less than {Q1}')
        print(f'75% {measure} values less than {Q3}')
        print(f'range middle of 50% {measure} values is {IQR}')
        max_range=Q3+1.5*IQR
        min_range=Q1-1.5*IQR
        v_f=df.loc[(df['Vehicle desc']==v)&(df['Y']==y)].groupby(['Vehicle desc'])[measure].std().sort_values(ascending=False) 
        if v_f.empty:
            print('No Data')
        else:  
            print('-'*70)
            print(f'Chassis and Variant Model and The Values of  {measure} that outliers values :-')
            print('-'*70)
            print(df[['Sales Type','Chassis','Vehicle desc','Profit','Sales','Purchases','Total Cost','Total Item Cost','Vehicle discounts']].loc[(df['Vehicle desc']==v) & (df['Y']==y) & (df[measure] > max_range) ].sort_values(measure,ascending=True) )
            print(df[['Sales Type','Chassis','Vehicle desc','Profit','Sales','Purchases','Total Cost','Total Item Cost','Vehicle discounts']].loc[(df['Vehicle desc']==v) & (df['Y']==y) & (df[measure] < min_range) ].sort_values(measure,ascending=True) )
            print('-'*70)
            print(f'Chassis and Variant Model and The Values of  {measure} that not outliers values :-')
            print('-'*70)
            print(df[['Sales Type','Chassis','Vehicle desc','Profit','Sales','Purchases','Total Cost','Total Item Cost','Vehicle discounts']].loc[(df['Vehicle desc']==v) & (df['Y']==y) & (df[measure] < max_range) & (df[measure] > min_range) ].head(1) )
            print(f'{max_range} {min_range}')
            print(f'The Standard Variation  for Measure {measure} for Vehicle desc {v} Year {y} ---> {v_f.iloc[0]}')
            data=df.loc[(df['Vehicle desc']==v)&(df['Y']==y)]
            brand_data=pd.DataFrame(data.reset_index())
            mean_brand=brand_data[measure].median()
            plt.figure(figsize=(10, 6))
            plt.scatter(brand_data['Chassis'],brand_data[measure])
            plt.title(f'The Vehicle desc  {v} Year {y} {measure} values that outside the range between {max_range:.2f} and {min_range:.2f} is outliers')
            plt.axhline(y=mean_brand, color='black', linestyle='--', label='Mean')
            plt.axhline(y=max_range, color='red', linestyle='--', label='max range')
            plt.axhline(y=min_range, color='green', linestyle='--', label='min range') 
            plt.xticks([], [])
            plt.legend()
            plt.show()
# Outliers_allvehicledesc_bymodel('Sales','XC60','2023')

def IQR(b,mea,y):
    
    measure_total=df[['brand','Profit','Sales','Total Cost','Total Item Cost','Y']]
    T_profit=pd.DataFrame(measure_total.reset_index())
    Q1=T_profit[mea].loc[(T_profit['brand']==b)&(T_profit['Y']==y)].quantile(0.25)
    Q3=T_profit[mea].loc[(T_profit['brand']==b)&(T_profit['Y']==y)].quantile(0.75)
    IQR=Q3-Q1
    print(f'25% {mea} values  less than {Q1}')
    print(f'75% {mea} values less than {Q3}')
    print(f'range middle of 50% {mea} values is {IQR}')
    max_range=Q3+1.5*IQR
    min_range=Q1-1.5*IQR
    print(f'Year {y} The {mea} values that outside the range between {max_range} and {min_range} is outliers')
# IQR('VOLVO','Sales','2023')

def kur_skew_bybrand(measure,b,y):
    sk_ku=df[measure].loc[(df['brand']==b) & (df['Y']==y)]
    sk_ku=pd.DataFrame(sk_ku.reset_index())
    
    print(f'Year {y} The describe {measure} ----> {sk_ku[measure].describe()}')
    print(f'Skewness: {sk_ku[measure].skew():.2f}')
    print(f'kurtosis: {sk_ku[measure].kurtosis():.2f}')
    plt.hist(sk_ku[measure], bins=10, alpha=0.7, color='blue', edgecolor='black')
    
    # Add labels and title
    plt.xlabel(measure)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {measure} Data brand {b} Year {y}')
    
    # Display skewness value
    plt.legend()
    plt.show()
# kur_skew_bybrand('Profit','MERCEDES','2024')    

def kur_skew_allvehdesc_bymodel(measure,m,y):
    veh_desc=df['Vehicle desc'].loc[df['Model']==m].unique()
    for v in veh_desc:
        sk_ku=df[measure].loc[(df['Vehicle desc']==v) & (df['Y']==y)]
        sk_ku=pd.DataFrame(sk_ku.reset_index())
        
        print(f'Year {y} Vehicle Variant {v} The describe {measure} ----> {sk_ku[measure].describe()}')
        print(f'Skewness Vehicle Variant {v} : {sk_ku[measure].skew():.2f}')
        print(f'kurtosis Vehicle Variant {v} : {sk_ku[measure].kurtosis():.2f}')
        plt.hist(sk_ku[measure], bins=10, alpha=0.7, color='blue', edgecolor='black')
        # Add labels and title
        plt.xlabel(measure)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {measure} Data Vehicle Variant {v} Year {y}')
        # Display skewness value
        plt.legend()
        plt.show()
# kur_skew_allvehdesc_bymodel('Profit','XC60','2023')   

#plt.boxplot(df['Profit'].loc[df['brand']=='DS'])

    
                 #--Correlation Analysis--
   

def correlation_coefficient_bybrand_byyear(m1,m2,b,y):
    corr=df[[m1,m2]].loc[(df['Manufacturer']==b) & (df['Y']==y)].corr(method='spearman')
    print('correlation by spearman')
    print ('*'*55)
    print(f' The Correlation Coefficient for {corr.iloc[:1,1:2].stack()}')
    
    plt.figure(figsize=(15,5)) 
    sns.regplot(x=df[m1].loc[(df['Manufacturer']==b) & (df['Y']==y)],y=df[m2].loc[(df['Manufacturer']==b) & (df['Y']==y)],data=df['Chassis'].loc[(df['Manufacturer']==b) & (df['Y']==y)])          
    plt.title(f'Correlation Coefficient between {m1} and {m2} for {b} Year {y}') 
    plt.show()
    stat, p = spearmanr(df[m1].loc[(df['Manufacturer']==b) & (df['Y']==y)], df[m2].loc[(df['Manufacturer']==b) & (df['Y']==y)])
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
      print(f'{p:.2f} --> Dont reject Ho at 5% (sig level) there is not a statistic significant correlation between {m1} and {m2}  ')
    else:
      print(f'{p:.2f} -->  reject Ho at 5% (sig level) there is  a statistic significant correlation between {m1} and {m2}  ')
# correlation_coefficient_bybrand_byyear('Sales','Profit','MERCEDES','2023')

def correlation_coefficient_bybrand_allyears(m1,m2,b):
    corr=df[[m1,m2]].loc[df['Manufacturer']==b].corr(method='spearman')
    print('correlation by spearman')
    print ('*'*55)
    print(f' The Correlation Coefficient for {corr.iloc[:1,1:2].stack()}')
    
    plt.figure(figsize=(15,5)) 
    sns.regplot(x=df[m1].loc[df['Manufacturer']==b],y=df[m2].loc[df['Manufacturer']==b],data=df['Chassis'].loc[df['Manufacturer']==b])          
    plt.title(f'Correlation Coefficient between {m1} and {m2} for {b}') 
    plt.show()
    stat, p = spearmanr(df[m1].loc[df['Manufacturer']==b], df[m2].loc[df['Manufacturer']==b])
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
      print(f'{p:.2f} --> Dont reject Ho at 5% (sig level) there is not a statistic significant correlation between {m1} and {m2}  ')
    else:
      print(f'{p:.2f} -->  reject Ho at 5% (sig level) there is  a statistic significant correlation between {m1} and {m2}  ')
# correlation_coefficient_bybrand_allyears('Sales','Profit','VOLVO')

def correlation_coefficient_bybrand_by_sales_type_allyears(m1,m2,b):
    data_sales_type=df['Sales Type'].loc[df['Manufacturer']==b].unique()
    for  sal_type in data_sales_type :
        corr=df[[m1,m2]].loc[(df['Manufacturer']==b) & (df['Sales Type']==sal_type )].corr(method='spearman')
        print('correlation by spearman')
        print ('*'*55)
        print(f' The Correlation Coefficient for {corr.iloc[:1,1:2].stack()}')
        
        plt.figure(figsize=(15,5)) 
        sns.regplot(x=df[m1].loc[(df['Manufacturer']==b) & (df['Sales Type']==sal_type )],y=df[m2].loc[(df['Manufacturer']==b) & (df['Sales Type']==sal_type )],data=df['Chassis'].loc[(df['Manufacturer']==b) & (df['Sales Type']==sal_type )])          
        plt.title(f'Correlation Coefficient between {m1} and {m2} for  {sal_type} {b}') 
        plt.show()
        stat, p = spearmanr(df[m1].loc[(df['Manufacturer']==b) & (df['Sales Type']==sal_type )], df[m2].loc[(df['Manufacturer']==b) & (df['Sales Type']==sal_type )])
        print('stat=%.3f, p=%.3f' % (stat, p))
        if p > 0.05:
          print(f'{p:.2f} --> Dont reject Ho at 5% (sig level) there is not a statistic significant correlation between {m1} and {m2}  ')
        else:
          print(f'{p:.2f} -->  reject Ho at 5% (sig level) there is  a statistic significant correlation between {m1} and {m2}  ')

def correlation_coefficient_bybrand_by_sales_type_byyear(m1,m2,b,y):
    data_sales_type=df['Sales Type'].loc[(df['Manufacturer']==b) & (df['Y']==y)].unique()
    for  sal_type in data_sales_type :
        corr=df[[m1,m2]].loc[(df['Manufacturer']==b) & (df['Sales Type']==sal_type )  & (df['Y']==y)].corr(method='spearman')
        print('correlation by spearman')
        print ('*'*55)
        print(f' The Correlation Coefficient for {corr.iloc[:1,1:2].stack()}')
        
        plt.figure(figsize=(15,5)) 
        sns.regplot(x=df[m1].loc[(df['Manufacturer']==b) & (df['Sales Type']==sal_type ) & (df['Y']==y)],y=df[m2].loc[(df['Manufacturer']==b) & (df['Sales Type']==sal_type ) & (df['Y']==y)],data=df['Chassis'].loc[(df['Manufacturer']==b) & (df['Sales Type']==sal_type ) & (df['Y']==y)])          
        plt.title(f'Correlation Coefficient between {m1} and {m2} for  {sal_type} {b}') 
        plt.show()
        stat, p = spearmanr(df[m1].loc[(df['Manufacturer']==b) & (df['Sales Type']==sal_type ) & (df['Y']==y)], df[m2].loc[(df['Manufacturer']==b) & (df['Sales Type']==sal_type ) & (df['Y']==y)])
        print('stat=%.3f, p=%.3f' % (stat, p))
        if p > 0.05:
          print(f'{p:.2f} --> Dont reject Ho at 5% (sig level) there is not a statistic significant correlation between {m1} and {m2}  ')
        else:
          print(f'{p:.2f} -->  reject Ho at 5% (sig level) there is  a statistic significant correlation between {m1} and {m2}  ')
          
# correlation_coefficient_bybrand_by_sales_type_byyear('Sales','Profit','VOLVO','2024')

# to know Types of Models 
#print(df.loc[df['brand']=='VOLVO','Model'].unique())
def correlation_coefficient_bymodel_allyears(m1,m2,model):
    corr=df[[m1,m2]].loc[df['Model']==model].corr(method='pearson')
    print(f' The Correlation Coefficient Model {model} for {corr.iloc[:1,1:2].stack()}')
    print ('*'*55)    
    plt.figure(figsize=(15,5)) 
    sns.regplot(x=df[m1].loc[df['Model']==model],y=df[m2].loc[df['Model']==model],data=df['Chassis'].loc[df['Model']==model])          
    plt.title(f'Correlation Coefficient between {m1} and {m2} for Model {model}') 
    plt.show()
    stat, p = spearmanr(df[m1].loc[df['Model']==model], df[m2].loc[df['Model']==model])
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
      print(f'{p:.2f} --> Dont reject Ho at 5% (sig level) there is not a statistic significant correlation between {m1} and {m2}  ')
    else:
        print(f'{p:.2f} -->  reject Ho at 5% (sig level) there is  a statistic significant correlation between {m1} and {m2}  ')
# correlation_coefficient_bymodel_allyears('Sales','Profit','DS7')


def correlation_coefficient_bymodel_byyear(m1,m2,model,y):
    corr=df[[m1,m2]].loc[(df['Model']==model) & (df['Y']==y)].corr(method='spearman')
    print(f' The Correlation Coefficient Model {model} Year {y} for {corr.iloc[:1,1:2].stack()}')
    print ('*'*55)    
    plt.figure(figsize=(15,5)) 
    sns.regplot(x=df[m1].loc[(df['Model']==model) & (df['Y']==y)],y=df[m2].loc[(df['Model']==model) & (df['Y']==y)],data=df['Chassis'].loc[df['Model']==model])          
    plt.title(f'Correlation Coefficient between {m1} and {m2} for Model {model}') 
    plt.show()
    stat, p = spearmanr(df[m1].loc[(df['Model']==model) & (df['Y']==y)], df[m2].loc[(df['Model']==model) & (df['Y']==y)])
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
      print(f'{p:.2f} --> Dont reject Ho at 5% (sig level) there is not a statistic significant correlation between {m1} and {m2}  ')
    else:
        print(f'{p:.2f} -->  reject Ho at 5% (sig level) there is  a statistic significant correlation between {m1} and {m2}  ')
# correlation_coefficient_bymodel_byyear('Sales','Purchases','DS7','2024')
        

def correlation_coefficient_allmodels_bybrand_allyears(m1,m2,b):
    model_data=df['Model'].loc[df['Manufacturer']==b].unique()
    for models in model_data:
        
        corr=df[[m1,m2]].loc[df['Model']==models].corr(method='spearman')
        print(f' The Correlation Coefficient Model {models} for {corr.iloc[:1,1:2].stack()}')
        print ('*'*55)    
        plt.figure(figsize=(15,5)) 
        sns.regplot(x=df[m1].loc[df['Model']==models],y=df[m2].loc[df['Model']==models],data=df['Chassis'].loc[df['Model']==models])          
        plt.title(f'Correlation Coefficient between {m1} and {m2} for Model {models}') 
        plt.show()
        stat, p = spearmanr(df[m1].loc[df['Model']==models], df[m2].loc[df['Model']==models])
        print('stat=%.3f, p=%.3f' % (stat, p))
        if p > 0.05:
          print(f'{p:.2f} --> Dont reject Ho at 5% (sig level) there is not a statistic significant correlation between {m1} and {m2}  ')
        else:
            print(f'{p:.2f} -->  reject Ho at 5% (sig level) there is  a statistic significant correlation between {m1} and {m2}  ')
        print ('-'*100)
# correlation_coefficient_allmodels_bybrand_allyears('Sales','Profit','MERCEDES')

def correlation_coefficient_allmodels_bybrand_byyear(m1,m2,b,y):
    model_data=df['Model'].loc[(df['Manufacturer']==b) & (df['Y']==y)].unique()
    for models in model_data:
        
        corr=df[[m1,m2]].loc[(df['Model']==models) & (df['Y']==y)].corr(method='spearman')
        print(f' The Correlation Coefficient Model {models} Year {y} for {corr.iloc[:1,1:2].stack()}')
        print ('*'*55)    
        plt.figure(figsize=(15,5)) 
        sns.regplot(x=df[m1].loc[(df['Model']==models) & (df['Y']==y)],y=df[m2].loc[(df['Model']==models) & (df['Y']==y)],data=df['Chassis'].loc[(df['Model']==models) & (df['Y']==y)])          
        plt.title(f'Correlation Coefficient between {m1} and {m2} for Model {models}') 
        plt.show()
        stat, p = spearmanr(df[m1].loc[(df['Model']==models) & (df['Y']==y)], df[m2].loc[(df['Model']==models) & (df['Y']==y)])
        print('stat=%.3f, p=%.3f' % (stat, p))
        if p > 0.05:
          print(f'{p:.2f} --> Dont reject Ho at 5% (sig level) there is not a statistic significant correlation between {m1} and {m2}  ')
        else:
            print(f'{p:.2f} -->  reject Ho at 5% (sig level) there is  a statistic significant correlation between {m1} and {m2}  ')
        print ('-'*100)
# correlation_coefficient_allmodels_bybrand_byyear('Sales','Profit','DS','2024')

# to know Types of Vehicle desc     
# print(df.loc[df['brand']=='DS','Vehicle desc'].unique())
def correlation_coefficient_byv_desc_allyears(m1,m2,v_desc):
    corr=df[[m1,m2]].loc[df['Vehicle desc']==v_desc].corr(method='spearman')
    print(f' The Correlation Coefficient vehicle desc {v_desc} for {corr.iloc[:1,1:2].stack()}')
    print ('*'*55)    
    plt.figure(figsize=(15,5)) 
    sns.regplot(x=df[m1].loc[df['Vehicle desc']==v_desc],y=df[m2].loc[df['Vehicle desc']==v_desc],data=df['Chassis'].loc[df['Vehicle desc']==v_desc])          
    plt.title(f'Correlation Coefficient between {m1} and {m2} for Vehicle desc {v_desc}') 
    plt.show()
    stat, p = spearmanr(df[m1].loc[df['Vehicle desc']==v_desc], df[m2].loc[df['Vehicle desc']==v_desc])
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
      print(f'{p:.2f} --> Dont reject Ho at 5% (sig level) there is not a statistic significant correlation between {m1} and {m2}  ')
    else:
        print(f'{p:.2f} -->  reject Ho at 5% (sig level) there is  a statistic significant correlation between {m1} and {m2}  ')
    print ('-'*100)
# correlation_coefficient_byv_desc('Sales','Profit','DS7 Rivoli')

def correlation_coefficient_byv_desc_byyear(m1,m2,v_desc,y):
    corr=df[[m1,m2]].loc[(df['Vehicle desc']==v_desc) &(df['Y']==y)].corr(method='spearman')
    print(f' The Correlation Coefficient vehicle desc {v_desc} Year {y} for {corr.iloc[:1,1:2].stack()}')
    print ('*'*55)    
    plt.figure(figsize=(15,5)) 
    sns.regplot(x=df[m1].loc[(df['Vehicle desc']==v_desc) &(df['Y']==y)],y=df[m2].loc[(df['Vehicle desc']==v_desc) &(df['Y']==y)],data=df['Chassis'].loc[(df['Vehicle desc']==v_desc) &(df['Y']==y)])          
    plt.title(f'Correlation Coefficient between {m1} and {m2} for Vehicle desc {v_desc}') 
    plt.show()
    stat, p = spearmanr(df[m1].loc[(df['Vehicle desc']==v_desc) &(df['Y']==y)], df[m2].loc[(df['Vehicle desc']==v_desc) &(df['Y']==y)])
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
      print(f'{p:.2f} --> Dont reject Ho at 5% (sig level) there is not a statistic significant correlation between {m1} and {m2}  ')
    else:
        print(f'{p:.2f} -->  reject Ho at 5% (sig level) there is  a statistic significant correlation between {m1} and {m2}  ')
    print ('-'*100)
# correlation_coefficient_byv_desc_byyear('Sales','Profit','DS7 Rivoli','2024')
# correlation_coefficient_byv_desc_byyear('Sales','Profit','DS7 Rivoli','2023')

def correlation_coefficient_all_varaint_by_model_allyears(m1,m2,model):
    veh_des=df['Vehicle desc'].loc[df['Model']==model].unique()
    for v_desc in veh_des:
        
        corr=df[[m1,m2]].loc[df['Vehicle desc']==v_desc].corr(method='spearman')
        print(f' The Correlation Coefficient vehicle desc {v_desc} for {corr.iloc[:1,1:2].stack()}')
        print ('*'*55)    
        plt.figure(figsize=(15,5)) 
        sns.regplot(x=df[m1].loc[df['Vehicle desc']==v_desc],y=df[m2].loc[df['Vehicle desc']==v_desc],data=df['Chassis'].loc[df['Vehicle desc']==v_desc])          
        plt.title(f'Correlation Coefficient between {m1} and {m2} for Vehicle desc {v_desc}') 
        plt.show()
        stat, p = spearmanr(df[m1].loc[df['Vehicle desc']==v_desc], df[m2].loc[df['Vehicle desc']==v_desc])
        print('stat=%.3f, p=%.3f' % (stat, p))
        if p > 0.05:
          print(f'{p:.2f} --> Dont reject Ho at 5% (sig level) there is not a statistic significant correlation between {m1} and {m2}  ')
        else:
            print(f'{p:.2f} -->  reject Ho at 5% (sig level) there is  a statistic significant correlation between {m1} and {m2}  ')
        print ('-'*100)
# correlation_coefficient_all_varaint_by_model_allyears('Sales','Profit','DS4')

def correlation_coefficient_all_varaint_by_model_byyear(m1,m2,model,y):
    veh_des=df['Vehicle desc'].loc[(df['Model']==model) & (df['Y']==y)].unique()
    for v_desc in veh_des:
        
        corr=df[[m1,m2]].loc[(df['Vehicle desc']==v_desc)& (df['Y']==y)].corr(method='spearman')
        print(f' The Correlation Coefficient vehicle desc {v_desc} Year {y} for {corr.iloc[:1,1:2].stack()}')
        print ('*'*55)    
        plt.figure(figsize=(15,5)) 
        sns.regplot(x=df[m1].loc[(df['Vehicle desc']==v_desc)& (df['Y']==y)],y=df[m2].loc[(df['Vehicle desc']==v_desc)& (df['Y']==y)],data=df['Chassis'].loc[(df['Vehicle desc']==v_desc)& (df['Y']==y)])          
        plt.title(f'Correlation Coefficient between {m1} and {m2} for Vehicle desc {v_desc} Year {y}') 
        plt.show()
        stat, p = spearmanr(df[m1].loc[(df['Vehicle desc']==v_desc)& (df['Y']==y)], df[m2].loc[(df['Vehicle desc']==v_desc)& (df['Y']==y)])
        print('stat=%.3f, p=%.3f' % (stat, p))
        if p > 0.05:
          print(f'{p:.2f} --> Dont reject Ho at 5% (sig level) there is not a statistic significant correlation between {m1} and {m2}  ')
        else:
            print(f'{p:.2f} -->  reject Ho at 5% (sig level) there is  a statistic significant correlation between {m1} and {m2}  ')
        print ('-'*100)
# correlation_coefficient_all_varaint_by_model_byyear('Sales','Profit','DS4','2024')
# print ('*'*88)

def correlation_coe_purchases_items_costbybrand_allyears(b):
        data_cost=df[['Export Charges','GAFI Duty  0#1%','GAFI DUTY  2%','land Insurance',
        'Local Import Charges','Local Marine Insurance','Storage','Transportation'
        ,'PDI Elezz For Service','Sea Freigh','Other Expenses','GPS','Purchases','Manufacturer']] 
        corr_cost=data_cost.loc[data_cost.Manufacturer==b].corr(method='spearman')
        df_corr_cost=corr_cost.unstack()
        sort_df_corr_cost=df_corr_cost.sort_values(ascending=False)
        high_corr=sort_df_corr_cost[(sort_df_corr_cost) > -0.9 ]
        df_high_corr=pd.DataFrame(high_corr.reset_index())
        print(b)
        print(df_high_corr)
        cost_cor=df_high_corr.loc[df_high_corr.level_0=='Purchases']
        print(cost_cor)
        plt.figure(figsize=(15,15))
        sns.heatmap(corr_cost,cmap='coolwarm',linewidths=5,annot=True)
        plt.show()
        
def correlation_coe_purchases_items_costbybrand_byyear(b,y):
        data_cost=df[['Export Charges','GAFI Duty  0#1%','GAFI DUTY  2%','land Insurance',
        'Local Import Charges','Local Marine Insurance','Storage','Transportation'
        ,'PDI Elezz For Service','Sea Freigh','Other Expenses','GPS','Purchases','Manufacturer','Y']] 
        corr_cost=data_cost.loc[(data_cost.Manufacturer==b) & (data_cost.Y==y)].corr(method='spearman')
        df_corr_cost=corr_cost.unstack()
        sort_df_corr_cost=df_corr_cost.sort_values(ascending=False)
        high_corr=sort_df_corr_cost[(sort_df_corr_cost) > -0.9 ]
        df_high_corr=pd.DataFrame(high_corr.reset_index())
        print(b)
        print(df_high_corr)
        cost_cor=df_high_corr.loc[df_high_corr.level_0=='Purchases']
        print(cost_cor)
        plt.figure(figsize=(15,15))
        sns.heatmap(corr_cost,cmap='coolwarm',linewidths=5,annot=True)
        plt.show()       
# correlation_coe_purchases_items_costbybrand_byyear('VOLVO','2024')
def correlation_coe_purchases_items_costbymodel_allyears(m):
        data_cost=df[['Export Charges','GAFI Duty  0#1%','GAFI DUTY  2%','land Insurance',
        'Local Import Charges','Local Marine Insurance','Storage','Transportation'
        ,'PDI Elezz For Service','Sea Freigh','Other Expenses','GPS','Purchases','Model']] 
        corr_cost=data_cost.loc[data_cost.Model==m].corr(method='spearman')
        df_corr_cost=corr_cost.unstack()
        sort_df_corr_cost=df_corr_cost.sort_values(ascending=False)
        high_corr=sort_df_corr_cost[(sort_df_corr_cost) > -0.9 ]
        df_high_corr=pd.DataFrame(high_corr.reset_index())
        print(f'the model {m}')
        print(df_high_corr)
        cost_cor=df_high_corr.loc[df_high_corr.level_0=='Purchases']
        print(cost_cor)
        plt.figure(figsize=(15,15))
        sns.heatmap(corr_cost,cmap='coolwarm',linewidths=5,annot=True)
        plt.show()      
def correlation_coe_purchases_items_costbymodel_byyear(m,y):
        data_cost=df[['Export Charges','GAFI Duty  0#1%','GAFI DUTY  2%','land Insurance',
        'Local Import Charges','Local Marine Insurance','Storage','Transportation'
        ,'PDI Elezz For Service','Sea Freigh','Other Expenses','GPS','Purchases','Model','Y']] 
        corr_cost=data_cost.loc[(data_cost.Model==m) & (data_cost.Y==y)].corr(method='spearman')
        df_corr_cost=corr_cost.unstack()
        sort_df_corr_cost=df_corr_cost.sort_values(ascending=False)
        high_corr=sort_df_corr_cost[(sort_df_corr_cost) > -0.9 ]
        df_high_corr=pd.DataFrame(high_corr.reset_index())
        print(f'the model {m}')
        print(df_high_corr)
        cost_cor=df_high_corr.loc[df_high_corr.level_0=='Purchases']
        print(cost_cor)
        plt.figure(figsize=(15,15))
        sns.heatmap(corr_cost,cmap='coolwarm',linewidths=5,annot=True)
        plt.show()        
# correlation_coe_purchases_items_costbymodel_byyear('XC60','2024')        

def correlation_coe_purchases_items_costbyvedesc_allyears(v):
        
        data_cost=df[['Export Charges','GAFI Duty  0#1%','GAFI DUTY  2%','land Insurance',
        'Local Import Charges','Local Marine Insurance','Storage','Transportation'
        ,'PDI Elezz For Service','Sea Freigh','Other Expenses','GPS','Purchases','Vehicle desc']] 
        corr_cost=data_cost.loc[data_cost['Vehicle desc']==v].corr(method='spearman')
        df_corr_cost=corr_cost.unstack()
        sort_df_corr_cost=df_corr_cost.sort_values(ascending=False)
        high_corr=sort_df_corr_cost[(sort_df_corr_cost) > -0.9 ]
        df_high_corr=pd.DataFrame(high_corr.reset_index())
        print(f'the Vehicle_desc {v}')
        print(df_high_corr)
        cost_cor=df_high_corr.loc[df_high_corr.level_0=='Purchases']
        print(cost_cor)
        plt.figure(figsize=(15,15))
        sns.heatmap(corr_cost,cmap='coolwarm',linewidths=5,annot=True)
        plt.show()      
        
def correlation_coe_purchases_items_costbyvedesc_byyear(v,y):
        
        data_cost=df[['Export Charges','GAFI Duty  0#1%','GAFI DUTY  2%','land Insurance',
        'Local Import Charges','Local Marine Insurance','Storage','Transportation'
        ,'PDI Elezz For Service','Sea Freigh','Other Expenses','GPS','Purchases','Vehicle desc','Y']] 
        corr_cost=data_cost.loc[(data_cost['Vehicle desc']==v) & (data_cost['Y']==y)].corr(method='spearman')
        df_corr_cost=corr_cost.unstack()
        sort_df_corr_cost=df_corr_cost.sort_values(ascending=False)
        high_corr=sort_df_corr_cost[(sort_df_corr_cost) > -0.9 ]
        df_high_corr=pd.DataFrame(high_corr.reset_index())
        print(f'the Vehicle_desc {v}')
        print(df_high_corr)
        cost_cor=df_high_corr.loc[df_high_corr.level_0=='Purchases']
        print(cost_cor)
        plt.figure(figsize=(15,15))
        sns.heatmap(corr_cost,cmap='coolwarm',linewidths=5,annot=True)
        plt.show()            
# correlation_coe_purchases_items_costbyvedesc_byyear('DS7 Rivoli','2024')

vehicle_spec_non_elec=df[['Model','spec_Tank_Capcity','spec_AVG_fuel_Consum','spec_Engine_cc','spec_Horsepower','spec_Number_of_doors','spec_Body_Style','spec_Fuel_type','Sales','brand','Vehicle desc']].loc[df['spec_vehicle_type']=='Non_Elec']


vehicle_spec_elec=df[['Model','spec_Tank_Capcity','spec_AVG_fuel_Consum','spec_Engine_cc','spec_Horsepower','spec_Number_of_doors','spec_Body_Style','spec_Fuel_type','Sales','brand','Vehicle desc']].loc[df['spec_vehicle_type']=='ELEC']
  

# vehicle_spec_non_elec.hist(bins=50, figsize=(15,10))
# plt.show()

def more_impact_on_car_sales():

     for p in vehicle_spec_non_elec.columns:
         plt.figure(figsize=(12, 5))
         sns.histplot(vehicle_spec_non_elec[p], bins=10,kde=True)
         plt.xticks(rotation='vertical',size=8)
         plt.title(p)
         plt.show()
# more_impact_on_car_sales()

def correlation_coe_heatmap_veh_spec_sales_bybrand():
    desc_brand=df['brand'].unique()
    for v_desc in desc_brand:
        corr_spec_sales=vehicle_spec_non_elec.loc[vehicle_spec_non_elec['brand']==v_desc].corr(method='spearman')
        df_corr_cost=corr_spec_sales.unstack()
        sort_df_corr_cost=df_corr_cost.sort_values(ascending=False)
        high_corr=sort_df_corr_cost[(sort_df_corr_cost) > -0.9 ]
        df_high_corr=pd.DataFrame(high_corr.reset_index())
        spec_corr=df_high_corr.loc[df_high_corr.level_0=='Sales']
        print(f'{v_desc} ')
        print(f'{spec_corr}')
        plt.figure(figsize=(15,10))
        sns.heatmap(corr_spec_sales,cmap='coolwarm',linewidths=5,annot=True)
        plt.show()
        for m in vehicle_spec_non_elec:
            stat, p = spearmanr(vehicle_spec_non_elec[m].loc[vehicle_spec_non_elec['brand']==v_desc], vehicle_spec_non_elec['Sales'].loc[vehicle_spec_non_elec['brand']==v_desc])
            print('stat=%.3f, p=%.3f' % (stat, p))
            if p > 0.05:
              print(f'{p:.2f} --> Dont reject Ho at 5% (sig level) there is not a statistic significant correlation between {m} and sales  ')
            else:
                print(f'{p:.2f} -->  reject Ho at 5% (sig level) there is  a statistic significant correlation between {m} and sales  ')
            print ('-'*100)        
  
# correlation_coe_heatmap_veh_spec_sales_bybrand()



def correlation_coe_heatmap_veh_spec_sales_allbrands():
    corr_spec_sales=vehicle_spec_non_elec.corr(method='spearman')
    df_corr_cost=corr_spec_sales.unstack()
    sort_df_corr_cost=df_corr_cost.sort_values(ascending=False)
    high_corr=sort_df_corr_cost[(sort_df_corr_cost) > -0.9 ]
    df_high_corr=pd.DataFrame(high_corr.reset_index())
    spec_corr=df_high_corr.loc[df_high_corr.level_0=='Sales']
    print(f'{spec_corr}') 
    plt.figure(figsize=(15,5))
    sns.heatmap(corr_spec_sales,cmap='coolwarm',linewidths=5,annot=True)
    plt.show()
    for m in vehicle_spec_non_elec:
        stat, p = spearmanr(vehicle_spec_non_elec[m], vehicle_spec_non_elec['Sales'])
        print('stat=%.3f, p=%.3f' % (stat, p)  )
        if p > 0.05:
              print(f'{p:.2f} --> Dont reject Ho at 5% (sig level) there is not a statistic significant correlation between {m} and sales  ')
        else:
            print(f'{p:.2f} -->  reject Ho at 5% (sig level) there is  a statistic significant correlation between {m} and sales  ')
        print ('____'*20)  
# correlation_coe_heatmap_veh_spec_sales_allbrands()


veh_spec=['spec_Tank_Capcity','spec_AVG_fuel_Consum','spec_Engine_cc','spec_Horsepower','spec_Number_of_doors','spec_Body_Style','spec_Fuel_type']

def graphs_veh_spec_and_sales_allbrand():
    for v in veh_spec:
        vehicle_spec_non_elec.plot(kind="scatter", x=v, y="Sales")


# graphs_veh_spec_and_sales_allbrand()


