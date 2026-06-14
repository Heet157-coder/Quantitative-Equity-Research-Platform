import pandas as pd
import os

data_folder = "data/raw"

all_data = [] 

for file in os.listdir(data_folder):  #os.listdir() give all the files in the given folder

    if file.endswith(".csv"):         #so that only csv files are operated

        filepath = os.path.join(data_folder, file)   #to get the filepath better than writing manuallly as there may be error

        df = pd.read_csv(filepath)

        
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])  #If there is index in any of the Unamed: 0 is added so this step removes that

        stock_name = file.replace(".csv", "")   #removes .csv from the data file name so that we get the stock name only

        df["Stock"] = stock_name   #df means dataframe, creates a new coloumn named stock and adds the data of stock name in it

        all_data.append(df)  #to add the dataframe to the all_data list

master_df = pd.concat(all_data, ignore_index=True)  #stacks all datas vertically and ignore index as without it 0101 and with it 0123

os.makedirs("data/processed", exist_ok=True) #if false instead of true and the same named file exist then python will crash

master_df.to_csv(
    "data/processed/master_stock_data.csv",
    index=False  #Else we will get extra coloumn of 0123
)

print(master_df.head())
print()
print(master_df.columns)
print()
print(master_df.shape)