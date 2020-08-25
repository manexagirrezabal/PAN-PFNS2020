
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def cast(val):
    try:
        try:
            return float(val.replace(",","."))
        except AttributeError:
            return 0.0
    except ValueError:
        return 0.0

def filteredmean(df):
    esm = df['ES'][df['ES']!=0.0].mean()
    enm = df['EN'][df['EN']!=0.0].mean()
    avm = df['AVG'][df['AVG']!=0.0].mean()
    return [esm,enm,avm]
    
data = pd.read_csv("Book2.txt",sep="\t")

data['EN']=data['EN'].map(cast)
data['ES']=data['ES'].map(cast)
data['AVG']=data['AVG'].map(cast)


fix,ax = plt.subplots(figsize=(7,5))
data.boxplot(ax=ax, showfliers=False)
ax.set_ylim(0.0,1.0)
ax.scatter([1,2,3],[0.6900, 0.7250,0.7075], color="blue",label="Our accuracy")
#ax.scatter([1,2,3],filteredmean(data[['EN','ES','AVG']]), color="red")
plt.legend()
#plt.title("Results of all participants, excluding outliers")
plt.savefig("result.png")
#plt.show()
