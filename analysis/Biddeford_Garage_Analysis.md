```python
# ! pip install mapclassify
# ! pip install fiona
# ! pip install shapely descartes geopandas

from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import fiona
import geopandas as gdp

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
```

## About


Biddeford, Maine, just went through a contentious mayoral campaign where the big issue centered around adding a giant parking lot in the heart of the downtown and whether it was a smart idea to continue being so development focus in that one spot. The incumbent heavily favored the continued development push while the challenger gained his legitimacy to run via earlier challenges to the parking lot development pushes. The incumbent won by a little over 200 votes so this garage will likely be built but the opposing side never ran the analytics on other scenarios.

**add tweets from candidates?**

### Biddeford, Maine

<img src="map-biddeford-maine-satellite.jpg" />




## Research Question
This begs the question: are there other possible options to increase overall valuation of the city in a similar fashion? 


## The Big Picture
Biddeford, Maine is a "city" of a little over 20,000 year long residents. Whether a parking gararge is developed or not likely will not have far reaching consequences. What can though is showing others that it is possible for reqular citizens to critically analyaze their municipality's fiannces in a sophisitcated fashion.


## Criteria 
City Hall already created a presentation with predicitions on the value that the garage will produce for the city which means there's a rough benchmark to grade ideas against. `$16,407,604` in property taxes in the first 10 years of operation and a benefit of `$39,772,744` over the 25 year lifetime of the agreement: https://www.biddefordmaine.org/2913/Downtown-Parking-Garage-Project-Informat.

To keep things simple and focused on other possible growth solutions instead of analysing the results posted on the city's website we are going to avoid doing typical financial analysis like net projected value calculations, discounting future dollars and accounting for inflation. 


## Data
We will be valuing everything in late 2019/ early 2020 dollars so will use financial year 2020, which ends June 2020, stats as a base for all analysis.


**need to update how data was obtained**
Biddeford does not have their propery assessment data available in a `SQL`,`CSV`, or any other other file types readily available to download that was used in class but they do provide a detail database in `dbf` formatting and `shapefile`s with much of the `dbf` data joined with it. All of this data can be found in the GIS section of the city's website: https://www.biddefordmaine.org/2522/GIS-Data-Catalog

There is also a seperate DBF file with just property related info called `Vision Parcel Data.` Doing a clean convert to CSV for this and the other joined databased is not smooth because of poorly formated records tracking previous owner(s) data. Luckily this study is agnostic to owner related data and not looking to anylyse sale data.

The joined data ended up being used because it produces less wrong results when comparing publicaly stated total property tax revenues less all other publicaly stated considerations. Someone who knows public finance likely can spot the error. It was also better formatted and needed little cleaning besides removing columns (variables) tracking personal identification and political district info for local, state and federal.

There are two officail versions of the dataset that came with the GIS and Parcels data joined - `.dbf` and a `.shp` file. The `.dbf` was marginally cleaned via dropping personal id and political districting info. The rest of the data set was not editted minus minor column name edits for the `CSV` file version.

Again, this exercise to think of other viable total property growth strategies for the city

## Steps
- Download Data
- Understand and Clean Data
- Verify Assumptions
- Visualize and Ananlyze Data
- Suggest Proposals
- Conclusions

### Download Data


```python
# cleaned property value data - not being used
nonjoined_biddeford_tax = r'vision-2018-2019.csv'
basictaxdata = pd.read_csv(nonjoined_biddeford_tax, sep=',')

# uncleaned parcel data - meant to be inner joined w/ basictaxdata
# to see if this elimanates what seems to be sporidic junk data in 
# joined data set given by muni
# did: tax_parcels_joined = gdp.concat([basictaxdata, parcels], axis=1, join='inner')
# but it broke geo heat maps of property values
nonjoin_parcels = 'Parcels/Parcels.shp'
parcels = gdp.read_file(nonjoin_parcels)
#tax_columns = basictaxdata.columns
#parcels_columns = parcels.columns
#parcels.describe().transpose()
# basictaxdata['Bed_Cnt']
#tax_columns
# parcels['LinkID'].unique().shape
# (7594,)
# basictaxdata['GISID'].unique().shape
# 7611,
## parcels.shape looks like LinkID and GISID are keys for the two databases
# tax_parcels_joined.head(30)
# biddeford_parcels = tax_parcels_joined
# taxdata = tax_parcels_joined

# used for heat maps
joined_vision_parcels = 'Parcels/Parcels_Vision_Join.shp'
biddeford_parcels = gdp.read_file(joined_vision_parcels)

# This will be used for all other analysis
cleaned_csv_copy = r'parcels-joined-cleaned.csv'
taxdata = pd.read_csv(cleaned_csv_copy, sep=',')
```

### Understand and Clean Data

Taking a moment to explore and clean your data set allows one to gain some understanding of what possible methods one can take in tackling the project and make your dataset eaiser to use. Things to check while doing this are:

- printing the dataset
- checking shape
- listing columns
- checking descreptive stats
- summation of columns


```python
taxdata
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Map</th>
      <th>Lot</th>
      <th>SubLot</th>
      <th>Polytype</th>
      <th>LinkID</th>
      <th>TM_update_</th>
      <th>Hist_Dist</th>
      <th>Pine_Tr_Zn</th>
      <th>OL_CAC</th>
      <th>Lot_Sub</th>
      <th>...</th>
      <th>Sale_Date</th>
      <th>Year_Built</th>
      <th>Land_Area</th>
      <th>Build_Styl</th>
      <th>Bath_Cnt</th>
      <th>Exempt_Val</th>
      <th>Bed_Cnt</th>
      <th>Tax_Total</th>
      <th>LU_CODE</th>
      <th>LU_DESC4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>parcel</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>parcel</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>parcel</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>parcel</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>parcel</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>8027</td>
      <td>9</td>
      <td>9</td>
      <td>1</td>
      <td>parcel</td>
      <td>9-9-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9-1</td>
      <td>...</td>
      <td>04/11/99</td>
      <td>1993</td>
      <td>139392</td>
      <td>Ranch</td>
      <td>1.0</td>
      <td>20000</td>
      <td>2.0</td>
      <td>269600</td>
      <td>1010</td>
      <td>SINGLE FAM  MDL-01</td>
    </tr>
    <tr>
      <td>8028</td>
      <td>9</td>
      <td>9</td>
      <td>2</td>
      <td>parcel</td>
      <td>9-9-2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9-2</td>
      <td>...</td>
      <td>08/19/91</td>
      <td>1992</td>
      <td>43996</td>
      <td>Cape</td>
      <td>1.5</td>
      <td>20000</td>
      <td>3.0</td>
      <td>228500</td>
      <td>1010</td>
      <td>SINGLE FAM  MDL-01</td>
    </tr>
    <tr>
      <td>8029</td>
      <td>9</td>
      <td>9</td>
      <td>3</td>
      <td>parcel</td>
      <td>9-9-3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9-3</td>
      <td>...</td>
      <td>12/18/12</td>
      <td>2000</td>
      <td>195584</td>
      <td>Cape</td>
      <td>2.0</td>
      <td>0</td>
      <td>2.0</td>
      <td>243700</td>
      <td>1010</td>
      <td>SINGLE FAM  MDL-01</td>
    </tr>
    <tr>
      <td>8030</td>
      <td>9</td>
      <td>9</td>
      <td>4</td>
      <td>parcel</td>
      <td>9-9-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9-4</td>
      <td>...</td>
      <td>10/31/08</td>
      <td>2001</td>
      <td>101059</td>
      <td>Ranch</td>
      <td>1.5</td>
      <td>20000</td>
      <td>3.0</td>
      <td>247500</td>
      <td>1010</td>
      <td>SINGLE FAM  MDL-01</td>
    </tr>
    <tr>
      <td>8031</td>
      <td>9</td>
      <td>9</td>
      <td>5</td>
      <td>parcel</td>
      <td>9-9-5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9-5</td>
      <td>...</td>
      <td>12/18/12</td>
      <td>0</td>
      <td>1000188</td>
      <td>Vacant Land</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>98800</td>
      <td>1300</td>
      <td>RES ACLNDV  MDL-00</td>
    </tr>
  </tbody>
</table>
<p>8032 rows × 53 columns</p>
</div>




```python
taxdata.transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>8022</th>
      <th>8023</th>
      <th>8024</th>
      <th>8025</th>
      <th>8026</th>
      <th>8027</th>
      <th>8028</th>
      <th>8029</th>
      <th>8030</th>
      <th>8031</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Map</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>33</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>...</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <td>Lot</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>10</td>
      <td>10</td>
      <td>...</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <td>SubLot</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <td>Polytype</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>ROW</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>...</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
    </tr>
    <tr>
      <td>LinkID</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10-1</td>
      <td>10-10</td>
      <td>10-10-1</td>
      <td>...</td>
      <td>9-8-4</td>
      <td>9-8-5</td>
      <td>9-8-6</td>
      <td>9-8-7</td>
      <td>9-9</td>
      <td>9-9-1</td>
      <td>9-9-2</td>
      <td>9-9-3</td>
      <td>9-9-4</td>
      <td>9-9-5</td>
    </tr>
    <tr>
      <td>TM_update_</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2012</td>
      <td>2012</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2014</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Hist_Dist</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Pine_Tr_Zn</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>OL_CAC</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>Lot_Sub</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>10</td>
      <td>10-1</td>
      <td>...</td>
      <td>8-4</td>
      <td>8-5</td>
      <td>8-6</td>
      <td>8-7</td>
      <td>9</td>
      <td>9-1</td>
      <td>9-2</td>
      <td>9-3</td>
      <td>9-4</td>
      <td>9-5</td>
    </tr>
    <tr>
      <td>Ward_2013</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>Export_Vis</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>FD</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>14</td>
      <td>Bask</td>
      <td>Stag</td>
      <td>Stag</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <td>X10D_ID</td>
      <td>1000000000</td>
      <td>1000000000</td>
      <td>1000000000</td>
      <td>1000000000</td>
      <td>1000000000</td>
      <td>1000000000</td>
      <td>0</td>
      <td>1010001000</td>
      <td>1010010000</td>
      <td>1010010001</td>
      <td>...</td>
      <td>1009008004</td>
      <td>1009008005</td>
      <td>1009008006</td>
      <td>1009008007</td>
      <td>1009009000</td>
      <td>1009009001</td>
      <td>1009009002</td>
      <td>1009009003</td>
      <td>1009009004</td>
      <td>1009009005</td>
    </tr>
    <tr>
      <td>OBJECTID_1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1581</td>
      <td>1598</td>
      <td>1599</td>
      <td>...</td>
      <td>1507</td>
      <td>1508</td>
      <td>1509</td>
      <td>1510</td>
      <td>1511</td>
      <td>1512</td>
      <td>1513</td>
      <td>1514</td>
      <td>1515</td>
      <td>1516</td>
    </tr>
    <tr>
      <td>Vision_PID</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1091</td>
      <td>1108</td>
      <td>120512</td>
      <td>...</td>
      <td>1021</td>
      <td>1022</td>
      <td>1023</td>
      <td>120768</td>
      <td>1024</td>
      <td>1025</td>
      <td>1026</td>
      <td>100342</td>
      <td>101842</td>
      <td>108300</td>
    </tr>
    <tr>
      <td>GISID</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10-1</td>
      <td>10-10</td>
      <td>10-10-1</td>
      <td>...</td>
      <td>9-8-4</td>
      <td>9-8-5</td>
      <td>9-8-6</td>
      <td>9-8-7</td>
      <td>9-9</td>
      <td>9-9-1</td>
      <td>9-9-2</td>
      <td>9-9-3</td>
      <td>9-9-4</td>
      <td>9-9-5</td>
    </tr>
    <tr>
      <td>F10D_ID</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1010001000</td>
      <td>1010010000</td>
      <td>1010010001</td>
      <td>...</td>
      <td>1009008004</td>
      <td>1009008005</td>
      <td>1009008006</td>
      <td>1009008007</td>
      <td>1009009000</td>
      <td>1009009001</td>
      <td>1009009002</td>
      <td>1009009003</td>
      <td>1009009004</td>
      <td>1009009005</td>
    </tr>
    <tr>
      <td>Map_1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>...</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <td>Block</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>10</td>
      <td>10</td>
      <td>...</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <td>Lot_1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>...</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <td>St_Num</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>66</td>
      <td>56</td>
      <td>62</td>
      <td>0</td>
      <td>625</td>
      <td>40</td>
      <td>42</td>
      <td>32</td>
      <td>36</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Street</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>BASKET ISLAND</td>
      <td>STAGE ISLAND</td>
      <td>STAGE ISLAND</td>
      <td>...</td>
      <td>OLD POOL RD</td>
      <td>OLD POOL RD</td>
      <td>OLD POOL RD</td>
      <td>OLD POOL RD</td>
      <td>POOL ST</td>
      <td>OLD POOL RD</td>
      <td>OLD POOL RD</td>
      <td>OLD POOL RD</td>
      <td>OLD POOL RD</td>
      <td>OLD POOL RD (REAR OF)</td>
    </tr>
    <tr>
      <td>Location</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1 BASKET ISLAND</td>
      <td>1 STAGE ISLAND</td>
      <td>1  STAGE ISLAND</td>
      <td>...</td>
      <td>66 OLD POOL RD</td>
      <td>56 OLD POOL RD</td>
      <td>62 OLD POOL RD</td>
      <td>OLD POOL RD</td>
      <td>625 POOL ST</td>
      <td>40 OLD POOL RD</td>
      <td>42 OLD POOL RD</td>
      <td>32 OLD POOL RD</td>
      <td>36 OLD POOL RD</td>
      <td>OLD POOL RD (REAR OF)</td>
    </tr>
    <tr>
      <td>CurOwn_Dat</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>07/13/09</td>
      <td>10/27/71</td>
      <td>10/27/71</td>
      <td>...</td>
      <td>02/08/16</td>
      <td>01/30/08</td>
      <td>10/16/84</td>
      <td>04/26/12</td>
      <td>04/25/06</td>
      <td>04/11/99</td>
      <td>08/19/91</td>
      <td>12/18/12</td>
      <td>10/31/08</td>
      <td>12/18/12</td>
    </tr>
    <tr>
      <td>CurOwn_QU</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>U</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>Q</td>
      <td>U</td>
      <td>NaN</td>
      <td>U</td>
      <td>Q</td>
      <td>U</td>
      <td>Q</td>
      <td>U</td>
      <td>Q</td>
      <td>U</td>
    </tr>
    <tr>
      <td>CurOwn_VI</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>I</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>I</td>
      <td>I</td>
      <td>NaN</td>
      <td>V</td>
      <td>V</td>
      <td>V</td>
      <td>I</td>
      <td>I</td>
      <td>I</td>
      <td>V</td>
    </tr>
    <tr>
      <td>CurOwn_Sal</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>305000</td>
      <td>0</td>
      <td>0</td>
      <td>118000</td>
      <td>180000</td>
      <td>0</td>
      <td>76237</td>
      <td>215000</td>
      <td>291000</td>
      <td>215000</td>
    </tr>
    <tr>
      <td>PROwn1_Dat</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>07/07/93</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>06/07/01</td>
      <td>06/23/03</td>
      <td>NaN</td>
      <td>10/09/01</td>
      <td>11/03/94</td>
      <td>02/17/94</td>
      <td>05/15/91</td>
      <td>06/28/01</td>
      <td>11/04/02</td>
      <td>06/28/01</td>
    </tr>
    <tr>
      <td>PROwn1_QU</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>U</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>U</td>
      <td>Q</td>
      <td>NaN</td>
      <td>U</td>
      <td>U</td>
      <td>U</td>
      <td>U</td>
      <td>U</td>
      <td>U</td>
      <td>U</td>
    </tr>
    <tr>
      <td>PROwn1_VI</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>I</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>I</td>
      <td>I</td>
      <td>NaN</td>
      <td>I</td>
      <td>V</td>
      <td>I</td>
      <td>V</td>
      <td>I</td>
      <td>I</td>
      <td>V</td>
    </tr>
    <tr>
      <td>PROwn1_Sal</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>269000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>86121</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>PROwn2_Sal</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>259900</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>30000</td>
      <td>60000</td>
      <td>0</td>
    </tr>
    <tr>
      <td>PROwn3_Sal</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>60000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>PROwn4_Sal</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>PROwn5_Sal</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Land_Val</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>300500</td>
      <td>3200</td>
      <td>3800</td>
      <td>...</td>
      <td>119400</td>
      <td>123300</td>
      <td>115300</td>
      <td>7900</td>
      <td>151300</td>
      <td>125500</td>
      <td>115100</td>
      <td>128200</td>
      <td>126400</td>
      <td>98800</td>
    </tr>
    <tr>
      <td>Build_Val</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>31300</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>246300</td>
      <td>165900</td>
      <td>139100</td>
      <td>0</td>
      <td>0</td>
      <td>164100</td>
      <td>133400</td>
      <td>115500</td>
      <td>141100</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Total_Val</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>331800</td>
      <td>3200</td>
      <td>3800</td>
      <td>...</td>
      <td>365700</td>
      <td>289200</td>
      <td>254400</td>
      <td>7900</td>
      <td>151300</td>
      <td>289600</td>
      <td>248500</td>
      <td>243700</td>
      <td>267500</td>
      <td>98800</td>
    </tr>
    <tr>
      <td>Zone</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>CR RP</td>
      <td>CR RP</td>
      <td>CR RP</td>
      <td>...</td>
      <td>CR</td>
      <td>CR</td>
      <td>CR</td>
      <td>CR</td>
      <td>CR</td>
      <td>CR</td>
      <td>CR</td>
      <td>CR</td>
      <td>CR</td>
      <td>CR</td>
    </tr>
    <tr>
      <td>Room_Cnt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>6</td>
      <td>7</td>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5</td>
      <td>6</td>
      <td>5</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Build_SF</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>922</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2238</td>
      <td>1854</td>
      <td>1464</td>
      <td>0</td>
      <td>0</td>
      <td>1120</td>
      <td>1652</td>
      <td>994</td>
      <td>1498</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Sale_Price</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>305000</td>
      <td>0</td>
      <td>0</td>
      <td>118000</td>
      <td>180000</td>
      <td>0</td>
      <td>76237</td>
      <td>215000</td>
      <td>291000</td>
      <td>215000</td>
    </tr>
    <tr>
      <td>Sale_Date</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>07/13/09</td>
      <td>10/27/71</td>
      <td>10/27/71</td>
      <td>...</td>
      <td>02/08/16</td>
      <td>01/30/08</td>
      <td>10/16/84</td>
      <td>04/26/12</td>
      <td>04/25/06</td>
      <td>04/11/99</td>
      <td>08/19/91</td>
      <td>12/18/12</td>
      <td>10/31/08</td>
      <td>12/18/12</td>
    </tr>
    <tr>
      <td>Year_Built</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1915</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1996</td>
      <td>1983</td>
      <td>1985</td>
      <td>0</td>
      <td>0</td>
      <td>1993</td>
      <td>1992</td>
      <td>2000</td>
      <td>2001</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Land_Area</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>16988</td>
      <td>392040</td>
      <td>392040</td>
      <td>...</td>
      <td>91476</td>
      <td>91476</td>
      <td>87120</td>
      <td>980100</td>
      <td>685199</td>
      <td>139392</td>
      <td>43996</td>
      <td>195584</td>
      <td>101059</td>
      <td>1000188</td>
    </tr>
    <tr>
      <td>Build_Styl</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cottage</td>
      <td>Vacant Land</td>
      <td>Vacant Land</td>
      <td>...</td>
      <td>Ranch</td>
      <td>Cape</td>
      <td>Modern/Contemp</td>
      <td>Vacant Land</td>
      <td>Vacant Land</td>
      <td>Ranch</td>
      <td>Cape</td>
      <td>Cape</td>
      <td>Ranch</td>
      <td>Vacant Land</td>
    </tr>
    <tr>
      <td>Bath_Cnt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>1.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>1.5</td>
      <td>2</td>
      <td>1.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Exempt_Val</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>20000</td>
      <td>0</td>
      <td>0</td>
      <td>20000</td>
      <td>20000</td>
      <td>0</td>
      <td>20000</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Bed_Cnt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Tax_Total</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>331800</td>
      <td>3200</td>
      <td>3800</td>
      <td>...</td>
      <td>365700</td>
      <td>289200</td>
      <td>234400</td>
      <td>7900</td>
      <td>151300</td>
      <td>269600</td>
      <td>228500</td>
      <td>243700</td>
      <td>247500</td>
      <td>98800</td>
    </tr>
    <tr>
      <td>LU_CODE</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1012</td>
      <td>9010</td>
      <td>1320</td>
      <td>...</td>
      <td>1010</td>
      <td>1010</td>
      <td>1010</td>
      <td>7200</td>
      <td>1300</td>
      <td>1010</td>
      <td>1010</td>
      <td>1010</td>
      <td>1010</td>
      <td>1300</td>
    </tr>
    <tr>
      <td>LU_DESC4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>OCN FT  MDL-01</td>
      <td>STATE OWND  MDL-00</td>
      <td>RES ACLNUD  MDL-00</td>
      <td>...</td>
      <td>SINGLE FAM  MDL-01</td>
      <td>SINGLE FAM  MDL-01</td>
      <td>SINGLE FAM  MDL-01</td>
      <td>NONPRNECLD  MDL-00</td>
      <td>RES ACLNDV  MDL-00</td>
      <td>SINGLE FAM  MDL-01</td>
      <td>SINGLE FAM  MDL-01</td>
      <td>SINGLE FAM  MDL-01</td>
      <td>SINGLE FAM  MDL-01</td>
      <td>RES ACLNDV  MDL-00</td>
    </tr>
  </tbody>
</table>
<p>53 rows × 8032 columns</p>
</div>



Looks like there's some junk data in this set. Going to `dropna` all records with the `Location` variable set to `NaN`


```python
cleanedtax = taxdata.dropna(subset=['Location'])
```


```python
cleanedtax
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Map</th>
      <th>Lot</th>
      <th>SubLot</th>
      <th>Polytype</th>
      <th>LinkID</th>
      <th>TM_update_</th>
      <th>Hist_Dist</th>
      <th>Pine_Tr_Zn</th>
      <th>OL_CAC</th>
      <th>Lot_Sub</th>
      <th>...</th>
      <th>Sale_Date</th>
      <th>Year_Built</th>
      <th>Land_Area</th>
      <th>Build_Styl</th>
      <th>Bath_Cnt</th>
      <th>Exempt_Val</th>
      <th>Bed_Cnt</th>
      <th>Tax_Total</th>
      <th>LU_CODE</th>
      <th>LU_DESC4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>7</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>parcel</td>
      <td>10-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>07/13/09</td>
      <td>1915</td>
      <td>16988</td>
      <td>Cottage</td>
      <td>0.5</td>
      <td>0</td>
      <td>3.0</td>
      <td>331800</td>
      <td>1012</td>
      <td>OCN FT  MDL-01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>10</td>
      <td>10</td>
      <td>0</td>
      <td>parcel</td>
      <td>10-10</td>
      <td>2012</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>10</td>
      <td>...</td>
      <td>10/27/71</td>
      <td>0</td>
      <td>392040</td>
      <td>Vacant Land</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>3200</td>
      <td>9010</td>
      <td>STATE OWND  MDL-00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>10</td>
      <td>10</td>
      <td>1</td>
      <td>parcel</td>
      <td>10-10-1</td>
      <td>2012</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>10-1</td>
      <td>...</td>
      <td>10/27/71</td>
      <td>0</td>
      <td>392040</td>
      <td>Vacant Land</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>3800</td>
      <td>1320</td>
      <td>RES ACLNUD  MDL-00</td>
    </tr>
    <tr>
      <td>10</td>
      <td>10</td>
      <td>11</td>
      <td>1</td>
      <td>parcel</td>
      <td>10-11-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>11-1</td>
      <td>...</td>
      <td>12/11/15</td>
      <td>0</td>
      <td>599821</td>
      <td>Vacant Land</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>334800</td>
      <td>9000</td>
      <td>US GOVT  MDL-00</td>
    </tr>
    <tr>
      <td>11</td>
      <td>10</td>
      <td>12</td>
      <td>0</td>
      <td>parcel</td>
      <td>10-12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>12</td>
      <td>...</td>
      <td>10/26/02</td>
      <td>1998</td>
      <td>30492</td>
      <td>Colonial</td>
      <td>2.0</td>
      <td>0</td>
      <td>3.0</td>
      <td>302300</td>
      <td>1010</td>
      <td>SINGLE FAM  MDL-01</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>8027</td>
      <td>9</td>
      <td>9</td>
      <td>1</td>
      <td>parcel</td>
      <td>9-9-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9-1</td>
      <td>...</td>
      <td>04/11/99</td>
      <td>1993</td>
      <td>139392</td>
      <td>Ranch</td>
      <td>1.0</td>
      <td>20000</td>
      <td>2.0</td>
      <td>269600</td>
      <td>1010</td>
      <td>SINGLE FAM  MDL-01</td>
    </tr>
    <tr>
      <td>8028</td>
      <td>9</td>
      <td>9</td>
      <td>2</td>
      <td>parcel</td>
      <td>9-9-2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9-2</td>
      <td>...</td>
      <td>08/19/91</td>
      <td>1992</td>
      <td>43996</td>
      <td>Cape</td>
      <td>1.5</td>
      <td>20000</td>
      <td>3.0</td>
      <td>228500</td>
      <td>1010</td>
      <td>SINGLE FAM  MDL-01</td>
    </tr>
    <tr>
      <td>8029</td>
      <td>9</td>
      <td>9</td>
      <td>3</td>
      <td>parcel</td>
      <td>9-9-3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9-3</td>
      <td>...</td>
      <td>12/18/12</td>
      <td>2000</td>
      <td>195584</td>
      <td>Cape</td>
      <td>2.0</td>
      <td>0</td>
      <td>2.0</td>
      <td>243700</td>
      <td>1010</td>
      <td>SINGLE FAM  MDL-01</td>
    </tr>
    <tr>
      <td>8030</td>
      <td>9</td>
      <td>9</td>
      <td>4</td>
      <td>parcel</td>
      <td>9-9-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9-4</td>
      <td>...</td>
      <td>10/31/08</td>
      <td>2001</td>
      <td>101059</td>
      <td>Ranch</td>
      <td>1.5</td>
      <td>20000</td>
      <td>3.0</td>
      <td>247500</td>
      <td>1010</td>
      <td>SINGLE FAM  MDL-01</td>
    </tr>
    <tr>
      <td>8031</td>
      <td>9</td>
      <td>9</td>
      <td>5</td>
      <td>parcel</td>
      <td>9-9-5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9-5</td>
      <td>...</td>
      <td>12/18/12</td>
      <td>0</td>
      <td>1000188</td>
      <td>Vacant Land</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>98800</td>
      <td>1300</td>
      <td>RES ACLNDV  MDL-00</td>
    </tr>
  </tbody>
</table>
<p>8018 rows × 53 columns</p>
</div>




```python
# There seems to be some columns that this research does not care about like owner and political district data.
# Going to pring out all columns and create a list of the columns that this research does not need so the data 
# can be systemically deleted from the dataframe.

cleanedtax.columns
```




    Index(['Map', 'Lot', 'SubLot', 'Polytype', 'LinkID', 'TM_update_', 'Hist_Dist',
           'Pine_Tr_Zn', 'OL_CAC', 'Lot_Sub', 'Ward_2013', 'Export_Vis', 'FD',
           'X10D_ID', 'OBJECTID_1', 'Vision_PID', 'GISID', 'F10D_ID', 'Map_1',
           'Block', 'Lot_1', 'St_Num', 'Street', 'Location', 'CurOwn_Dat',
           'CurOwn_QU', 'CurOwn_VI', 'CurOwn_Sal', 'PROwn1_Dat', 'PROwn1_QU',
           'PROwn1_VI', 'PROwn1_Sal', 'PROwn2_Sal', 'PROwn3_Sal', 'PROwn4_Sal',
           'PROwn5_Sal', 'Land_Val', 'Build_Val', 'Total_Val', 'Zone', 'Room_Cnt',
           'Build_SF', 'Sale_Price', 'Sale_Date', 'Year_Built', 'Land_Area',
           'Build_Styl', 'Bath_Cnt', 'Exempt_Val', 'Bed_Cnt', 'Tax_Total',
           'LU_CODE', 'LU_DESC4'],
          dtype='object')




```python
#'Map_1', 'Block', 'Lot_1' appear to duplicates to 'Map', 'Lot', 'SubLot'
# so adding a set to the delete these columns list as well

# 'OL_CAC', looks useless as well

delete_these_columns = [
'CurOwn_Dat',
'CurOwn_QU',
'CurOwn_VI',
'CurOwn_Sal',
'Sale_Price',
'Sale_Date',
'TM_update_',
'Map_1',
'Block',
'Lot_1',
'OL_CAC'
]


cleanedtax = cleanedtax.drop(delete_these_columns, axis=1)
```


```python
cleanedtax.columns
```




    Index(['Map', 'Lot', 'SubLot', 'Polytype', 'LinkID', 'Hist_Dist', 'Pine_Tr_Zn',
           'Lot_Sub', 'Ward_2013', 'Export_Vis', 'FD', 'X10D_ID', 'OBJECTID_1',
           'Vision_PID', 'GISID', 'F10D_ID', 'St_Num', 'Street', 'Location',
           'PROwn1_Dat', 'PROwn1_QU', 'PROwn1_VI', 'PROwn1_Sal', 'PROwn2_Sal',
           'PROwn3_Sal', 'PROwn4_Sal', 'PROwn5_Sal', 'Land_Val', 'Build_Val',
           'Total_Val', 'Zone', 'Room_Cnt', 'Build_SF', 'Year_Built', 'Land_Area',
           'Build_Styl', 'Bath_Cnt', 'Exempt_Val', 'Bed_Cnt', 'Tax_Total',
           'LU_CODE', 'LU_DESC4'],
          dtype='object')




```python
# appears the .columns method does not call all columns in dataframe
# need to do more cleaning

more_column_deletes = [
'Ward_2013',
'PROwn1_Dat',
'PROwn1_QU',
'PROwn1_VI',
'PROwn1_Sal',
'PROwn2_Sal',
'PROwn3_Sal',
'PROwn4_Sal',
'PROwn5_Sal'
]


cleanedtax = cleanedtax.drop(more_column_deletes, axis=1)
```


```python
cleanedtax = cleanedtax.drop(['Export_Vis'], axis=1)

cleanedtax
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Map</th>
      <th>Lot</th>
      <th>SubLot</th>
      <th>Polytype</th>
      <th>LinkID</th>
      <th>Hist_Dist</th>
      <th>Pine_Tr_Zn</th>
      <th>Lot_Sub</th>
      <th>FD</th>
      <th>X10D_ID</th>
      <th>...</th>
      <th>Build_SF</th>
      <th>Year_Built</th>
      <th>Land_Area</th>
      <th>Build_Styl</th>
      <th>Bath_Cnt</th>
      <th>Exempt_Val</th>
      <th>Bed_Cnt</th>
      <th>Tax_Total</th>
      <th>LU_CODE</th>
      <th>LU_DESC4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>7</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>parcel</td>
      <td>10-1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Bask</td>
      <td>1010001000</td>
      <td>...</td>
      <td>922</td>
      <td>1915</td>
      <td>16988</td>
      <td>Cottage</td>
      <td>0.5</td>
      <td>0</td>
      <td>3.0</td>
      <td>331800</td>
      <td>1012</td>
      <td>OCN FT  MDL-01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>10</td>
      <td>10</td>
      <td>0</td>
      <td>parcel</td>
      <td>10-10</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>Stag</td>
      <td>1010010000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>392040</td>
      <td>Vacant Land</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>3200</td>
      <td>9010</td>
      <td>STATE OWND  MDL-00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>10</td>
      <td>10</td>
      <td>1</td>
      <td>parcel</td>
      <td>10-10-1</td>
      <td>0</td>
      <td>0</td>
      <td>10-1</td>
      <td>Stag</td>
      <td>1010010001</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>392040</td>
      <td>Vacant Land</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>3800</td>
      <td>1320</td>
      <td>RES ACLNUD  MDL-00</td>
    </tr>
    <tr>
      <td>10</td>
      <td>10</td>
      <td>11</td>
      <td>1</td>
      <td>parcel</td>
      <td>10-11-1</td>
      <td>0</td>
      <td>0</td>
      <td>11-1</td>
      <td>3</td>
      <td>1010011001</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>599821</td>
      <td>Vacant Land</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>334800</td>
      <td>9000</td>
      <td>US GOVT  MDL-00</td>
    </tr>
    <tr>
      <td>11</td>
      <td>10</td>
      <td>12</td>
      <td>0</td>
      <td>parcel</td>
      <td>10-12</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>3</td>
      <td>1010012000</td>
      <td>...</td>
      <td>1664</td>
      <td>1998</td>
      <td>30492</td>
      <td>Colonial</td>
      <td>2.0</td>
      <td>0</td>
      <td>3.0</td>
      <td>302300</td>
      <td>1010</td>
      <td>SINGLE FAM  MDL-01</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>8027</td>
      <td>9</td>
      <td>9</td>
      <td>1</td>
      <td>parcel</td>
      <td>9-9-1</td>
      <td>0</td>
      <td>0</td>
      <td>9-1</td>
      <td>3</td>
      <td>1009009001</td>
      <td>...</td>
      <td>1120</td>
      <td>1993</td>
      <td>139392</td>
      <td>Ranch</td>
      <td>1.0</td>
      <td>20000</td>
      <td>2.0</td>
      <td>269600</td>
      <td>1010</td>
      <td>SINGLE FAM  MDL-01</td>
    </tr>
    <tr>
      <td>8028</td>
      <td>9</td>
      <td>9</td>
      <td>2</td>
      <td>parcel</td>
      <td>9-9-2</td>
      <td>0</td>
      <td>0</td>
      <td>9-2</td>
      <td>3</td>
      <td>1009009002</td>
      <td>...</td>
      <td>1652</td>
      <td>1992</td>
      <td>43996</td>
      <td>Cape</td>
      <td>1.5</td>
      <td>20000</td>
      <td>3.0</td>
      <td>228500</td>
      <td>1010</td>
      <td>SINGLE FAM  MDL-01</td>
    </tr>
    <tr>
      <td>8029</td>
      <td>9</td>
      <td>9</td>
      <td>3</td>
      <td>parcel</td>
      <td>9-9-3</td>
      <td>0</td>
      <td>0</td>
      <td>9-3</td>
      <td>3</td>
      <td>1009009003</td>
      <td>...</td>
      <td>994</td>
      <td>2000</td>
      <td>195584</td>
      <td>Cape</td>
      <td>2.0</td>
      <td>0</td>
      <td>2.0</td>
      <td>243700</td>
      <td>1010</td>
      <td>SINGLE FAM  MDL-01</td>
    </tr>
    <tr>
      <td>8030</td>
      <td>9</td>
      <td>9</td>
      <td>4</td>
      <td>parcel</td>
      <td>9-9-4</td>
      <td>0</td>
      <td>0</td>
      <td>9-4</td>
      <td>3</td>
      <td>1009009004</td>
      <td>...</td>
      <td>1498</td>
      <td>2001</td>
      <td>101059</td>
      <td>Ranch</td>
      <td>1.5</td>
      <td>20000</td>
      <td>3.0</td>
      <td>247500</td>
      <td>1010</td>
      <td>SINGLE FAM  MDL-01</td>
    </tr>
    <tr>
      <td>8031</td>
      <td>9</td>
      <td>9</td>
      <td>5</td>
      <td>parcel</td>
      <td>9-9-5</td>
      <td>0</td>
      <td>0</td>
      <td>9-5</td>
      <td>3</td>
      <td>1009009005</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1000188</td>
      <td>Vacant Land</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>98800</td>
      <td>1300</td>
      <td>RES ACLNDV  MDL-00</td>
    </tr>
  </tbody>
</table>
<p>8018 rows × 32 columns</p>
</div>



#### Check Shape

This is down to understand the size of data set


```python
cleanedtax.shape
```




    (8018, 32)



#### List Columns

This checks the type of variables each sample in theory should be tracking


```python
cleanedtax.columns
```




    Index(['Map', 'Lot', 'SubLot', 'Polytype', 'LinkID', 'Hist_Dist', 'Pine_Tr_Zn',
           'Lot_Sub', 'FD', 'X10D_ID', 'OBJECTID_1', 'Vision_PID', 'GISID',
           'F10D_ID', 'St_Num', 'Street', 'Location', 'Land_Val', 'Build_Val',
           'Total_Val', 'Zone', 'Room_Cnt', 'Build_SF', 'Year_Built', 'Land_Area',
           'Build_Styl', 'Bath_Cnt', 'Exempt_Val', 'Bed_Cnt', 'Tax_Total',
           'LU_CODE', 'LU_DESC4'],
          dtype='object')




```python
cleanedtax.describe().transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Map</td>
      <td>8018.0</td>
      <td>3.593303e+01</td>
      <td>2.319166e+01</td>
      <td>1.000000e+00</td>
      <td>2.000000e+01</td>
      <td>3.400000e+01</td>
      <td>5.000000e+01</td>
      <td>8.800000e+01</td>
    </tr>
    <tr>
      <td>Lot</td>
      <td>8018.0</td>
      <td>7.111537e+01</td>
      <td>8.370630e+01</td>
      <td>1.000000e+00</td>
      <td>1.700000e+01</td>
      <td>3.900000e+01</td>
      <td>8.800000e+01</td>
      <td>4.760000e+02</td>
    </tr>
    <tr>
      <td>SubLot</td>
      <td>8018.0</td>
      <td>8.509603e-01</td>
      <td>2.475750e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>3.100000e+01</td>
    </tr>
    <tr>
      <td>Hist_Dist</td>
      <td>8018.0</td>
      <td>2.419556e-02</td>
      <td>2.128778e-01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
    </tr>
    <tr>
      <td>Pine_Tr_Zn</td>
      <td>8018.0</td>
      <td>1.970566e-02</td>
      <td>2.162439e-01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>3.000000e+00</td>
    </tr>
    <tr>
      <td>X10D_ID</td>
      <td>8018.0</td>
      <td>1.036004e+09</td>
      <td>2.318829e+07</td>
      <td>1.001001e+09</td>
      <td>1.020046e+09</td>
      <td>1.034252e+09</td>
      <td>1.050024e+09</td>
      <td>1.088055e+09</td>
    </tr>
    <tr>
      <td>OBJECTID_1</td>
      <td>8018.0</td>
      <td>4.028685e+03</td>
      <td>2.320216e+03</td>
      <td>1.000000e+00</td>
      <td>2.023250e+03</td>
      <td>4.031500e+03</td>
      <td>6.036750e+03</td>
      <td>8.043000e+03</td>
    </tr>
    <tr>
      <td>Vision_PID</td>
      <td>8018.0</td>
      <td>1.617226e+04</td>
      <td>3.449062e+04</td>
      <td>8.000000e+00</td>
      <td>2.162250e+03</td>
      <td>4.200500e+03</td>
      <td>6.274750e+03</td>
      <td>1.221280e+05</td>
    </tr>
    <tr>
      <td>F10D_ID</td>
      <td>8018.0</td>
      <td>1.035249e+09</td>
      <td>3.660295e+07</td>
      <td>0.000000e+00</td>
      <td>1.020045e+09</td>
      <td>1.034252e+09</td>
      <td>1.050024e+09</td>
      <td>1.088055e+09</td>
    </tr>
    <tr>
      <td>St_Num</td>
      <td>8018.0</td>
      <td>9.540758e+01</td>
      <td>1.489711e+02</td>
      <td>0.000000e+00</td>
      <td>8.000000e+00</td>
      <td>2.500000e+01</td>
      <td>1.100000e+02</td>
      <td>9.240000e+02</td>
    </tr>
    <tr>
      <td>Land_Val</td>
      <td>8018.0</td>
      <td>1.384926e+05</td>
      <td>2.513711e+05</td>
      <td>0.000000e+00</td>
      <td>5.510000e+04</td>
      <td>6.920000e+04</td>
      <td>9.847500e+04</td>
      <td>6.395900e+06</td>
    </tr>
    <tr>
      <td>Build_Val</td>
      <td>8018.0</td>
      <td>2.000633e+05</td>
      <td>9.857796e+05</td>
      <td>0.000000e+00</td>
      <td>8.790000e+04</td>
      <td>1.268000e+05</td>
      <td>1.813000e+05</td>
      <td>5.735940e+07</td>
    </tr>
    <tr>
      <td>Total_Val</td>
      <td>8018.0</td>
      <td>3.385558e+05</td>
      <td>1.127664e+06</td>
      <td>0.000000e+00</td>
      <td>1.546250e+05</td>
      <td>2.074000e+05</td>
      <td>2.988750e+05</td>
      <td>6.264060e+07</td>
    </tr>
    <tr>
      <td>Room_Cnt</td>
      <td>6146.0</td>
      <td>7.263261e+00</td>
      <td>3.265074e+00</td>
      <td>0.000000e+00</td>
      <td>5.000000e+00</td>
      <td>6.000000e+00</td>
      <td>8.000000e+00</td>
      <td>3.000000e+01</td>
    </tr>
    <tr>
      <td>Build_SF</td>
      <td>8018.0</td>
      <td>2.562432e+03</td>
      <td>8.595049e+03</td>
      <td>0.000000e+00</td>
      <td>1.050000e+03</td>
      <td>1.540500e+03</td>
      <td>2.308750e+03</td>
      <td>2.625800e+05</td>
    </tr>
    <tr>
      <td>Year_Built</td>
      <td>8018.0</td>
      <td>1.699273e+03</td>
      <td>6.548728e+02</td>
      <td>0.000000e+00</td>
      <td>1.900000e+03</td>
      <td>1.950000e+03</td>
      <td>1.984000e+03</td>
      <td>2.018000e+03</td>
    </tr>
    <tr>
      <td>Land_Area</td>
      <td>8018.0</td>
      <td>1.401570e+05</td>
      <td>8.779283e+05</td>
      <td>0.000000e+00</td>
      <td>7.841000e+03</td>
      <td>1.524600e+04</td>
      <td>4.356000e+04</td>
      <td>5.732496e+07</td>
    </tr>
    <tr>
      <td>Bath_Cnt</td>
      <td>5795.0</td>
      <td>1.894737e+00</td>
      <td>8.811677e-01</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>2.000000e+00</td>
      <td>2.500000e+00</td>
      <td>7.500000e+00</td>
    </tr>
    <tr>
      <td>Exempt_Val</td>
      <td>8018.0</td>
      <td>8.714630e+03</td>
      <td>1.034775e+04</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+04</td>
      <td>3.200000e+04</td>
    </tr>
    <tr>
      <td>Bed_Cnt</td>
      <td>6365.0</td>
      <td>3.422781e+00</td>
      <td>1.406263e+00</td>
      <td>0.000000e+00</td>
      <td>3.000000e+00</td>
      <td>3.000000e+00</td>
      <td>4.000000e+00</td>
      <td>9.000000e+00</td>
    </tr>
    <tr>
      <td>Tax_Total</td>
      <td>8018.0</td>
      <td>3.298412e+05</td>
      <td>1.128469e+06</td>
      <td>0.000000e+00</td>
      <td>1.428000e+05</td>
      <td>1.949000e+05</td>
      <td>2.870000e+05</td>
      <td>6.264060e+07</td>
    </tr>
  </tbody>
</table>
</div>



There are records with `Land_Val`, `Build_Val`, `Total_Val`, and `Tax_Total` as zero. This confirms the need to sort and check the max and min records for these varaibles - especially for `Total_Val` and `Tax_Total`


```python
cleanedtax = cleanedtax.sort_values(['Total_Val'])

cleanedtax.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Map</th>
      <th>Lot</th>
      <th>SubLot</th>
      <th>Polytype</th>
      <th>LinkID</th>
      <th>Hist_Dist</th>
      <th>Pine_Tr_Zn</th>
      <th>Lot_Sub</th>
      <th>FD</th>
      <th>X10D_ID</th>
      <th>...</th>
      <th>Build_SF</th>
      <th>Year_Built</th>
      <th>Land_Area</th>
      <th>Build_Styl</th>
      <th>Bath_Cnt</th>
      <th>Exempt_Val</th>
      <th>Bed_Cnt</th>
      <th>Tax_Total</th>
      <th>LU_CODE</th>
      <th>LU_DESC4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1988</td>
      <td>28</td>
      <td>53</td>
      <td>0</td>
      <td>parcel</td>
      <td>28-53</td>
      <td>0</td>
      <td>0</td>
      <td>53</td>
      <td>11</td>
      <td>1028053000</td>
      <td>...</td>
      <td>0</td>
      <td>1890</td>
      <td>0</td>
      <td>Condominium</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>995</td>
      <td>CONDO MAIN</td>
    </tr>
    <tr>
      <td>2666</td>
      <td>33</td>
      <td>15</td>
      <td>0</td>
      <td>parcel</td>
      <td>33-15</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>13</td>
      <td>1033015000</td>
      <td>...</td>
      <td>0</td>
      <td>1986</td>
      <td>0</td>
      <td>Condominium</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>995</td>
      <td>CONDO MAIN</td>
    </tr>
    <tr>
      <td>2447</td>
      <td>31</td>
      <td>10</td>
      <td>0</td>
      <td>parcel</td>
      <td>31-10</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>14</td>
      <td>1031010000</td>
      <td>...</td>
      <td>0</td>
      <td>2009</td>
      <td>99317</td>
      <td>Condominium</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>995</td>
      <td>CONDO MAIN</td>
    </tr>
    <tr>
      <td>500</td>
      <td>17</td>
      <td>22</td>
      <td>0</td>
      <td>parcel</td>
      <td>17-22</td>
      <td>0</td>
      <td>0</td>
      <td>22</td>
      <td>18</td>
      <td>1017022000</td>
      <td>...</td>
      <td>0</td>
      <td>1986</td>
      <td>0</td>
      <td>Condo Office</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>995</td>
      <td>CONDO MAIN</td>
    </tr>
    <tr>
      <td>1344</td>
      <td>2</td>
      <td>56</td>
      <td>1</td>
      <td>parcel</td>
      <td>2-56-1</td>
      <td>0</td>
      <td>0</td>
      <td>56-1</td>
      <td>18</td>
      <td>1002056001</td>
      <td>...</td>
      <td>0</td>
      <td>1984</td>
      <td>0</td>
      <td>Condo Office</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>995</td>
      <td>CONDO MAIN</td>
    </tr>
    <tr>
      <td>708</td>
      <td>20</td>
      <td>24</td>
      <td>0</td>
      <td>parcel</td>
      <td>20-24</td>
      <td>0</td>
      <td>0</td>
      <td>24</td>
      <td>18</td>
      <td>1020024000</td>
      <td>...</td>
      <td>0</td>
      <td>1988</td>
      <td>0</td>
      <td>Condo Office</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>995</td>
      <td>CONDO MAIN</td>
    </tr>
    <tr>
      <td>2164</td>
      <td>29</td>
      <td>158</td>
      <td>1</td>
      <td>parcel</td>
      <td>29-158-1</td>
      <td>0</td>
      <td>0</td>
      <td>158-1</td>
      <td>11</td>
      <td>1029158001</td>
      <td>...</td>
      <td>0</td>
      <td>1986</td>
      <td>0</td>
      <td>Condominium</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>995</td>
      <td>CONDO MAIN</td>
    </tr>
    <tr>
      <td>5665</td>
      <td>54</td>
      <td>40</td>
      <td>0</td>
      <td>parcel</td>
      <td>54-40</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>3</td>
      <td>1054040000</td>
      <td>...</td>
      <td>0</td>
      <td>1988</td>
      <td>0</td>
      <td>Condominium</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>995</td>
      <td>CONDO MAIN</td>
    </tr>
    <tr>
      <td>5718</td>
      <td>54</td>
      <td>81</td>
      <td>0</td>
      <td>parcel</td>
      <td>54-81</td>
      <td>0</td>
      <td>0</td>
      <td>81</td>
      <td>3</td>
      <td>1054081000</td>
      <td>...</td>
      <td>0</td>
      <td>1970</td>
      <td>0</td>
      <td>Condominium</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>995</td>
      <td>CONDO MAIN</td>
    </tr>
    <tr>
      <td>2955</td>
      <td>34</td>
      <td>205</td>
      <td>0</td>
      <td>parcel</td>
      <td>34-205</td>
      <td>0</td>
      <td>0</td>
      <td>205</td>
      <td>14</td>
      <td>1034205000</td>
      <td>...</td>
      <td>0</td>
      <td>2001</td>
      <td>0</td>
      <td>Condominium</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>995</td>
      <td>CONDO MAIN</td>
    </tr>
    <tr>
      <td>3556</td>
      <td>37</td>
      <td>129</td>
      <td>0</td>
      <td>parcel</td>
      <td>37-129</td>
      <td>0</td>
      <td>0</td>
      <td>129</td>
      <td>13</td>
      <td>1037129000</td>
      <td>...</td>
      <td>0</td>
      <td>1900</td>
      <td>0</td>
      <td>Condominium</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>995</td>
      <td>CONDO MAIN</td>
    </tr>
    <tr>
      <td>7419</td>
      <td>82</td>
      <td>35</td>
      <td>0</td>
      <td>parcel</td>
      <td>82-35</td>
      <td>0</td>
      <td>0</td>
      <td>35</td>
      <td>17</td>
      <td>1082035000</td>
      <td>...</td>
      <td>0</td>
      <td>1997</td>
      <td>0</td>
      <td>Condominium</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>995</td>
      <td>CONDO MAIN</td>
    </tr>
    <tr>
      <td>1887</td>
      <td>28</td>
      <td>22</td>
      <td>0</td>
      <td>parcel</td>
      <td>28-22</td>
      <td>0</td>
      <td>0</td>
      <td>22</td>
      <td>14</td>
      <td>1028022000</td>
      <td>...</td>
      <td>0</td>
      <td>1927</td>
      <td>0</td>
      <td>Condominium</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>995</td>
      <td>CONDO MAIN</td>
    </tr>
    <tr>
      <td>5778</td>
      <td>55</td>
      <td>29</td>
      <td>1</td>
      <td>parcel</td>
      <td>55-29-1</td>
      <td>0</td>
      <td>0</td>
      <td>29-1</td>
      <td>3</td>
      <td>1055029001</td>
      <td>...</td>
      <td>0</td>
      <td>1940</td>
      <td>0</td>
      <td>Condominium</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>995</td>
      <td>CONDO MAIN</td>
    </tr>
    <tr>
      <td>4729</td>
      <td>41</td>
      <td>30</td>
      <td>0</td>
      <td>parcel</td>
      <td>41-30</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>9</td>
      <td>1041030000</td>
      <td>...</td>
      <td>0</td>
      <td>1900</td>
      <td>0</td>
      <td>Condominium</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>995</td>
      <td>CONDO MAIN</td>
    </tr>
    <tr>
      <td>5728</td>
      <td>54</td>
      <td>85</td>
      <td>0</td>
      <td>parcel</td>
      <td>54-85</td>
      <td>0</td>
      <td>0</td>
      <td>85</td>
      <td>3</td>
      <td>1054085000</td>
      <td>...</td>
      <td>0</td>
      <td>1950</td>
      <td>0</td>
      <td>Condominium</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>995</td>
      <td>CONDO MAIN</td>
    </tr>
    <tr>
      <td>3597</td>
      <td>37</td>
      <td>145</td>
      <td>0</td>
      <td>parcel</td>
      <td>37-145</td>
      <td>0</td>
      <td>0</td>
      <td>145</td>
      <td>13</td>
      <td>1037145000</td>
      <td>...</td>
      <td>0</td>
      <td>1988</td>
      <td>0</td>
      <td>Condominium</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>995</td>
      <td>CONDO MAIN</td>
    </tr>
    <tr>
      <td>1192</td>
      <td>2</td>
      <td>42</td>
      <td>5</td>
      <td>parcel</td>
      <td>2-42-5</td>
      <td>0</td>
      <td>3</td>
      <td>42-5</td>
      <td>18</td>
      <td>1002042005</td>
      <td>...</td>
      <td>0</td>
      <td>2008</td>
      <td>0</td>
      <td>Condo Office</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>995</td>
      <td>CONDO MAIN</td>
    </tr>
    <tr>
      <td>1998</td>
      <td>2</td>
      <td>86</td>
      <td>0</td>
      <td>parcel</td>
      <td>2-86</td>
      <td>0</td>
      <td>0</td>
      <td>86</td>
      <td>18</td>
      <td>1002086000</td>
      <td>...</td>
      <td>0</td>
      <td>1977</td>
      <td>0</td>
      <td>Condo Office</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>995</td>
      <td>CONDO MAIN</td>
    </tr>
    <tr>
      <td>2074</td>
      <td>29</td>
      <td>110</td>
      <td>0</td>
      <td>parcel</td>
      <td>29-110</td>
      <td>0</td>
      <td>0</td>
      <td>110</td>
      <td>11</td>
      <td>1029110000</td>
      <td>...</td>
      <td>0</td>
      <td>1986</td>
      <td>0</td>
      <td>Condominium</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>995</td>
      <td>CONDO MAIN</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 32 columns</p>
</div>



These min values need to be analyzed some more. It looks like the city assigns all multi-units as a value of 0 and tracks the value of each individual unit. These 0 have the potential to skew any statistical analysis so going to dig deeper by filtering for all records for `Total_Val` = 0


```python
zero_value_properties = cleanedtax[cleanedtax['Total_Val'] == 0]
zero_value_properties.shape
```




    (29, 32)




```python
zero_value_properties.transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1988</th>
      <th>2666</th>
      <th>2447</th>
      <th>500</th>
      <th>1344</th>
      <th>708</th>
      <th>2164</th>
      <th>5665</th>
      <th>5718</th>
      <th>2955</th>
      <th>...</th>
      <th>2074</th>
      <th>6023</th>
      <th>6976</th>
      <th>685</th>
      <th>2367</th>
      <th>6500</th>
      <th>958</th>
      <th>1091</th>
      <th>1259</th>
      <th>6007</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Map</td>
      <td>28</td>
      <td>33</td>
      <td>31</td>
      <td>17</td>
      <td>2</td>
      <td>20</td>
      <td>29</td>
      <td>54</td>
      <td>54</td>
      <td>34</td>
      <td>...</td>
      <td>29</td>
      <td>59</td>
      <td>7</td>
      <td>19</td>
      <td>29</td>
      <td>64</td>
      <td>22</td>
      <td>2</td>
      <td>2</td>
      <td>59</td>
    </tr>
    <tr>
      <td>Lot</td>
      <td>53</td>
      <td>15</td>
      <td>10</td>
      <td>22</td>
      <td>56</td>
      <td>24</td>
      <td>158</td>
      <td>40</td>
      <td>81</td>
      <td>205</td>
      <td>...</td>
      <td>110</td>
      <td>25</td>
      <td>20</td>
      <td>57</td>
      <td>99</td>
      <td>49</td>
      <td>39</td>
      <td>30</td>
      <td>47</td>
      <td>18</td>
    </tr>
    <tr>
      <td>SubLot</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Polytype</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>...</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>open space</td>
      <td>open space</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>ROW</td>
      <td>parcel</td>
    </tr>
    <tr>
      <td>LinkID</td>
      <td>28-53</td>
      <td>33-15</td>
      <td>31-10</td>
      <td>17-22</td>
      <td>2-56-1</td>
      <td>20-24</td>
      <td>29-158-1</td>
      <td>54-40</td>
      <td>54-81</td>
      <td>34-205</td>
      <td>...</td>
      <td>29-110</td>
      <td>59-25</td>
      <td>7-20</td>
      <td>19-57</td>
      <td>29-99</td>
      <td>64-49</td>
      <td>22-39</td>
      <td>2-30</td>
      <td>2-47</td>
      <td>59-18</td>
    </tr>
    <tr>
      <td>Hist_Dist</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Pine_Tr_Zn</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Lot_Sub</td>
      <td>53</td>
      <td>15</td>
      <td>10</td>
      <td>22</td>
      <td>56-1</td>
      <td>24</td>
      <td>158-1</td>
      <td>40</td>
      <td>81</td>
      <td>205</td>
      <td>...</td>
      <td>110</td>
      <td>25</td>
      <td>20</td>
      <td>57</td>
      <td>99</td>
      <td>49</td>
      <td>39</td>
      <td>30</td>
      <td>47</td>
      <td>18</td>
    </tr>
    <tr>
      <td>FD</td>
      <td>11</td>
      <td>13</td>
      <td>14</td>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>11</td>
      <td>3</td>
      <td>3</td>
      <td>14</td>
      <td>...</td>
      <td>11</td>
      <td>2</td>
      <td>14</td>
      <td>17</td>
      <td>11</td>
      <td>1</td>
      <td>11</td>
      <td>18</td>
      <td>18</td>
      <td>2</td>
    </tr>
    <tr>
      <td>X10D_ID</td>
      <td>1028053000</td>
      <td>1033015000</td>
      <td>1031010000</td>
      <td>1017022000</td>
      <td>1002056001</td>
      <td>1020024000</td>
      <td>1029158001</td>
      <td>1054040000</td>
      <td>1054081000</td>
      <td>1034205000</td>
      <td>...</td>
      <td>1029110000</td>
      <td>1059025000</td>
      <td>1007020000</td>
      <td>1019057000</td>
      <td>1029099000</td>
      <td>1064049000</td>
      <td>1022039000</td>
      <td>1002030000</td>
      <td>1002047000</td>
      <td>1059018000</td>
    </tr>
    <tr>
      <td>OBJECTID_1</td>
      <td>2877</td>
      <td>3613</td>
      <td>3504</td>
      <td>1894</td>
      <td>540</td>
      <td>2042</td>
      <td>3297</td>
      <td>6166</td>
      <td>6213</td>
      <td>3966</td>
      <td>...</td>
      <td>3231</td>
      <td>6478</td>
      <td>69</td>
      <td>2018</td>
      <td>3215</td>
      <td>6879</td>
      <td>2268</td>
      <td>461</td>
      <td>516</td>
      <td>6465</td>
    </tr>
    <tr>
      <td>Vision_PID</td>
      <td>119459</td>
      <td>107728</td>
      <td>2939</td>
      <td>107723</td>
      <td>107722</td>
      <td>107725</td>
      <td>107727</td>
      <td>107731</td>
      <td>107732</td>
      <td>3425</td>
      <td>...</td>
      <td>107726</td>
      <td>107700</td>
      <td>121134</td>
      <td>108388</td>
      <td>2703</td>
      <td>107735</td>
      <td>121469</td>
      <td>108500</td>
      <td>119204</td>
      <td>107734</td>
    </tr>
    <tr>
      <td>GISID</td>
      <td>28-53</td>
      <td>33-15</td>
      <td>31-10</td>
      <td>17-22</td>
      <td>2-56-1</td>
      <td>20-24</td>
      <td>29-158-1</td>
      <td>54-40</td>
      <td>54-81</td>
      <td>34-205</td>
      <td>...</td>
      <td>29-110</td>
      <td>59-25</td>
      <td>7-20</td>
      <td>19-57</td>
      <td>29-99</td>
      <td>64-49</td>
      <td>22-39</td>
      <td>2-30</td>
      <td>2-47</td>
      <td>59-18</td>
    </tr>
    <tr>
      <td>F10D_ID</td>
      <td>1028053000</td>
      <td>1033015000</td>
      <td>1031010000</td>
      <td>1017022000</td>
      <td>1002056001</td>
      <td>1020024000</td>
      <td>1029158001</td>
      <td>1054040000</td>
      <td>1054081000</td>
      <td>1034205000</td>
      <td>...</td>
      <td>1029110000</td>
      <td>1059025000</td>
      <td>1007020000</td>
      <td>1019057000</td>
      <td>1029099000</td>
      <td>1064049000</td>
      <td>1022039000</td>
      <td>1002030000</td>
      <td>1002047000</td>
      <td>1059018000</td>
    </tr>
    <tr>
      <td>St_Num</td>
      <td>317</td>
      <td>91</td>
      <td>0</td>
      <td>24</td>
      <td>2</td>
      <td>409</td>
      <td>108</td>
      <td>8</td>
      <td>110</td>
      <td>236</td>
      <td>...</td>
      <td>55</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>28</td>
      <td>0</td>
      <td>185</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
    </tr>
    <tr>
      <td>Street</td>
      <td>ALFRED ST</td>
      <td>WESTERN AVE</td>
      <td>COLTON LN</td>
      <td>WEST COLE RD</td>
      <td>MEDICAL CENTER DR</td>
      <td>ALFRED ST</td>
      <td>GRANITE ST</td>
      <td>PLEASANT AVE</td>
      <td>HILLS BEACH RD</td>
      <td>ALFRED ST</td>
      <td>...</td>
      <td>WEST ST</td>
      <td>LESTER B ORCUTT BLVD</td>
      <td>BERRY LN</td>
      <td>PONDEROSA LN</td>
      <td>RAYMOND ST</td>
      <td>FORTUNES ROCKS RD</td>
      <td>GRANITE ST</td>
      <td>ALFRED ST</td>
      <td>WELLSPRING RD</td>
      <td>LESTER B ORCUTT BLVD</td>
    </tr>
    <tr>
      <td>Location</td>
      <td>317 ALFRED ST</td>
      <td>91 WESTERN AVE</td>
      <td>COLTON LN</td>
      <td>24 WEST COLE RD</td>
      <td>2 MEDICAL CENTER DR</td>
      <td>409 ALFRED ST</td>
      <td>108 GRANITE ST</td>
      <td>8-12 PLEASANT AVE</td>
      <td>110 HILLS BEACH RD</td>
      <td>236 ALFRED ST</td>
      <td>...</td>
      <td>55 WEST ST</td>
      <td>LESTER B ORCUTT BLVD</td>
      <td>BERRY LN</td>
      <td>PONDEROSA LN</td>
      <td>28 RAYMOND ST</td>
      <td>FORTUNES ROCKS RD</td>
      <td>185 GRANITE ST</td>
      <td>ALFRED ST</td>
      <td>6-6 WELLSPRING RD</td>
      <td>20 LESTER B ORCUTT BLVD</td>
    </tr>
    <tr>
      <td>Land_Val</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Build_Val</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Total_Val</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Zone</td>
      <td>NaN</td>
      <td>R2</td>
      <td>R3</td>
      <td>NaN</td>
      <td>M</td>
      <td>NaN</td>
      <td>R2</td>
      <td>CR</td>
      <td>CR</td>
      <td>R2</td>
      <td>...</td>
      <td>R2</td>
      <td>CR</td>
      <td>R3</td>
      <td>RF/R1A</td>
      <td>R2</td>
      <td>CR</td>
      <td>R1A</td>
      <td>B2</td>
      <td>NaN</td>
      <td>CR</td>
    </tr>
    <tr>
      <td>Room_Cnt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Build_SF</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Year_Built</td>
      <td>1890</td>
      <td>1986</td>
      <td>2009</td>
      <td>1986</td>
      <td>1984</td>
      <td>1988</td>
      <td>1986</td>
      <td>1988</td>
      <td>1970</td>
      <td>2001</td>
      <td>...</td>
      <td>1986</td>
      <td>1900</td>
      <td>2016</td>
      <td>0</td>
      <td>2013</td>
      <td>1950</td>
      <td>2016</td>
      <td>2006</td>
      <td>1989</td>
      <td>1900</td>
    </tr>
    <tr>
      <td>Land_Area</td>
      <td>0</td>
      <td>0</td>
      <td>99317</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>21780</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Build_Styl</td>
      <td>Condominium</td>
      <td>Condominium</td>
      <td>Condominium</td>
      <td>Condo Office</td>
      <td>Condo Office</td>
      <td>Condo Office</td>
      <td>Condominium</td>
      <td>Condominium</td>
      <td>Condominium</td>
      <td>Condominium</td>
      <td>...</td>
      <td>Condominium</td>
      <td>Condominium</td>
      <td>Condominium</td>
      <td>Condominium</td>
      <td>Condominium</td>
      <td>Condominium</td>
      <td>Condominium</td>
      <td>Condominium</td>
      <td>Condo Office</td>
      <td>Condominium</td>
    </tr>
    <tr>
      <td>Bath_Cnt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Exempt_Val</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Bed_Cnt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Tax_Total</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>LU_CODE</td>
      <td>995</td>
      <td>995</td>
      <td>995</td>
      <td>995</td>
      <td>995</td>
      <td>995</td>
      <td>995</td>
      <td>995</td>
      <td>995</td>
      <td>995</td>
      <td>...</td>
      <td>995</td>
      <td>995</td>
      <td>995</td>
      <td>995</td>
      <td>995</td>
      <td>995</td>
      <td>995</td>
      <td>995</td>
      <td>995</td>
      <td>995</td>
    </tr>
    <tr>
      <td>LU_DESC4</td>
      <td>CONDO MAIN</td>
      <td>CONDO MAIN</td>
      <td>CONDO MAIN</td>
      <td>CONDO MAIN</td>
      <td>CONDO MAIN</td>
      <td>CONDO MAIN</td>
      <td>CONDO MAIN</td>
      <td>CONDO MAIN</td>
      <td>CONDO MAIN</td>
      <td>CONDO MAIN</td>
      <td>...</td>
      <td>CONDO MAIN</td>
      <td>CONDO MAIN</td>
      <td>CONDO MAIN</td>
      <td>CONDO MAIN</td>
      <td>CONDO MAIN</td>
      <td>CONDO MAIN</td>
      <td>CONDO MAIN</td>
      <td>CONDO MAIN</td>
      <td>CONDO MAIN</td>
      <td>CONDO MAIN</td>
    </tr>
  </tbody>
</table>
<p>32 rows × 29 columns</p>
</div>




```python
zero_value_properties.describe().transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Map</td>
      <td>29.0</td>
      <td>3.213793e+01</td>
      <td>2.148209e+01</td>
      <td>2.000000e+00</td>
      <td>1.900000e+01</td>
      <td>2.900000e+01</td>
      <td>5.400000e+01</td>
      <td>8.200000e+01</td>
    </tr>
    <tr>
      <td>Lot</td>
      <td>29.0</td>
      <td>6.072414e+01</td>
      <td>4.876174e+01</td>
      <td>1.000000e+01</td>
      <td>2.500000e+01</td>
      <td>4.200000e+01</td>
      <td>8.500000e+01</td>
      <td>2.050000e+02</td>
    </tr>
    <tr>
      <td>SubLot</td>
      <td>29.0</td>
      <td>2.758621e-01</td>
      <td>9.597824e-01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>5.000000e+00</td>
    </tr>
    <tr>
      <td>Hist_Dist</td>
      <td>29.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <td>Pine_Tr_Zn</td>
      <td>29.0</td>
      <td>1.034483e-01</td>
      <td>5.570860e-01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>3.000000e+00</td>
    </tr>
    <tr>
      <td>X10D_ID</td>
      <td>29.0</td>
      <td>1.032199e+09</td>
      <td>2.148216e+07</td>
      <td>1.002030e+09</td>
      <td>1.019057e+09</td>
      <td>1.029158e+09</td>
      <td>1.054040e+09</td>
      <td>1.082035e+09</td>
    </tr>
    <tr>
      <td>OBJECTID_1</td>
      <td>29.0</td>
      <td>3.599172e+03</td>
      <td>2.295672e+03</td>
      <td>6.900000e+01</td>
      <td>2.018000e+03</td>
      <td>3.297000e+03</td>
      <td>6.166000e+03</td>
      <td>7.666000e+03</td>
    </tr>
    <tr>
      <td>Vision_PID</td>
      <td>29.0</td>
      <td>9.553662e+04</td>
      <td>3.786839e+04</td>
      <td>2.703000e+03</td>
      <td>1.077220e+05</td>
      <td>1.077310e+05</td>
      <td>1.080800e+05</td>
      <td>1.214690e+05</td>
    </tr>
    <tr>
      <td>F10D_ID</td>
      <td>29.0</td>
      <td>1.032199e+09</td>
      <td>2.148216e+07</td>
      <td>1.002030e+09</td>
      <td>1.019057e+09</td>
      <td>1.029158e+09</td>
      <td>1.054040e+09</td>
      <td>1.082035e+09</td>
    </tr>
    <tr>
      <td>St_Num</td>
      <td>29.0</td>
      <td>8.889655e+01</td>
      <td>1.194569e+02</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>2.800000e+01</td>
      <td>1.100000e+02</td>
      <td>4.090000e+02</td>
    </tr>
    <tr>
      <td>Land_Val</td>
      <td>29.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <td>Build_Val</td>
      <td>29.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <td>Total_Val</td>
      <td>29.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <td>Room_Cnt</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Build_SF</td>
      <td>29.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <td>Year_Built</td>
      <td>29.0</td>
      <td>1.901759e+03</td>
      <td>3.679108e+02</td>
      <td>0.000000e+00</td>
      <td>1.940000e+03</td>
      <td>1.986000e+03</td>
      <td>1.997000e+03</td>
      <td>2.016000e+03</td>
    </tr>
    <tr>
      <td>Land_Area</td>
      <td>29.0</td>
      <td>4.175759e+03</td>
      <td>1.873934e+04</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>9.931700e+04</td>
    </tr>
    <tr>
      <td>Bath_Cnt</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Exempt_Val</td>
      <td>29.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <td>Bed_Cnt</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Tax_Total</td>
      <td>29.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
  </tbody>
</table>
</div>




```python
zero_value_properties[zero_value_properties['Land_Area'] > 0 ].transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2447</th>
      <th>2367</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Map</td>
      <td>31</td>
      <td>29</td>
    </tr>
    <tr>
      <td>Lot</td>
      <td>10</td>
      <td>99</td>
    </tr>
    <tr>
      <td>SubLot</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Polytype</td>
      <td>parcel</td>
      <td>parcel</td>
    </tr>
    <tr>
      <td>LinkID</td>
      <td>31-10</td>
      <td>29-99</td>
    </tr>
    <tr>
      <td>Hist_Dist</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Pine_Tr_Zn</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Lot_Sub</td>
      <td>10</td>
      <td>99</td>
    </tr>
    <tr>
      <td>FD</td>
      <td>14</td>
      <td>11</td>
    </tr>
    <tr>
      <td>X10D_ID</td>
      <td>1031010000</td>
      <td>1029099000</td>
    </tr>
    <tr>
      <td>OBJECTID_1</td>
      <td>3504</td>
      <td>3215</td>
    </tr>
    <tr>
      <td>Vision_PID</td>
      <td>2939</td>
      <td>2703</td>
    </tr>
    <tr>
      <td>GISID</td>
      <td>31-10</td>
      <td>29-99</td>
    </tr>
    <tr>
      <td>F10D_ID</td>
      <td>1031010000</td>
      <td>1029099000</td>
    </tr>
    <tr>
      <td>St_Num</td>
      <td>0</td>
      <td>28</td>
    </tr>
    <tr>
      <td>Street</td>
      <td>COLTON LN</td>
      <td>RAYMOND ST</td>
    </tr>
    <tr>
      <td>Location</td>
      <td>COLTON LN</td>
      <td>28 RAYMOND ST</td>
    </tr>
    <tr>
      <td>Land_Val</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Build_Val</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Total_Val</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Zone</td>
      <td>R3</td>
      <td>R2</td>
    </tr>
    <tr>
      <td>Room_Cnt</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Build_SF</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Year_Built</td>
      <td>2009</td>
      <td>2013</td>
    </tr>
    <tr>
      <td>Land_Area</td>
      <td>99317</td>
      <td>21780</td>
    </tr>
    <tr>
      <td>Build_Styl</td>
      <td>Condominium</td>
      <td>Condominium</td>
    </tr>
    <tr>
      <td>Bath_Cnt</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Exempt_Val</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Bed_Cnt</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Tax_Total</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>LU_CODE</td>
      <td>995</td>
      <td>995</td>
    </tr>
    <tr>
      <td>LU_DESC4</td>
      <td>CONDO MAIN</td>
      <td>CONDO MAIN</td>
    </tr>
  </tbody>
</table>
</div>



These are the only two properties with assessed property values at 0 but supposedly have land attached to it so assuming this is junk data relative to the pending analysis work in answering research question.

Time to further refine cleanedtax dataframe by dropping all `Total_Val` == 0.


```python
more_junked_data = cleanedtax[cleanedtax['Total_Val'] == 0].index
cleanedtax = cleanedtax.drop(more_junked_data)
```


```python
cleanedtax.describe().transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Map</td>
      <td>7989.0</td>
      <td>3.594680e+01</td>
      <td>2.319775e+01</td>
      <td>1.000000e+00</td>
      <td>2.000000e+01</td>
      <td>3.400000e+01</td>
      <td>5.000000e+01</td>
      <td>8.800000e+01</td>
    </tr>
    <tr>
      <td>Lot</td>
      <td>7989.0</td>
      <td>7.115309e+01</td>
      <td>8.380606e+01</td>
      <td>1.000000e+00</td>
      <td>1.700000e+01</td>
      <td>3.900000e+01</td>
      <td>8.800000e+01</td>
      <td>4.760000e+02</td>
    </tr>
    <tr>
      <td>SubLot</td>
      <td>7989.0</td>
      <td>8.530479e-01</td>
      <td>2.479346e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>3.100000e+01</td>
    </tr>
    <tr>
      <td>Hist_Dist</td>
      <td>7989.0</td>
      <td>2.428339e-02</td>
      <td>2.132589e-01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
    </tr>
    <tr>
      <td>Pine_Tr_Zn</td>
      <td>7989.0</td>
      <td>1.940168e-02</td>
      <td>2.140510e-01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>3.000000e+00</td>
    </tr>
    <tr>
      <td>X10D_ID</td>
      <td>7989.0</td>
      <td>1.036018e+09</td>
      <td>2.319437e+07</td>
      <td>1.001001e+09</td>
      <td>1.020048e+09</td>
      <td>1.034255e+09</td>
      <td>1.050023e+09</td>
      <td>1.088055e+09</td>
    </tr>
    <tr>
      <td>OBJECTID_1</td>
      <td>7989.0</td>
      <td>4.030244e+03</td>
      <td>2.320302e+03</td>
      <td>1.000000e+00</td>
      <td>2.024000e+03</td>
      <td>4.035000e+03</td>
      <td>6.036000e+03</td>
      <td>8.043000e+03</td>
    </tr>
    <tr>
      <td>Vision_PID</td>
      <td>7989.0</td>
      <td>1.588417e+04</td>
      <td>3.414593e+04</td>
      <td>8.000000e+00</td>
      <td>2.155000e+03</td>
      <td>4.190000e+03</td>
      <td>6.257000e+03</td>
      <td>1.221280e+05</td>
    </tr>
    <tr>
      <td>F10D_ID</td>
      <td>7989.0</td>
      <td>1.035260e+09</td>
      <td>3.664681e+07</td>
      <td>0.000000e+00</td>
      <td>1.020047e+09</td>
      <td>1.034255e+09</td>
      <td>1.050023e+09</td>
      <td>1.088055e+09</td>
    </tr>
    <tr>
      <td>St_Num</td>
      <td>7989.0</td>
      <td>9.543122e+01</td>
      <td>1.490730e+02</td>
      <td>0.000000e+00</td>
      <td>8.000000e+00</td>
      <td>2.500000e+01</td>
      <td>1.100000e+02</td>
      <td>9.240000e+02</td>
    </tr>
    <tr>
      <td>Land_Val</td>
      <td>7989.0</td>
      <td>1.389953e+05</td>
      <td>2.516881e+05</td>
      <td>0.000000e+00</td>
      <td>5.530000e+04</td>
      <td>6.930000e+04</td>
      <td>9.860000e+04</td>
      <td>6.395900e+06</td>
    </tr>
    <tr>
      <td>Build_Val</td>
      <td>7989.0</td>
      <td>2.007895e+05</td>
      <td>9.874936e+05</td>
      <td>0.000000e+00</td>
      <td>8.860000e+04</td>
      <td>1.271000e+05</td>
      <td>1.817000e+05</td>
      <td>5.735940e+07</td>
    </tr>
    <tr>
      <td>Total_Val</td>
      <td>7989.0</td>
      <td>3.397848e+05</td>
      <td>1.129524e+06</td>
      <td>1.000000e+02</td>
      <td>1.553000e+05</td>
      <td>2.077000e+05</td>
      <td>2.994000e+05</td>
      <td>6.264060e+07</td>
    </tr>
    <tr>
      <td>Room_Cnt</td>
      <td>6146.0</td>
      <td>7.263261e+00</td>
      <td>3.265074e+00</td>
      <td>0.000000e+00</td>
      <td>5.000000e+00</td>
      <td>6.000000e+00</td>
      <td>8.000000e+00</td>
      <td>3.000000e+01</td>
    </tr>
    <tr>
      <td>Build_SF</td>
      <td>7989.0</td>
      <td>2.571734e+03</td>
      <td>8.609248e+03</td>
      <td>0.000000e+00</td>
      <td>1.056000e+03</td>
      <td>1.544000e+03</td>
      <td>2.312000e+03</td>
      <td>2.625800e+05</td>
    </tr>
    <tr>
      <td>Year_Built</td>
      <td>7989.0</td>
      <td>1.698538e+03</td>
      <td>6.555848e+02</td>
      <td>0.000000e+00</td>
      <td>1.900000e+03</td>
      <td>1.950000e+03</td>
      <td>1.984000e+03</td>
      <td>2.018000e+03</td>
    </tr>
    <tr>
      <td>Land_Area</td>
      <td>7989.0</td>
      <td>1.406506e+05</td>
      <td>8.794815e+05</td>
      <td>0.000000e+00</td>
      <td>7.841000e+03</td>
      <td>1.524600e+04</td>
      <td>4.356000e+04</td>
      <td>5.732496e+07</td>
    </tr>
    <tr>
      <td>Bath_Cnt</td>
      <td>5795.0</td>
      <td>1.894737e+00</td>
      <td>8.811677e-01</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>2.000000e+00</td>
      <td>2.500000e+00</td>
      <td>7.500000e+00</td>
    </tr>
    <tr>
      <td>Exempt_Val</td>
      <td>7989.0</td>
      <td>8.746264e+03</td>
      <td>1.035316e+04</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+04</td>
      <td>3.200000e+04</td>
    </tr>
    <tr>
      <td>Bed_Cnt</td>
      <td>6365.0</td>
      <td>3.422781e+00</td>
      <td>1.406263e+00</td>
      <td>0.000000e+00</td>
      <td>3.000000e+00</td>
      <td>3.000000e+00</td>
      <td>4.000000e+00</td>
      <td>9.000000e+00</td>
    </tr>
    <tr>
      <td>Tax_Total</td>
      <td>7989.0</td>
      <td>3.310385e+05</td>
      <td>1.130341e+06</td>
      <td>0.000000e+00</td>
      <td>1.434000e+05</td>
      <td>1.954000e+05</td>
      <td>2.873000e+05</td>
      <td>6.264060e+07</td>
    </tr>
  </tbody>
</table>
</div>



**Notice** Tax_Total max is two powers higher than at least 75% of the other properties and the standard deviation is a power bigger than the mean. There are some large size differences as well. To ensure that each of these parcels are being fairly compared creating a value by a shared unit of space likely should be done.


```python
cleanedtax.sum()
```




    Map                                                      287179
    Lot                                                      568442
    SubLot                                                     6815
    Polytype      parcelparcelparcelparcelparcelparcelparcelparc...
    LinkID        2-14-110-27-13-14-326-36-141-1392-9-310-2822-3...
    Hist_Dist                                                   194
    Pine_Tr_Zn                                                  155
    Lot_Sub       14-127-114-336-11399-32837-272210180-114092-13...
    FD            5Wood44CowI18BGoos115131113114518B332345544144...
    X10D_ID                                           8276747448815
    OBJECTID_1                                             32197620
    Vision_PID                                            126898600
    GISID         2-14-110-27-13-14-326-36-141-1392-9-310-2822-3...
    F10D_ID                                           8270696061330
    St_Num                                                   762400
    Street        ALFRED ST (OFF OF)WOOD ISLANDPROCTOR RDEVANTHI...
    Location      ALFRED ST (OFF OF)2 WOOD ISLANDPROCTOR RD0 EVA...
    Land_Val                                             1110433400
    Build_Val                                            1604107200
    Total_Val                                            2714540600
    Room_Cnt                                                  44640
    Build_SF                                               20545579
    Year_Built                                             13569623
    Land_Area                                            1123657868
    Bath_Cnt                                                  10980
    Exempt_Val                                             69873900
    Bed_Cnt                                                   21786
    Tax_Total                                            2644666700
    LU_CODE       9035132071229035132013209035132042301320132044...
    dtype: object



### Verify Assumptions

The goal is to come up with ways to increase total property value so it generates similar tax revenues. 

This means verifying:
- Effective Millrate
- Garage Valuations
- Establish Base Variable From Data Set


#### Effective Millrate

The effective millrate is the average rate total property of the city is taxed. This should be found on the city's website.

- FY 2020: `$19.98`
- FY 2019: `$20.07`

> "At their July 16 meeting, the Biddeford City Council set the tax rate for Fiscal Year 2020 at `$19.98` per `$1,000` in taxable value. At this tax rate, the owner of the median single-family home valued at `$227,100` can expect to pay `$4,145.85` in property taxes this year when factoring in the Homestead Exemption."

Sources: https://www.biddefordmaine.org/DocumentCenter/View/5136/072219-FY2020-Tax-Rate-Press-Release?bidId=



```python
millrate = 19.98 / 1000
example_home = 227100
tax_value_pre_homestead = example_home * millrate
print(tax_value_pre_homestead)
```

    4537.4580000000005


**Note:** This is not a PhD disseratation so there is a desire to ignore exemptions but looks like median home owners qualify for roughly a 10% rebate or savings though means not fully understood by researcher

 #### Garage Valuations
 
 The values given by the city were in today dollars with total revenue benefits 10 and 30 years out. These values include parking revenues collected and assume that the garage will make nearby properties worth more as welll as the underlaying parcel(s) it will sit on. 
 
 To figure out the total property value increases that hypothecally can match the city's predictions. The research assumes: 
  - a constant effective tax rate
  - zero appreciation or deprecation of assets
  - rest of the city's value remains constant unless otherwise stated
  
  The steps to figure out this critera is to find the 10 and 30 year revenue averages and divide them by the FY 2020 effective mill rate. This work will produce the valuation increase(s) needed in the city to match their projections for 10 and 30 years post construction. The 10 year and 30 year needed value increases will likely be different because assumingly their financial models are trying to picture life more accurately than this research is attempting.
 
 **Predicted Tax Revenues:**
- 10 Years: `$16,407,604`
- 30 Years: `$39,772,744`
 
 Time to test if these values grow in a linear fashion


```python
garage_10_year_total_rev = 16407604
garage_30_year_total_rev = 39772744

# 10 year slope check
average_10_year = garage_10_year_total_rev / 10

# 30 year slope check
average_30_year = garage_30_year_total_rev / 30

print(average_10_year)
```

    1640760.4



```python
print(average_30_year)
```

    1325758.1333333333


The 30 year average is less than the 10 year average benefit. 

This means for this research there are two seperate bars that proposals can choose to meet. The 10 year average and, or the 30 year average.

Time to figure out what are the corresponding tax base increases


`y = mx + b`

b is assumed to be 0. y is the year average while x is the millrate. m is the corresponding value increase.

which means:

`y/x = m`


```python
# 10 years prediction
value_increase_10_years = average_10_year / millrate

print(value_increase_10_years)
```

    82120140.14014013



```python
# 30 years prediction
value_increase_30_years = average_30_year / millrate

print(value_increase_30_years)
```

    66354260.92759425



```python
sum_current_propert_val = biddeford_parcels['Total_Val'].sum()

total_percent_increase_30_years = (value_increase_30_years / sum_current_propert_val) * 100 
total_percent_increase_10_years = (value_increase_10_years / sum_current_propert_val) * 100 

print(total_percent_increase_30_years, total_percent_increase_10_years)

```

    2.4444011236226952 3.025194765557757


In overall property value increase to the tax base, proposals need to consisently maintain these valuations:

- **30 Years:**`$66,354,300`
- **10 Years:**`$82,120,100` 

which equates to roughly 2.5% for 30 years and 3.1% total increases of property valuations for Biddeford 

#### Establish Base Variable From Data Set

`Tax_Total` and `Total_Val` appear to be the only candidates as the key variable in these data. Doing a quick analysis of their relationship to one another will guide the focus of this research.

If: `Tax_Total  =  Total_Val - Exempt_Val`
The research will use the `Total_Val` as the base variable unless further analysis proves otherwise because this research is assuming zero exemptions will be applicible to new developments and exemptions appear to be rewarded post assessment. 

In seperate research, the `Total_Val` variable was confirmed to be the number that is quoted to property owners. This was done by:
- randomly sampling 5 samples from the set
- taking their location data
- inserting it into the city's property assessment web app: http://gis.vgsi.com/biddefordme/Search.aspx
- comparing search results to specified sample



```python
summation_total_val = 2714540600
summation_tax_total = 2644666700
summation_exempt_val = 69873900

value_less_exemptions = summation_total_val - summation_exempt_val
print(value_less_exemptions)
```

    2644666700


The if case was proven true which means until further notice the main target will be increasing `Total_Val`

## Visualize and Ananlyze Data

We have our general target selected, critera set and effective tax rate defined. The data set is cleaned and there is a base understanding of it. It's time to dig in, visulalize this data so real analysis and proposal generation can happen.

This will happen in two parts. The first will be heat map generation to gain a visual understanding of the city's "portfolio" relative to its main limiting factor - land. The second part will be revolve around sorting and possibly creating graphs.



```python

```

#### Biddeford: Total Property Value by Parcel


```python
biddeford_parcels.plot(figsize=(110,20), column='Total_Val', scheme='QUANTILES', k=40, cmap='OrRd', legend=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f3e5f3f5f10>




![png](output_49_1.png)


This graph highlights the total value of each parcel of land in the city with the darker the color more value the city assigns it. If the city did not have limited amount of land it would be tempting to stop the heat map generation here and go into the graph creation to find the high value properties details.

Biddeford does have a limited amount of natural resources so this visualistion should be refined (think of it like a version of scaling used in assocative analysis) and show valuation data denominated in an uniform unit that's critical to this data set - square feet in this case.

#### Biddeford: Property Value by Square Foot per Parcel


```python
# https://www.naahq.org/news-publications/us-apartment-sizes-shrinking
# According to the National Apartment Association 504 sq feet is now the average size of a studio
# This is down from 614 in mid-2000's so going to use the 504 as bare mininum of Land Area 
# while filtering

min_land_area = 504

biddeford_parcels = biddeford_parcels[(biddeford_parcels['Land_Area'] > min_land_area)]
biddeford_parcels['Value_SQ_Foot'] = biddeford_parcels['Total_Val'] / biddeford_parcels['Land_Area']
biddeford_parcels.sort_values('Value_SQ_Foot').tail(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Map</th>
      <th>Lot</th>
      <th>SubLot</th>
      <th>Polytype</th>
      <th>LinkID</th>
      <th>Ward</th>
      <th>SenateDist</th>
      <th>HouseDist</th>
      <th>MeSPC_Loc</th>
      <th>Updated</th>
      <th>...</th>
      <th>Land_Area</th>
      <th>Build_Styl</th>
      <th>Bath_Cnt</th>
      <th>Exempt_Val</th>
      <th>Bed_Cnt</th>
      <th>Tax_Total_</th>
      <th>LU_CODE</th>
      <th>LU_DESC</th>
      <th>geometry</th>
      <th>Value_SQ_Foot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>6032</td>
      <td>59</td>
      <td>25</td>
      <td>0</td>
      <td>parcel</td>
      <td>59-25</td>
      <td>1</td>
      <td>33</td>
      <td>9</td>
      <td>0</td>
      <td>None</td>
      <td>...</td>
      <td>1307.0</td>
      <td>Condominium</td>
      <td>1</td>
      <td>0.0</td>
      <td>2</td>
      <td>315500.0</td>
      <td>1020</td>
      <td>CONDO  MDL-05</td>
      <td>POLYGON ((2903143.495 223573.782, 2903002.193 ...</td>
      <td>241.392502</td>
    </tr>
    <tr>
      <td>6315</td>
      <td>62</td>
      <td>4</td>
      <td>1</td>
      <td>parcel</td>
      <td>62-4-1</td>
      <td>1</td>
      <td>33</td>
      <td>9</td>
      <td>0</td>
      <td>None</td>
      <td>...</td>
      <td>3920.0</td>
      <td>Modern/Contemp</td>
      <td>2.5</td>
      <td>0.0</td>
      <td>3</td>
      <td>1138900.0</td>
      <td>1012</td>
      <td>OCN FT  MDL-01</td>
      <td>POLYGON ((2900825.239 220875.849, 2900783.402 ...</td>
      <td>290.535714</td>
    </tr>
    <tr>
      <td>6501</td>
      <td>64</td>
      <td>49</td>
      <td>0</td>
      <td>parcel</td>
      <td>64-49</td>
      <td>1</td>
      <td>33</td>
      <td>9</td>
      <td>0</td>
      <td>2008-02-12</td>
      <td>...</td>
      <td>1742.0</td>
      <td>Condominium</td>
      <td>1.5</td>
      <td>0.0</td>
      <td>3</td>
      <td>547500.0</td>
      <td>1020</td>
      <td>CONDO  MDL-05</td>
      <td>POLYGON ((2897335.060 217365.365, 2897289.714 ...</td>
      <td>314.293915</td>
    </tr>
    <tr>
      <td>6502</td>
      <td>64</td>
      <td>49</td>
      <td>0</td>
      <td>parcel</td>
      <td>64-49</td>
      <td>1</td>
      <td>33</td>
      <td>9</td>
      <td>0</td>
      <td>2008-02-12</td>
      <td>...</td>
      <td>1742.0</td>
      <td>Condominium</td>
      <td>1</td>
      <td>20000.0</td>
      <td>3</td>
      <td>531400.0</td>
      <td>1020</td>
      <td>CONDO  MDL-05</td>
      <td>POLYGON ((2897335.060 217365.365, 2897289.714 ...</td>
      <td>316.532721</td>
    </tr>
    <tr>
      <td>4638</td>
      <td>40</td>
      <td>55</td>
      <td>1</td>
      <td>parcel</td>
      <td>40-55-1</td>
      <td>5</td>
      <td>33</td>
      <td>12</td>
      <td>0</td>
      <td>None</td>
      <td>...</td>
      <td>2178.0</td>
      <td>Vacant Land</td>
      <td>None</td>
      <td>0.0</td>
      <td>None</td>
      <td>1064100.0</td>
      <td>4420</td>
      <td>IND LD UD  MDL-00</td>
      <td>POLYGON ((2876679.682 242114.326, 2876677.243 ...</td>
      <td>488.567493</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 121 columns</p>
</div>




```python
biddeford_parcels.plot(figsize=(200,20), column='Value_SQ_Foot', scheme='QUANTILES', k=40, cmap='OrRd', legend=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f3e5f193690>




![png](output_53_1.png)



The other dark red along the river by the coordinates 2890000, 2300000 is UNE's main campus.
<img src="une-hills-beach-sq-ft.jpg" />


#### Clear Take Aways
- Roads make the immediate area more valuable per unit square foot of land. 
- Downtown, the summer communities, UNE's main campus and a couple buildings that are likely the middle and high school are very vaulable sections of town when viewed in sq foot terms.
- There are parcels in all the above mention areas where said current development pattern could expand into
- Having a waterview alone is not a very strong valuation attribute when viewed in value per sq foot terms


#### Locations for Further Analysis
What likely deserves more attention are:
- The effects of growing out the high performing by sq foot areas 

and

- The parcels that are high value in total but relatively low when priced per sq foot


**Downtown**
<img src="down-town-total-parcel-value.jpg" alt="total valuation" />
<img src="down-town-sq-value.jpg" alt="sq foot valuation" />

**Biddeford Pool**

<img src="biddeford-pool-property-total-parcel-value.jpg" alt="total valuation" />

<img src="biddeford-pool-property-value-sq-foot.jpg" alt="sq foot valuation" />

**Timber Point**

Total Valuation
<img src="timber-granit-point-total-parcel-value.jpg" alt="total valuation" />

Square Foot Valution 
<img src="timber-granit-point-value-sq-foot.jpg" alt="sq foot valuation" />


It is now time to dive into the actual numbers and locations to figure out the logisitics of actual proposals. This will be done by first creating filtered view of the `cleanedtax` datframe that includes these columns:

- Location
- Total_Val
- Tax_Total
- Land_Area

From there, more filtering will take place but it will be in regards to records instead of columns and it will be centered around `Land_Area` being greater than 20 sq feet. The 20 sq feet conditional is somewhat abritarty because the anlysis is looking to avoid parcel data where the `Land_Area` is somehow 0 or 1. 

Creating this view is important because a new `Sq_Foot_Val` column can be created (by dividing the `Total_Val` by `Land_Area` columns) and using this new variable the view can be sorted to find the actual properties and their corresponding data that are visibile on the heat maps.


```python
# want to create a new column that tracks differnce between sq foot value and total property value

sq_foot_columns = ['Location', 'Total_Val', 'Tax_Total', 'Land_Area', 'Street',  'St_Num']
sq_foot_df = cleanedtax[sq_foot_columns]
sq_foot_df = sq_foot_df[sq_foot_df['Land_Area'] > min_land_area]
sq_foot_df['Sq_Foot_Val'] = sq_foot_df['Total_Val'] / sq_foot_df['Land_Area']

```


```python
sorted_sq_foot_df = sq_foot_df.sort_values('Sq_Foot_Val')

sorted_sq_foot_df.tail(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Location</th>
      <th>Total_Val</th>
      <th>Tax_Total</th>
      <th>Land_Area</th>
      <th>Street</th>
      <th>St_Num</th>
      <th>Sq_Foot_Val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>4572</td>
      <td>47 MAIN ST</td>
      <td>162100</td>
      <td>162100</td>
      <td>871</td>
      <td>MAIN ST</td>
      <td>47</td>
      <td>186.107922</td>
    </tr>
    <tr>
      <td>6051</td>
      <td>7 BAYVIEW AVE</td>
      <td>324500</td>
      <td>324500</td>
      <td>1742</td>
      <td>BAYVIEW AVE</td>
      <td>7</td>
      <td>186.280138</td>
    </tr>
    <tr>
      <td>6031</td>
      <td>9 LESTER B ORCUTT BLVD U-8</td>
      <td>247800</td>
      <td>247800</td>
      <td>1307</td>
      <td>LESTER B ORCUTT BLVD U-8</td>
      <td>9</td>
      <td>189.594491</td>
    </tr>
    <tr>
      <td>6175</td>
      <td>44 OCEAN AVE</td>
      <td>1008300</td>
      <td>1008300</td>
      <td>5227</td>
      <td>OCEAN AVE</td>
      <td>44</td>
      <td>192.902238</td>
    </tr>
    <tr>
      <td>6169</td>
      <td>23 SEVENTH ST</td>
      <td>589500</td>
      <td>589500</td>
      <td>3049</td>
      <td>SEVENTH ST</td>
      <td>23</td>
      <td>193.342079</td>
    </tr>
    <tr>
      <td>5775</td>
      <td>193 HILLS BEACH RD</td>
      <td>1059500</td>
      <td>1059500</td>
      <td>5445</td>
      <td>HILLS BEACH RD</td>
      <td>193</td>
      <td>194.582185</td>
    </tr>
    <tr>
      <td>5785</td>
      <td>3 SURF AVE</td>
      <td>311700</td>
      <td>311700</td>
      <td>1600</td>
      <td>SURF AVE</td>
      <td>3</td>
      <td>194.812500</td>
    </tr>
    <tr>
      <td>5845</td>
      <td>6 HANSONS LN</td>
      <td>439200</td>
      <td>439200</td>
      <td>2240</td>
      <td>HANSONS LN</td>
      <td>6</td>
      <td>196.071429</td>
    </tr>
    <tr>
      <td>6311</td>
      <td>50 MILE STRETCH RD</td>
      <td>791200</td>
      <td>791200</td>
      <td>3920</td>
      <td>MILE STRETCH RD</td>
      <td>50</td>
      <td>201.836735</td>
    </tr>
    <tr>
      <td>3774</td>
      <td>254 MAIN ST</td>
      <td>1961400</td>
      <td>1961400</td>
      <td>9583</td>
      <td>MAIN ST</td>
      <td>254</td>
      <td>204.674945</td>
    </tr>
    <tr>
      <td>6178</td>
      <td>38 OCEAN AVE</td>
      <td>1825600</td>
      <td>1825600</td>
      <td>8712</td>
      <td>OCEAN AVE</td>
      <td>38</td>
      <td>209.550046</td>
    </tr>
    <tr>
      <td>6954</td>
      <td>75 SACO FALLS WAY</td>
      <td>7761700</td>
      <td>7761700</td>
      <td>36898</td>
      <td>SACO FALLS WAY</td>
      <td>75</td>
      <td>210.355575</td>
    </tr>
    <tr>
      <td>4573</td>
      <td>49 MAIN ST</td>
      <td>187600</td>
      <td>187600</td>
      <td>871</td>
      <td>MAIN ST</td>
      <td>49</td>
      <td>215.384615</td>
    </tr>
    <tr>
      <td>4051</td>
      <td>145 MAIN ST</td>
      <td>754400</td>
      <td>754400</td>
      <td>3485</td>
      <td>MAIN ST</td>
      <td>145</td>
      <td>216.470588</td>
    </tr>
    <tr>
      <td>6212</td>
      <td>60 MILE STRETCH RD</td>
      <td>1204000</td>
      <td>1204000</td>
      <td>5227</td>
      <td>MILE STRETCH RD</td>
      <td>60</td>
      <td>230.342453</td>
    </tr>
    <tr>
      <td>6032</td>
      <td>9 LESTER B ORCUTT BLVD U-9</td>
      <td>315500</td>
      <td>315500</td>
      <td>1307</td>
      <td>LESTER B ORCUTT BLVD U-9</td>
      <td>9</td>
      <td>241.392502</td>
    </tr>
    <tr>
      <td>6315</td>
      <td>48 MILE STRETCH RD</td>
      <td>1138900</td>
      <td>1138900</td>
      <td>3920</td>
      <td>MILE STRETCH RD</td>
      <td>48</td>
      <td>290.535714</td>
    </tr>
    <tr>
      <td>6501</td>
      <td>81 FORTUNES ROCKS RD</td>
      <td>547500</td>
      <td>547500</td>
      <td>1742</td>
      <td>FORTUNES ROCKS RD</td>
      <td>81</td>
      <td>314.293915</td>
    </tr>
    <tr>
      <td>6502</td>
      <td>79 FORTUNES ROCKS RD</td>
      <td>551400</td>
      <td>531400</td>
      <td>1742</td>
      <td>FORTUNES ROCKS RD</td>
      <td>79</td>
      <td>316.532721</td>
    </tr>
    <tr>
      <td>4638</td>
      <td>GOOCH ST (OFF OF)</td>
      <td>1064100</td>
      <td>1064100</td>
      <td>2178</td>
      <td>GOOCH ST (OFF OF)</td>
      <td>0</td>
      <td>488.567493</td>
    </tr>
  </tbody>
</table>
</div>



A good portion of these properties are likely going to be underwater in the near future and this study is looking to expand the tax base into the future so the next step is to filter for the properties that likely will still be around in the future. This research is depending on the domain knowledge on likely underwater status in creating the filtering list. This list consists of all properties on:

- FORTUNES ROCKS RD
- MILE STRETCH RD
- OCEAN AVE

*This list is no where near comprehensive

This filters the original top 20 list to these locations:

- GOOCH ST (OFF OF)
- 9 LESTER B ORCUTT BLVD U-9
- 145 MAIN ST
- 49 MAIN ST
- 75 SACO FALLS WAY
- 6 HANSONS LN
- 193 HILLS BEACH RD
- 23 SEVENTH ST
- 9 LESTER B ORCUTT BLVD U-8
- 7 BAYVIEW AVE
- 47 MAIN ST

The Hills Beach Rd. and the Seventh St. are likely going to be underwater in the pending future as well which leaves the list:

- GOOCH ST (OFF OF)
- 9 LESTER B ORCUTT BLVD U-9
- 145 MAIN ST
- 49 MAIN ST
- 75 SACO FALLS WAY
- 6 HANSONS LN
- 9 LESTER B ORCUTT BLVD U-8
- 7 BAYVIEW AVE
- 47 MAIN ST

Something to note is that Gooch Street has a handful of borderline abandoned buildings and it's very close to the condo/ apartment complex that is 75 Saco Falls Way.


##### Aerial View of Gooch and Saco Falls Way
<img src="ariel-saco-falls-gooch-st.jpg" />

#### Street View of Gooch Street
<img src="gooch-st-street-view.jpg" />


#### Street View of Saco Falls Way
<img src="75-saco-falls-way-street-view.jpg" />

Visibily these lots look quiet similiar with the main difference between them is one has been redeveloped to be offices and apartments while the other spot has not.


```python
non_flood_valuable_sq_foot_properties = [
'GOOCH ST (OFF OF)',
# '9 LESTER B ORCUTT BLVD U-9',
#'145 MAIN ST',
#'49 MAIN ST',
'75 SACO FALLS WAY',
# '6 HANSONS LN',
# '9 LESTER B ORCUTT BLVD U-8',
# '7 BAYVIEW AVE',
# '47 MAIN ST'
]

non_flood_valuable_sq_foot_properties_data = cleanedtax[cleanedtax['Location'].isin(non_flood_valuable_sq_foot_properties)]
non_flood_valuable_sq_foot_properties_data.transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>4639</th>
      <th>4638</th>
      <th>6954</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Map</td>
      <td>40</td>
      <td>40</td>
      <td>71</td>
    </tr>
    <tr>
      <td>Lot</td>
      <td>55</td>
      <td>55</td>
      <td>9</td>
    </tr>
    <tr>
      <td>SubLot</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <td>Polytype</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
    </tr>
    <tr>
      <td>LinkID</td>
      <td>40-55-2</td>
      <td>40-55-1</td>
      <td>71-9-2</td>
    </tr>
    <tr>
      <td>Hist_Dist</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>Pine_Tr_Zn</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>Lot_Sub</td>
      <td>55-2</td>
      <td>55-1</td>
      <td>9-2</td>
    </tr>
    <tr>
      <td>FD</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <td>X10D_ID</td>
      <td>1040055002</td>
      <td>1040055001</td>
      <td>1071009002</td>
    </tr>
    <tr>
      <td>OBJECTID_1</td>
      <td>5485</td>
      <td>5484</td>
      <td>7241</td>
    </tr>
    <tr>
      <td>Vision_PID</td>
      <td>4902</td>
      <td>4901</td>
      <td>121166</td>
    </tr>
    <tr>
      <td>GISID</td>
      <td>40-55-2</td>
      <td>40-55-1</td>
      <td>71-9-2</td>
    </tr>
    <tr>
      <td>F10D_ID</td>
      <td>1040055002</td>
      <td>1040055001</td>
      <td>1071009002</td>
    </tr>
    <tr>
      <td>St_Num</td>
      <td>0</td>
      <td>0</td>
      <td>75</td>
    </tr>
    <tr>
      <td>Street</td>
      <td>GOOCH ST (OFF OF)</td>
      <td>GOOCH ST (OFF OF)</td>
      <td>SACO FALLS WAY</td>
    </tr>
    <tr>
      <td>Location</td>
      <td>GOOCH ST (OFF OF)</td>
      <td>GOOCH ST (OFF OF)</td>
      <td>75 SACO FALLS WAY</td>
    </tr>
    <tr>
      <td>Land_Val</td>
      <td>13100</td>
      <td>23100</td>
      <td>724300</td>
    </tr>
    <tr>
      <td>Build_Val</td>
      <td>0</td>
      <td>1041000</td>
      <td>7037400</td>
    </tr>
    <tr>
      <td>Total_Val</td>
      <td>13100</td>
      <td>1064100</td>
      <td>7761700</td>
    </tr>
    <tr>
      <td>Zone</td>
      <td>MSRD3</td>
      <td>MSRD3</td>
      <td>MSRD3</td>
    </tr>
    <tr>
      <td>Room_Cnt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Build_SF</td>
      <td>0</td>
      <td>0</td>
      <td>75382</td>
    </tr>
    <tr>
      <td>Year_Built</td>
      <td>0</td>
      <td>0</td>
      <td>1900</td>
    </tr>
    <tr>
      <td>Land_Area</td>
      <td>871</td>
      <td>2178</td>
      <td>36898</td>
    </tr>
    <tr>
      <td>Build_Styl</td>
      <td>Vacant Land</td>
      <td>Vacant Land</td>
      <td>Apartments</td>
    </tr>
    <tr>
      <td>Bath_Cnt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Exempt_Val</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Bed_Cnt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Tax_Total</td>
      <td>13100</td>
      <td>1064100</td>
      <td>7761700</td>
    </tr>
    <tr>
      <td>LU_CODE</td>
      <td>4420</td>
      <td>4420</td>
      <td>4000</td>
    </tr>
    <tr>
      <td>LU_DESC4</td>
      <td>IND LD UD  MDL-00</td>
      <td>IND LD UD  MDL-00</td>
      <td>FACTORY</td>
    </tr>
  </tbody>
</table>
</div>




```python
gooch_st_becomes_saco_falls = (7037400 * 2) - 1041000

gooch_st_improvements_10_years_percent = (gooch_st_becomes_saco_falls / value_increase_10_years) * 100
gooch_st_improvements_30_years_percent = (gooch_st_becomes_saco_falls / value_increase_30_years) * 100

print(gooch_st_improvements_30_years_percent, gooch_st_improvements_10_years_percent)
```

    19.64274760625015 15.87162415670198


Doing a rough calculation and assuming that cloning the build value pf 75 Saco Falls is a likely possibility for the two Gooch street populations the new value would cover ~ `20%` and `16%` for the `30 year` and `10 year` benchmarks.


These improvements alone will not suffice. Main Street improvements and the Lester B. Orcutt BLVD parcel(s) look promising.


##### Lester B. Orcutt BLVD

<img src="9-lb-orcutt-blvd-street-view.jpg" />


```python
non_flood_valuable_sq_foot_properties = [
# 'GOOCH ST (OFF OF)',
 '9 LESTER B ORCUTT BLVD U-9',
#'145 MAIN ST',
#'49 MAIN ST',
# '75 SACO FALLS WAY',
# '6 HANSONS LN',
 '9 LESTER B ORCUTT BLVD U-8',
# '7 BAYVIEW AVE',
# '47 MAIN ST'
]

non_flood_valuable_sq_foot_properties_data = cleanedtax[cleanedtax['Location'].isin(non_flood_valuable_sq_foot_properties)]
non_flood_valuable_sq_foot_properties_data.transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>6031</th>
      <th>6032</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Map</td>
      <td>59</td>
      <td>59</td>
    </tr>
    <tr>
      <td>Lot</td>
      <td>25</td>
      <td>25</td>
    </tr>
    <tr>
      <td>SubLot</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Polytype</td>
      <td>parcel</td>
      <td>parcel</td>
    </tr>
    <tr>
      <td>LinkID</td>
      <td>59-25</td>
      <td>59-25</td>
    </tr>
    <tr>
      <td>Hist_Dist</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Pine_Tr_Zn</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Lot_Sub</td>
      <td>25</td>
      <td>25</td>
    </tr>
    <tr>
      <td>FD</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <td>X10D_ID</td>
      <td>1059025000</td>
      <td>1059025000</td>
    </tr>
    <tr>
      <td>OBJECTID_1</td>
      <td>6486</td>
      <td>6487</td>
    </tr>
    <tr>
      <td>Vision_PID</td>
      <td>5907</td>
      <td>5908</td>
    </tr>
    <tr>
      <td>GISID</td>
      <td>59-25</td>
      <td>59-25</td>
    </tr>
    <tr>
      <td>F10D_ID</td>
      <td>1059025008</td>
      <td>1059025009</td>
    </tr>
    <tr>
      <td>St_Num</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <td>Street</td>
      <td>LESTER B ORCUTT BLVD U-8</td>
      <td>LESTER B ORCUTT BLVD U-9</td>
    </tr>
    <tr>
      <td>Location</td>
      <td>9 LESTER B ORCUTT BLVD U-8</td>
      <td>9 LESTER B ORCUTT BLVD U-9</td>
    </tr>
    <tr>
      <td>Land_Val</td>
      <td>195900</td>
      <td>250000</td>
    </tr>
    <tr>
      <td>Build_Val</td>
      <td>51900</td>
      <td>65500</td>
    </tr>
    <tr>
      <td>Total_Val</td>
      <td>247800</td>
      <td>315500</td>
    </tr>
    <tr>
      <td>Zone</td>
      <td>CR</td>
      <td>CR</td>
    </tr>
    <tr>
      <td>Room_Cnt</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <td>Build_SF</td>
      <td>674</td>
      <td>1429</td>
    </tr>
    <tr>
      <td>Year_Built</td>
      <td>1900</td>
      <td>1900</td>
    </tr>
    <tr>
      <td>Land_Area</td>
      <td>1307</td>
      <td>1307</td>
    </tr>
    <tr>
      <td>Build_Styl</td>
      <td>Condominium</td>
      <td>Condominium</td>
    </tr>
    <tr>
      <td>Bath_Cnt</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>Exempt_Val</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Bed_Cnt</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <td>Tax_Total</td>
      <td>247800</td>
      <td>315500</td>
    </tr>
    <tr>
      <td>LU_CODE</td>
      <td>1020</td>
      <td>1020</td>
    </tr>
    <tr>
      <td>LU_DESC4</td>
      <td>CONDO  MDL-05</td>
      <td>CONDO  MDL-05</td>
    </tr>
  </tbody>
</table>
</div>



Quickly looking at the build to land values and the total values these properties on their own are not going to make noticble dents to reaching the garage value add goals.

This analysis probably should also look at: 
- land value as a percent of of total value
- build value per sq foot

This is so time is not spent any further looking at properties where adding or modifying the development on said land would produce negible effects on reaching the garage valuation hurdle.



```python
cleanedtax['Sq_Foot_Val'] = cleanedtax['Total_Val'] / cleanedtax['Land_Area']
cleanedtax['Build_SQ_FT_Val'] = cleanedtax['Build_Val'] / cleanedtax['Build_SF']
cleanedtax['Land_Val_Per'] = cleanedtax['Land_Val'] / cleanedtax['Total_Val']
```

#### Main St. Parcels 

**47 and 49 Main St.**

Notice the left side is an empty mill building while across the street is some of the most productive real estate in the city. 

<img src="main-street-street-view.jpg" />



**145 Main St.***

Notice that the style is similar but renovated, one-two stories taller and next to a small park.

<img src="145-main-st-street-view.jpg" />


```python
non_flood_valuable_sq_foot_properties = [
# 'GOOCH ST (OFF OF)',
# '9 LESTER B ORCUTT BLVD U-9',
'145 MAIN ST',
'49 MAIN ST',
# '75 SACO FALLS WAY',
# '6 HANSONS LN',
# '9 LESTER B ORCUTT BLVD U-8',
# '7 BAYVIEW AVE',
 '47 MAIN ST'
]

non_flood_valuable_sq_foot_properties_data = cleanedtax[cleanedtax['Location'].isin(non_flood_valuable_sq_foot_properties)]
non_flood_valuable_sq_foot_properties_data.transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>4572</th>
      <th>4573</th>
      <th>4051</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Map</td>
      <td>39</td>
      <td>39</td>
      <td>38</td>
    </tr>
    <tr>
      <td>Lot</td>
      <td>74</td>
      <td>75</td>
      <td>370</td>
    </tr>
    <tr>
      <td>SubLot</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Polytype</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
    </tr>
    <tr>
      <td>LinkID</td>
      <td>39-74</td>
      <td>39-75</td>
      <td>38-370</td>
    </tr>
    <tr>
      <td>Hist_Dist</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <td>Pine_Tr_Zn</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Lot_Sub</td>
      <td>74</td>
      <td>75</td>
      <td>370</td>
    </tr>
    <tr>
      <td>FD</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <td>X10D_ID</td>
      <td>1039074000</td>
      <td>1039075000</td>
      <td>1038370000</td>
    </tr>
    <tr>
      <td>OBJECTID_1</td>
      <td>5124</td>
      <td>5125</td>
      <td>4963</td>
    </tr>
    <tr>
      <td>Vision_PID</td>
      <td>4541</td>
      <td>4542</td>
      <td>4378</td>
    </tr>
    <tr>
      <td>GISID</td>
      <td>39-74</td>
      <td>39-75</td>
      <td>38-370</td>
    </tr>
    <tr>
      <td>F10D_ID</td>
      <td>1039074000</td>
      <td>1039075000</td>
      <td>1038370000</td>
    </tr>
    <tr>
      <td>St_Num</td>
      <td>47</td>
      <td>49</td>
      <td>145</td>
    </tr>
    <tr>
      <td>Street</td>
      <td>MAIN ST</td>
      <td>MAIN ST</td>
      <td>MAIN ST</td>
    </tr>
    <tr>
      <td>Location</td>
      <td>47 MAIN ST</td>
      <td>49 MAIN ST</td>
      <td>145 MAIN ST</td>
    </tr>
    <tr>
      <td>Land_Val</td>
      <td>65400</td>
      <td>98000</td>
      <td>237300</td>
    </tr>
    <tr>
      <td>Build_Val</td>
      <td>96700</td>
      <td>89600</td>
      <td>517100</td>
    </tr>
    <tr>
      <td>Total_Val</td>
      <td>162100</td>
      <td>187600</td>
      <td>754400</td>
    </tr>
    <tr>
      <td>Zone</td>
      <td>MSRD1</td>
      <td>MSRD1</td>
      <td>MSRD1</td>
    </tr>
    <tr>
      <td>Room_Cnt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Build_SF</td>
      <td>1539</td>
      <td>1640</td>
      <td>13034</td>
    </tr>
    <tr>
      <td>Year_Built</td>
      <td>1903</td>
      <td>1900</td>
      <td>1900</td>
    </tr>
    <tr>
      <td>Land_Area</td>
      <td>871</td>
      <td>871</td>
      <td>3485</td>
    </tr>
    <tr>
      <td>Build_Styl</td>
      <td>Stores/Apt Com</td>
      <td>Stores/Apt Com</td>
      <td>Stores/Apt Com</td>
    </tr>
    <tr>
      <td>Bath_Cnt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Exempt_Val</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Bed_Cnt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Tax_Total</td>
      <td>162100</td>
      <td>187600</td>
      <td>754400</td>
    </tr>
    <tr>
      <td>LU_CODE</td>
      <td>3220</td>
      <td>3260</td>
      <td>031C</td>
    </tr>
    <tr>
      <td>LU_DESC4</td>
      <td>STORE/SHOP  MDL-94</td>
      <td>REST/CLUBS  MDL-94</td>
      <td>PRI COMM  MDL-94</td>
    </tr>
    <tr>
      <td>Sq_Foot_Val</td>
      <td>186.108</td>
      <td>215.385</td>
      <td>216.471</td>
    </tr>
    <tr>
      <td>Build_SQ_FT_Val</td>
      <td>62.833</td>
      <td>54.6341</td>
      <td>39.6732</td>
    </tr>
    <tr>
      <td>Land_Val_Per</td>
      <td>0.403455</td>
      <td>0.522388</td>
      <td>0.314555</td>
    </tr>
  </tbody>
</table>
</div>




```python
main_street = cleanedtax[(cleanedtax['Street'] == 'MAIN ST') & ((cleanedtax['Location'] != 'MAIN ST'))]
vacant_lots_main_street = cleanedtax[(cleanedtax['Location'] == 'MAIN ST')]
main_street = main_street.sort_values('St_Num')
main_street.transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>6895</th>
      <th>6897</th>
      <th>4677</th>
      <th>6893</th>
      <th>4714</th>
      <th>4715</th>
      <th>4716</th>
      <th>4572</th>
      <th>4573</th>
      <th>4574</th>
      <th>...</th>
      <th>3484</th>
      <th>3483</th>
      <th>3482</th>
      <th>3481</th>
      <th>3509</th>
      <th>2539</th>
      <th>2528</th>
      <th>2562</th>
      <th>2577</th>
      <th>2519</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Map</td>
      <td>71</td>
      <td>71</td>
      <td>41</td>
      <td>71</td>
      <td>41</td>
      <td>41</td>
      <td>41</td>
      <td>39</td>
      <td>39</td>
      <td>39</td>
      <td>...</td>
      <td>36</td>
      <td>36</td>
      <td>36</td>
      <td>36</td>
      <td>36</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
    </tr>
    <tr>
      <td>Lot</td>
      <td>11</td>
      <td>12</td>
      <td>1</td>
      <td>10</td>
      <td>136</td>
      <td>137</td>
      <td>138</td>
      <td>74</td>
      <td>75</td>
      <td>76</td>
      <td>...</td>
      <td>13</td>
      <td>12</td>
      <td>11</td>
      <td>10</td>
      <td>8</td>
      <td>3</td>
      <td>2</td>
      <td>5</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <td>SubLot</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>Polytype</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>...</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
    </tr>
    <tr>
      <td>LinkID</td>
      <td>71-11</td>
      <td>71-12</td>
      <td>41-1</td>
      <td>71-10-1</td>
      <td>41-136</td>
      <td>41-137</td>
      <td>41-138</td>
      <td>39-74</td>
      <td>39-75</td>
      <td>39-76</td>
      <td>...</td>
      <td>36-13</td>
      <td>36-12</td>
      <td>36-11</td>
      <td>36-10</td>
      <td>36-8</td>
      <td>32-3</td>
      <td>32-2</td>
      <td>32-5</td>
      <td>32-6</td>
      <td>32-1-1</td>
    </tr>
    <tr>
      <td>Hist_Dist</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Pine_Tr_Zn</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Lot_Sub</td>
      <td>11</td>
      <td>12</td>
      <td>1</td>
      <td>10-1</td>
      <td>136</td>
      <td>137</td>
      <td>138</td>
      <td>74</td>
      <td>75</td>
      <td>76</td>
      <td>...</td>
      <td>13</td>
      <td>12</td>
      <td>11</td>
      <td>10</td>
      <td>8</td>
      <td>3</td>
      <td>2</td>
      <td>5</td>
      <td>6</td>
      <td>1-1</td>
    </tr>
    <tr>
      <td>FD</td>
      <td>12</td>
      <td>12</td>
      <td>9</td>
      <td>12</td>
      <td>9</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>...</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <td>X10D_ID</td>
      <td>1071011000</td>
      <td>1071012000</td>
      <td>1041001000</td>
      <td>1071010001</td>
      <td>1041136000</td>
      <td>1041137000</td>
      <td>1041138000</td>
      <td>1039074000</td>
      <td>1039075000</td>
      <td>1039076000</td>
      <td>...</td>
      <td>1036013000</td>
      <td>1036012000</td>
      <td>1036011000</td>
      <td>1036010000</td>
      <td>1036008000</td>
      <td>1032003000</td>
      <td>1032002000</td>
      <td>1032005000</td>
      <td>1032006000</td>
      <td>1032001001</td>
    </tr>
    <tr>
      <td>OBJECTID_1</td>
      <td>7246</td>
      <td>7247</td>
      <td>5501</td>
      <td>7245</td>
      <td>5618</td>
      <td>5619</td>
      <td>5620</td>
      <td>5124</td>
      <td>5125</td>
      <td>5126</td>
      <td>...</td>
      <td>4387</td>
      <td>4386</td>
      <td>4385</td>
      <td>4384</td>
      <td>4383</td>
      <td>3517</td>
      <td>3516</td>
      <td>3519</td>
      <td>3520</td>
      <td>3515</td>
    </tr>
    <tr>
      <td>Vision_PID</td>
      <td>6642</td>
      <td>6643</td>
      <td>4918</td>
      <td>120846</td>
      <td>5051</td>
      <td>5052</td>
      <td>5053</td>
      <td>4541</td>
      <td>4542</td>
      <td>4543</td>
      <td>...</td>
      <td>3801</td>
      <td>3800</td>
      <td>3799</td>
      <td>3798</td>
      <td>3797</td>
      <td>2946</td>
      <td>2945</td>
      <td>2948</td>
      <td>2949</td>
      <td>2943</td>
    </tr>
    <tr>
      <td>GISID</td>
      <td>71-11</td>
      <td>71-12</td>
      <td>41-1</td>
      <td>71-10-1</td>
      <td>41-136</td>
      <td>41-137</td>
      <td>41-138</td>
      <td>39-74</td>
      <td>39-75</td>
      <td>39-76</td>
      <td>...</td>
      <td>36-13</td>
      <td>36-12</td>
      <td>36-11</td>
      <td>36-10</td>
      <td>36-8</td>
      <td>32-3</td>
      <td>32-2</td>
      <td>32-5</td>
      <td>32-6</td>
      <td>32-1-1</td>
    </tr>
    <tr>
      <td>F10D_ID</td>
      <td>1071011000</td>
      <td>1071012000</td>
      <td>1041001000</td>
      <td>1071010001</td>
      <td>1041136000</td>
      <td>1041137000</td>
      <td>1041138000</td>
      <td>1039074000</td>
      <td>1039075000</td>
      <td>1039076000</td>
      <td>...</td>
      <td>1036013000</td>
      <td>1036012000</td>
      <td>1036011000</td>
      <td>1036010000</td>
      <td>1036008000</td>
      <td>1032003000</td>
      <td>1032002000</td>
      <td>1032005000</td>
      <td>1032006000</td>
      <td>1032001001</td>
    </tr>
    <tr>
      <td>St_Num</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>29</td>
      <td>35</td>
      <td>41</td>
      <td>47</td>
      <td>49</td>
      <td>53</td>
      <td>...</td>
      <td>518</td>
      <td>522</td>
      <td>530</td>
      <td>532</td>
      <td>533</td>
      <td>541</td>
      <td>550</td>
      <td>601</td>
      <td>603</td>
      <td>630</td>
    </tr>
    <tr>
      <td>Street</td>
      <td>MAIN ST</td>
      <td>MAIN ST</td>
      <td>MAIN ST</td>
      <td>MAIN ST</td>
      <td>MAIN ST</td>
      <td>MAIN ST</td>
      <td>MAIN ST</td>
      <td>MAIN ST</td>
      <td>MAIN ST</td>
      <td>MAIN ST</td>
      <td>...</td>
      <td>MAIN ST</td>
      <td>MAIN ST</td>
      <td>MAIN ST</td>
      <td>MAIN ST</td>
      <td>MAIN ST</td>
      <td>MAIN ST</td>
      <td>MAIN ST</td>
      <td>MAIN ST</td>
      <td>MAIN ST</td>
      <td>MAIN ST</td>
    </tr>
    <tr>
      <td>Location</td>
      <td>2 MAIN ST</td>
      <td>2 MAIN ST</td>
      <td>3 MAIN ST</td>
      <td>6 MAIN ST</td>
      <td>29 MAIN ST</td>
      <td>35 MAIN ST</td>
      <td>41 MAIN ST</td>
      <td>47 MAIN ST</td>
      <td>49 MAIN ST</td>
      <td>53 MAIN ST</td>
      <td>...</td>
      <td>518 MAIN ST</td>
      <td>522 MAIN ST</td>
      <td>530 MAIN ST</td>
      <td>532 MAIN ST</td>
      <td>533 MAIN ST</td>
      <td>541 MAIN ST</td>
      <td>550 MAIN ST</td>
      <td>601 MAIN ST</td>
      <td>603 MAIN ST</td>
      <td>630 MAIN ST</td>
    </tr>
    <tr>
      <td>Land_Val</td>
      <td>647800</td>
      <td>116500</td>
      <td>118500</td>
      <td>123700</td>
      <td>124000</td>
      <td>124800</td>
      <td>140000</td>
      <td>65400</td>
      <td>98000</td>
      <td>98000</td>
      <td>...</td>
      <td>56500</td>
      <td>78800</td>
      <td>120200</td>
      <td>67500</td>
      <td>58900</td>
      <td>66000</td>
      <td>692900</td>
      <td>71300</td>
      <td>59300</td>
      <td>83200</td>
    </tr>
    <tr>
      <td>Build_Val</td>
      <td>16467500</td>
      <td>88900</td>
      <td>2714000</td>
      <td>159700</td>
      <td>138600</td>
      <td>197400</td>
      <td>160300</td>
      <td>96700</td>
      <td>89600</td>
      <td>80600</td>
      <td>...</td>
      <td>78400</td>
      <td>127500</td>
      <td>318000</td>
      <td>109000</td>
      <td>130200</td>
      <td>70300</td>
      <td>247800</td>
      <td>100600</td>
      <td>95500</td>
      <td>189800</td>
    </tr>
    <tr>
      <td>Total_Val</td>
      <td>17115300</td>
      <td>205400</td>
      <td>2832500</td>
      <td>283400</td>
      <td>262600</td>
      <td>322200</td>
      <td>300300</td>
      <td>162100</td>
      <td>187600</td>
      <td>178600</td>
      <td>...</td>
      <td>134900</td>
      <td>206300</td>
      <td>438200</td>
      <td>176500</td>
      <td>189100</td>
      <td>136300</td>
      <td>940700</td>
      <td>171900</td>
      <td>154800</td>
      <td>273000</td>
    </tr>
    <tr>
      <td>Zone</td>
      <td>MSRD3</td>
      <td>MSRD3</td>
      <td>W2</td>
      <td>MSRD3</td>
      <td>MSRD1</td>
      <td>MSRD1</td>
      <td>MSRD1</td>
      <td>MSRD1</td>
      <td>MSRD1</td>
      <td>MSRD1</td>
      <td>...</td>
      <td>R1A</td>
      <td>R2</td>
      <td>R1A</td>
      <td>R1A</td>
      <td>R1A</td>
      <td>R1A</td>
      <td>R1A</td>
      <td>R3</td>
      <td>R3</td>
      <td>R3</td>
    </tr>
    <tr>
      <td>Room_Cnt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>7</td>
      <td>11</td>
      <td>NaN</td>
      <td>7</td>
      <td>5</td>
      <td>4</td>
      <td>NaN</td>
      <td>6</td>
      <td>5</td>
      <td>15</td>
    </tr>
    <tr>
      <td>Build_SF</td>
      <td>105296</td>
      <td>12366</td>
      <td>0</td>
      <td>25200</td>
      <td>7252</td>
      <td>9650</td>
      <td>12992</td>
      <td>1539</td>
      <td>1640</td>
      <td>2550</td>
      <td>...</td>
      <td>2135</td>
      <td>2997</td>
      <td>7034</td>
      <td>1365</td>
      <td>1264</td>
      <td>780</td>
      <td>2560</td>
      <td>994</td>
      <td>1097</td>
      <td>3432</td>
    </tr>
    <tr>
      <td>Year_Built</td>
      <td>1848</td>
      <td>1916</td>
      <td>0</td>
      <td>1890</td>
      <td>1930</td>
      <td>1900</td>
      <td>1920</td>
      <td>1903</td>
      <td>1900</td>
      <td>1900</td>
      <td>...</td>
      <td>1900</td>
      <td>1900</td>
      <td>1982</td>
      <td>1875</td>
      <td>1919</td>
      <td>1940</td>
      <td>1971</td>
      <td>1948</td>
      <td>1950</td>
      <td>1900</td>
    </tr>
    <tr>
      <td>Land_Area</td>
      <td>191666</td>
      <td>38333</td>
      <td>60113</td>
      <td>6333</td>
      <td>6534</td>
      <td>6970</td>
      <td>22651</td>
      <td>871</td>
      <td>871</td>
      <td>1307</td>
      <td>...</td>
      <td>26136</td>
      <td>21780</td>
      <td>34848</td>
      <td>16117</td>
      <td>10019</td>
      <td>18295</td>
      <td>3218648</td>
      <td>20909</td>
      <td>8276</td>
      <td>108029</td>
    </tr>
    <tr>
      <td>Build_Styl</td>
      <td>Commercial</td>
      <td>Mill Building</td>
      <td>NaN</td>
      <td>Warehouse</td>
      <td>Stores/Apt Com</td>
      <td>Neighborhood Center</td>
      <td>Warehouse</td>
      <td>Stores/Apt Com</td>
      <td>Stores/Apt Com</td>
      <td>Stores/Apt Com</td>
      <td>...</td>
      <td>Conventional</td>
      <td>3 Unit</td>
      <td>Apartments</td>
      <td>Conventional</td>
      <td>Conventional</td>
      <td>Conventional</td>
      <td>Clubs/Lodges</td>
      <td>Conventional</td>
      <td>Ranch</td>
      <td>3 Unit</td>
    </tr>
    <tr>
      <td>Bath_Cnt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>2</td>
      <td>3</td>
      <td>NaN</td>
      <td>1.5</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <td>Exempt_Val</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>20000</td>
      <td>20000</td>
      <td>20000</td>
      <td>0</td>
      <td>20000</td>
      <td>0</td>
      <td>20000</td>
    </tr>
    <tr>
      <td>Bed_Cnt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>4</td>
      <td>4</td>
      <td>NaN</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <td>Tax_Total</td>
      <td>17115300</td>
      <td>205400</td>
      <td>2832500</td>
      <td>283400</td>
      <td>262600</td>
      <td>322200</td>
      <td>300300</td>
      <td>162100</td>
      <td>187600</td>
      <td>178600</td>
      <td>...</td>
      <td>134900</td>
      <td>206300</td>
      <td>438200</td>
      <td>156500</td>
      <td>169100</td>
      <td>116300</td>
      <td>940700</td>
      <td>151900</td>
      <td>154800</td>
      <td>253000</td>
    </tr>
    <tr>
      <td>LU_CODE</td>
      <td>4000</td>
      <td>4000</td>
      <td>9035</td>
      <td>4022</td>
      <td>3220</td>
      <td>031C</td>
      <td>3160</td>
      <td>3220</td>
      <td>3260</td>
      <td>3220</td>
      <td>...</td>
      <td>903R</td>
      <td>1050</td>
      <td>1114</td>
      <td>1013</td>
      <td>1010</td>
      <td>1010</td>
      <td>903C</td>
      <td>1010</td>
      <td>1010</td>
      <td>1050</td>
    </tr>
    <tr>
      <td>LU_DESC4</td>
      <td>FACTORY</td>
      <td>FACTORY</td>
      <td>TOWN-PROP  MDL-00</td>
      <td>IND BLDG  MDL-96</td>
      <td>STORE/SHOP  MDL-94</td>
      <td>PRI COMM  MDL-94</td>
      <td>COMM WHSE  MDL-94</td>
      <td>STORE/SHOP  MDL-94</td>
      <td>REST/CLUBS  MDL-94</td>
      <td>STORE/SHOP  MDL-94</td>
      <td>...</td>
      <td>TOWN-PROP  MDL-01</td>
      <td>THREE FAM  MDL-03</td>
      <td>EIGHT UNIT</td>
      <td>SFR WATER  MDL-01</td>
      <td>SINGLE FAM  MDL-01</td>
      <td>SINGLE FAM  MDL-01</td>
      <td>TOWN-PROP  MDL-94</td>
      <td>SINGLE FAM  MDL-01</td>
      <td>SINGLE FAM  MDL-01</td>
      <td>THREE FAM  MDL-03</td>
    </tr>
    <tr>
      <td>Sq_Foot_Val</td>
      <td>89.2975</td>
      <td>5.35831</td>
      <td>47.1196</td>
      <td>44.7497</td>
      <td>40.1898</td>
      <td>46.2267</td>
      <td>13.2577</td>
      <td>186.108</td>
      <td>215.385</td>
      <td>136.649</td>
      <td>...</td>
      <td>5.16146</td>
      <td>9.47199</td>
      <td>12.5746</td>
      <td>10.9512</td>
      <td>18.8741</td>
      <td>7.45012</td>
      <td>0.292266</td>
      <td>8.22134</td>
      <td>18.7047</td>
      <td>2.5271</td>
    </tr>
    <tr>
      <td>Build_SQ_FT_Val</td>
      <td>156.392</td>
      <td>7.18907</td>
      <td>inf</td>
      <td>6.3373</td>
      <td>19.112</td>
      <td>20.456</td>
      <td>12.3384</td>
      <td>62.833</td>
      <td>54.6341</td>
      <td>31.6078</td>
      <td>...</td>
      <td>36.7213</td>
      <td>42.5425</td>
      <td>45.209</td>
      <td>79.8535</td>
      <td>103.006</td>
      <td>90.1282</td>
      <td>96.7969</td>
      <td>101.207</td>
      <td>87.0556</td>
      <td>55.303</td>
    </tr>
    <tr>
      <td>Land_Val_Per</td>
      <td>0.0378492</td>
      <td>0.567186</td>
      <td>0.0418358</td>
      <td>0.436486</td>
      <td>0.472201</td>
      <td>0.387337</td>
      <td>0.4662</td>
      <td>0.403455</td>
      <td>0.522388</td>
      <td>0.548712</td>
      <td>...</td>
      <td>0.418829</td>
      <td>0.381968</td>
      <td>0.274304</td>
      <td>0.382436</td>
      <td>0.311475</td>
      <td>0.484226</td>
      <td>0.736579</td>
      <td>0.414776</td>
      <td>0.383075</td>
      <td>0.304762</td>
    </tr>
  </tbody>
</table>
<p>35 rows × 130 columns</p>
</div>



Note the difference between the two parts of 2 Main Street in square foot valuations


```python
main_street = main_street.replace([np.inf, -np.inf], np.nan)
#main_street = main_street.drop(infinite_build_sq_values, inplace=True)
main_street.describe().transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Map</td>
      <td>130.0</td>
      <td>3.842308e+01</td>
      <td>5.990775e+00</td>
      <td>3.200000e+01</td>
      <td>3.700000e+01</td>
      <td>3.800000e+01</td>
      <td>3.800000e+01</td>
      <td>7.100000e+01</td>
    </tr>
    <tr>
      <td>Lot</td>
      <td>130.0</td>
      <td>1.017308e+02</td>
      <td>1.148620e+02</td>
      <td>1.000000e+00</td>
      <td>1.525000e+01</td>
      <td>4.650000e+01</td>
      <td>1.377500e+02</td>
      <td>3.870000e+02</td>
    </tr>
    <tr>
      <td>SubLot</td>
      <td>130.0</td>
      <td>9.230769e-02</td>
      <td>3.161335e-01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
    </tr>
    <tr>
      <td>Hist_Dist</td>
      <td>130.0</td>
      <td>7.692308e-01</td>
      <td>9.607689e-01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
      <td>2.000000e+00</td>
    </tr>
    <tr>
      <td>Pine_Tr_Zn</td>
      <td>130.0</td>
      <td>2.307692e-02</td>
      <td>1.507287e-01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <td>X10D_ID</td>
      <td>130.0</td>
      <td>1.038525e+09</td>
      <td>5.984506e+06</td>
      <td>1.032001e+09</td>
      <td>1.037032e+09</td>
      <td>1.038016e+09</td>
      <td>1.038141e+09</td>
      <td>1.071012e+09</td>
    </tr>
    <tr>
      <td>OBJECTID_1</td>
      <td>130.0</td>
      <td>4.689277e+03</td>
      <td>5.722596e+02</td>
      <td>3.515000e+03</td>
      <td>4.423250e+03</td>
      <td>4.604500e+03</td>
      <td>4.732750e+03</td>
      <td>7.247000e+03</td>
    </tr>
    <tr>
      <td>Vision_PID</td>
      <td>130.0</td>
      <td>6.774708e+03</td>
      <td>1.752966e+04</td>
      <td>2.943000e+03</td>
      <td>3.837250e+03</td>
      <td>4.042500e+03</td>
      <td>4.146750e+03</td>
      <td>1.208460e+05</td>
    </tr>
    <tr>
      <td>F10D_ID</td>
      <td>130.0</td>
      <td>1.038525e+09</td>
      <td>5.984506e+06</td>
      <td>1.032001e+09</td>
      <td>1.037032e+09</td>
      <td>1.038016e+09</td>
      <td>1.038141e+09</td>
      <td>1.071012e+09</td>
    </tr>
    <tr>
      <td>St_Num</td>
      <td>130.0</td>
      <td>3.234923e+02</td>
      <td>1.518856e+02</td>
      <td>2.000000e+00</td>
      <td>2.247500e+02</td>
      <td>3.530000e+02</td>
      <td>4.427500e+02</td>
      <td>6.300000e+02</td>
    </tr>
    <tr>
      <td>Land_Val</td>
      <td>130.0</td>
      <td>1.005338e+05</td>
      <td>7.932645e+04</td>
      <td>0.000000e+00</td>
      <td>6.392500e+04</td>
      <td>8.920000e+04</td>
      <td>1.217000e+05</td>
      <td>6.929000e+05</td>
    </tr>
    <tr>
      <td>Build_Val</td>
      <td>130.0</td>
      <td>4.007292e+05</td>
      <td>1.479667e+06</td>
      <td>0.000000e+00</td>
      <td>9.465000e+04</td>
      <td>1.719500e+05</td>
      <td>2.981250e+05</td>
      <td>1.646750e+07</td>
    </tr>
    <tr>
      <td>Total_Val</td>
      <td>130.0</td>
      <td>5.012631e+05</td>
      <td>1.530600e+06</td>
      <td>1.100000e+04</td>
      <td>1.783000e+05</td>
      <td>2.755500e+05</td>
      <td>4.096750e+05</td>
      <td>1.711530e+07</td>
    </tr>
    <tr>
      <td>Room_Cnt</td>
      <td>38.0</td>
      <td>1.031579e+01</td>
      <td>4.356451e+00</td>
      <td>4.000000e+00</td>
      <td>6.250000e+00</td>
      <td>1.000000e+01</td>
      <td>1.375000e+01</td>
      <td>2.100000e+01</td>
    </tr>
    <tr>
      <td>Build_SF</td>
      <td>130.0</td>
      <td>6.711746e+03</td>
      <td>1.093926e+04</td>
      <td>0.000000e+00</td>
      <td>1.938000e+03</td>
      <td>3.595000e+03</td>
      <td>8.242000e+03</td>
      <td>1.052960e+05</td>
    </tr>
    <tr>
      <td>Year_Built</td>
      <td>130.0</td>
      <td>1.701862e+03</td>
      <td>5.941291e+02</td>
      <td>0.000000e+00</td>
      <td>1.890000e+03</td>
      <td>1.900000e+03</td>
      <td>1.914500e+03</td>
      <td>2.004000e+03</td>
    </tr>
    <tr>
      <td>Land_Area</td>
      <td>130.0</td>
      <td>4.076477e+04</td>
      <td>2.819078e+05</td>
      <td>1.000000e+00</td>
      <td>5.227000e+03</td>
      <td>8.712000e+03</td>
      <td>1.720625e+04</td>
      <td>3.218648e+06</td>
    </tr>
    <tr>
      <td>Bath_Cnt</td>
      <td>36.0</td>
      <td>2.388889e+00</td>
      <td>1.121931e+00</td>
      <td>1.000000e+00</td>
      <td>1.500000e+00</td>
      <td>2.000000e+00</td>
      <td>3.000000e+00</td>
      <td>5.000000e+00</td>
    </tr>
    <tr>
      <td>Exempt_Val</td>
      <td>130.0</td>
      <td>2.246154e+03</td>
      <td>6.527297e+03</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.600000e+04</td>
    </tr>
    <tr>
      <td>Bed_Cnt</td>
      <td>50.0</td>
      <td>4.860000e+00</td>
      <td>2.213226e+00</td>
      <td>2.000000e+00</td>
      <td>3.000000e+00</td>
      <td>5.000000e+00</td>
      <td>6.000000e+00</td>
      <td>9.000000e+00</td>
    </tr>
    <tr>
      <td>Tax_Total</td>
      <td>130.0</td>
      <td>4.990169e+05</td>
      <td>1.531047e+06</td>
      <td>1.100000e+04</td>
      <td>1.730250e+05</td>
      <td>2.755500e+05</td>
      <td>4.096750e+05</td>
      <td>1.711530e+07</td>
    </tr>
    <tr>
      <td>Sq_Foot_Val</td>
      <td>130.0</td>
      <td>7.166755e+02</td>
      <td>5.510464e+03</td>
      <td>2.922656e-01</td>
      <td>1.409263e+01</td>
      <td>3.573822e+01</td>
      <td>6.856196e+01</td>
      <td>5.330000e+04</td>
    </tr>
    <tr>
      <td>Build_SQ_FT_Val</td>
      <td>116.0</td>
      <td>5.066459e+01</td>
      <td>2.675271e+01</td>
      <td>6.337302e+00</td>
      <td>3.282663e+01</td>
      <td>4.572936e+01</td>
      <td>6.007346e+01</td>
      <td>1.563925e+02</td>
    </tr>
    <tr>
      <td>Land_Val_Per</td>
      <td>130.0</td>
      <td>3.748779e-01</td>
      <td>2.314138e-01</td>
      <td>0.000000e+00</td>
      <td>2.431343e-01</td>
      <td>3.186798e-01</td>
      <td>4.352472e-01</td>
      <td>1.000000e+00</td>
    </tr>
  </tbody>
</table>
</div>



Note:
- Sq_Foot_Val: `7.166755e+02 ($71.66)`
- Build_SQ_FT_Val: `5.066459e+01 ($50.66)`


##### Value Generation Ideas:

- Wonder how much new value would be added if the second part of 2 Main Street's building was valued at the same rate of the first part on a sq foot basis?
- What does 6 Main Street look like with build sq valution at the mean instead of the low end?

#### 2 Main Street Build Improvements


```python
# new value of part B of 2 Main St if build value becomes comparable to A 
# by multplying it by Build_SQ_FT_Val of A
two_main_st_B_potenial_value_add = (156.392 * 12366) - 88900
print(two_main_st_B_potenial_value_add)
```

    1845043.4719999998


#### 6 Main St Build Improvements


```python
six_main_st_build_improv = (50.66 * 25200) - 159700
print(six_main_st_build_improv)
```

    1116932.0



```python
six_main_st_build_improv_high_end = (156.392 * 25200) - 159700
print(six_main_st_build_improv_high_end)
```

    3781378.4


These piecemeal improvements to the lower end building stock on the street that houses some of the most productive parcels of property in the city do not make much of an impact on reaching either of the garage valuation targets which is starting to make the idea that of developing the exisiting lower end building stock to a similar level of its current most productive as viable solutions look very questionable.

It is time to go larger in scale.

#### All of Main Street

What does all of Main Street look like with building sq/foot valuations like its max?


```python
main_street.sum().transpose()
```




    Map                                                             4995
    Lot                                                            13225
    SubLot                                                            12
    Polytype           parcelparcelparcelparcelparcelparcelparcelparc...
    LinkID             71-1171-1241-171-10-141-13641-13741-13839-7439...
    Hist_Dist                                                        100
    Pine_Tr_Zn                                                         3
    Lot_Sub            1112110-11361371387475767778384382381385380379...
    FD                 1212912912121212121212121212121212121212121212...
    X10D_ID                                                 135008225012
    OBJECTID_1                                                    609606
    Vision_PID                                                    880712
    GISID              71-1171-1241-171-10-141-13641-13741-13839-7439...
    F10D_ID                                                 135008225017
    St_Num                                                         42054
    Street             MAIN STMAIN STMAIN STMAIN STMAIN STMAIN STMAIN...
    Location           2 MAIN ST2 MAIN ST3 MAIN ST6 MAIN ST29 MAIN ST...
    Land_Val                                                    13069400
    Build_Val                                                   52094800
    Total_Val                                                   65164200
    Zone               MSRD3MSRD3W2MSRD3MSRD1MSRD1MSRD1MSRD1MSRD1MSRD...
    Room_Cnt                                                         392
    Build_SF                                                      872527
    Year_Built                                                    221242
    Land_Area                                                    5299420
    Bath_Cnt                                                          86
    Exempt_Val                                                    292000
    Bed_Cnt                                                          243
    Tax_Total                                                   64872200
    LU_CODE            40004000903540223220031C3160322032603220322032...
    LU_DESC4           FACTORYFACTORYTOWN-PROP  MDL-00IND BLDG  MDL-9...
    Sq_Foot_Val                                                  93167.8
    Build_SQ_FT_Val                                              5877.09
    Land_Val_Per                                                 48.7341
    dtype: object




```python
# Max Build_SQ_FT_Val: 1.563925e+02

all_main_st_max_improv = (872527 * 156.39) - 52094800
print(all_main_st_max_improv)
```

    84359697.53



```python
#47 Main Street build sq foot value: 62.833
all_main_st_middle_improv = (872527 * 62.833) - 52094800
print(all_main_st_middle_improv)
```

    2728688.9909999967


## Conclusions


This analysis has not looked at other municipalities and what are normal sq foot valuations in Maine or in the US but to achieve similar valuation increases as the proposed garage project means multiple large for Biddeford development projects (8-12 Saco Falls Way redevelopment projects) or making already relatively very productive streets much more productive (i.e. making the max building value per square foot the average).


```python
# looking for max build sq foot
cleanedtax = cleanedtax.replace([np.inf, -np.inf], np.nan)
cleanedtax.describe().transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Map</td>
      <td>7989.0</td>
      <td>3.594680e+01</td>
      <td>2.319775e+01</td>
      <td>1.000000e+00</td>
      <td>2.000000e+01</td>
      <td>3.400000e+01</td>
      <td>5.000000e+01</td>
      <td>8.800000e+01</td>
    </tr>
    <tr>
      <td>Lot</td>
      <td>7989.0</td>
      <td>7.115309e+01</td>
      <td>8.380606e+01</td>
      <td>1.000000e+00</td>
      <td>1.700000e+01</td>
      <td>3.900000e+01</td>
      <td>8.800000e+01</td>
      <td>4.760000e+02</td>
    </tr>
    <tr>
      <td>SubLot</td>
      <td>7989.0</td>
      <td>8.530479e-01</td>
      <td>2.479346e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>3.100000e+01</td>
    </tr>
    <tr>
      <td>Hist_Dist</td>
      <td>7989.0</td>
      <td>2.428339e-02</td>
      <td>2.132589e-01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
    </tr>
    <tr>
      <td>Pine_Tr_Zn</td>
      <td>7989.0</td>
      <td>1.940168e-02</td>
      <td>2.140510e-01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>3.000000e+00</td>
    </tr>
    <tr>
      <td>X10D_ID</td>
      <td>7989.0</td>
      <td>1.036018e+09</td>
      <td>2.319437e+07</td>
      <td>1.001001e+09</td>
      <td>1.020048e+09</td>
      <td>1.034255e+09</td>
      <td>1.050023e+09</td>
      <td>1.088055e+09</td>
    </tr>
    <tr>
      <td>OBJECTID_1</td>
      <td>7989.0</td>
      <td>4.030244e+03</td>
      <td>2.320302e+03</td>
      <td>1.000000e+00</td>
      <td>2.024000e+03</td>
      <td>4.035000e+03</td>
      <td>6.036000e+03</td>
      <td>8.043000e+03</td>
    </tr>
    <tr>
      <td>Vision_PID</td>
      <td>7989.0</td>
      <td>1.588417e+04</td>
      <td>3.414593e+04</td>
      <td>8.000000e+00</td>
      <td>2.155000e+03</td>
      <td>4.190000e+03</td>
      <td>6.257000e+03</td>
      <td>1.221280e+05</td>
    </tr>
    <tr>
      <td>F10D_ID</td>
      <td>7989.0</td>
      <td>1.035260e+09</td>
      <td>3.664681e+07</td>
      <td>0.000000e+00</td>
      <td>1.020047e+09</td>
      <td>1.034255e+09</td>
      <td>1.050023e+09</td>
      <td>1.088055e+09</td>
    </tr>
    <tr>
      <td>St_Num</td>
      <td>7989.0</td>
      <td>9.543122e+01</td>
      <td>1.490730e+02</td>
      <td>0.000000e+00</td>
      <td>8.000000e+00</td>
      <td>2.500000e+01</td>
      <td>1.100000e+02</td>
      <td>9.240000e+02</td>
    </tr>
    <tr>
      <td>Land_Val</td>
      <td>7989.0</td>
      <td>1.389953e+05</td>
      <td>2.516881e+05</td>
      <td>0.000000e+00</td>
      <td>5.530000e+04</td>
      <td>6.930000e+04</td>
      <td>9.860000e+04</td>
      <td>6.395900e+06</td>
    </tr>
    <tr>
      <td>Build_Val</td>
      <td>7989.0</td>
      <td>2.007895e+05</td>
      <td>9.874936e+05</td>
      <td>0.000000e+00</td>
      <td>8.860000e+04</td>
      <td>1.271000e+05</td>
      <td>1.817000e+05</td>
      <td>5.735940e+07</td>
    </tr>
    <tr>
      <td>Total_Val</td>
      <td>7989.0</td>
      <td>3.397848e+05</td>
      <td>1.129524e+06</td>
      <td>1.000000e+02</td>
      <td>1.553000e+05</td>
      <td>2.077000e+05</td>
      <td>2.994000e+05</td>
      <td>6.264060e+07</td>
    </tr>
    <tr>
      <td>Room_Cnt</td>
      <td>6146.0</td>
      <td>7.263261e+00</td>
      <td>3.265074e+00</td>
      <td>0.000000e+00</td>
      <td>5.000000e+00</td>
      <td>6.000000e+00</td>
      <td>8.000000e+00</td>
      <td>3.000000e+01</td>
    </tr>
    <tr>
      <td>Build_SF</td>
      <td>7989.0</td>
      <td>2.571734e+03</td>
      <td>8.609248e+03</td>
      <td>0.000000e+00</td>
      <td>1.056000e+03</td>
      <td>1.544000e+03</td>
      <td>2.312000e+03</td>
      <td>2.625800e+05</td>
    </tr>
    <tr>
      <td>Year_Built</td>
      <td>7989.0</td>
      <td>1.698538e+03</td>
      <td>6.555848e+02</td>
      <td>0.000000e+00</td>
      <td>1.900000e+03</td>
      <td>1.950000e+03</td>
      <td>1.984000e+03</td>
      <td>2.018000e+03</td>
    </tr>
    <tr>
      <td>Land_Area</td>
      <td>7989.0</td>
      <td>1.406506e+05</td>
      <td>8.794815e+05</td>
      <td>0.000000e+00</td>
      <td>7.841000e+03</td>
      <td>1.524600e+04</td>
      <td>4.356000e+04</td>
      <td>5.732496e+07</td>
    </tr>
    <tr>
      <td>Bath_Cnt</td>
      <td>5795.0</td>
      <td>1.894737e+00</td>
      <td>8.811677e-01</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>2.000000e+00</td>
      <td>2.500000e+00</td>
      <td>7.500000e+00</td>
    </tr>
    <tr>
      <td>Exempt_Val</td>
      <td>7989.0</td>
      <td>8.746264e+03</td>
      <td>1.035316e+04</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+04</td>
      <td>3.200000e+04</td>
    </tr>
    <tr>
      <td>Bed_Cnt</td>
      <td>6365.0</td>
      <td>3.422781e+00</td>
      <td>1.406263e+00</td>
      <td>0.000000e+00</td>
      <td>3.000000e+00</td>
      <td>3.000000e+00</td>
      <td>4.000000e+00</td>
      <td>9.000000e+00</td>
    </tr>
    <tr>
      <td>Tax_Total</td>
      <td>7989.0</td>
      <td>3.310385e+05</td>
      <td>1.130341e+06</td>
      <td>0.000000e+00</td>
      <td>1.434000e+05</td>
      <td>1.954000e+05</td>
      <td>2.873000e+05</td>
      <td>6.264060e+07</td>
    </tr>
    <tr>
      <td>Sq_Foot_Val</td>
      <td>7864.0</td>
      <td>3.215161e+02</td>
      <td>5.390838e+03</td>
      <td>3.478321e-03</td>
      <td>4.755823e+00</td>
      <td>1.453673e+01</td>
      <td>2.661134e+01</td>
      <td>3.715000e+05</td>
    </tr>
    <tr>
      <td>Build_SQ_FT_Val</td>
      <td>6956.0</td>
      <td>8.662137e+01</td>
      <td>4.808704e+01</td>
      <td>0.000000e+00</td>
      <td>5.874462e+01</td>
      <td>8.551535e+01</td>
      <td>1.063041e+02</td>
      <td>1.499564e+03</td>
    </tr>
    <tr>
      <td>Land_Val_Per</td>
      <td>7989.0</td>
      <td>4.574797e-01</td>
      <td>2.600106e-01</td>
      <td>0.000000e+00</td>
      <td>2.855706e-01</td>
      <td>3.552708e-01</td>
      <td>5.569106e-01</td>
      <td>1.000000e+00</td>
    </tr>
  </tbody>
</table>
</div>




```python
cleanedtax.sort_values('Build_SQ_FT_Val').tail(100).transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>5430</th>
      <th>765</th>
      <th>3978</th>
      <th>564</th>
      <th>7088</th>
      <th>1190</th>
      <th>5123</th>
      <th>3773</th>
      <th>5101</th>
      <th>3788</th>
      <th>...</th>
      <th>6667</th>
      <th>4638</th>
      <th>756</th>
      <th>5122</th>
      <th>6900</th>
      <th>1340</th>
      <th>878</th>
      <th>5507</th>
      <th>4677</th>
      <th>725</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Map</td>
      <td>49</td>
      <td>2</td>
      <td>38</td>
      <td>18</td>
      <td>75</td>
      <td>2</td>
      <td>4</td>
      <td>38</td>
      <td>4</td>
      <td>38</td>
      <td>...</td>
      <td>67</td>
      <td>40</td>
      <td>20</td>
      <td>4</td>
      <td>71</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>41</td>
      <td>20</td>
    </tr>
    <tr>
      <td>Lot</td>
      <td>9</td>
      <td>10</td>
      <td>296</td>
      <td>29</td>
      <td>6</td>
      <td>42</td>
      <td>40</td>
      <td>124</td>
      <td>35</td>
      <td>136</td>
      <td>...</td>
      <td>30</td>
      <td>55</td>
      <td>53</td>
      <td>4</td>
      <td>2</td>
      <td>56</td>
      <td>19</td>
      <td>14</td>
      <td>1</td>
      <td>26</td>
    </tr>
    <tr>
      <td>SubLot</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>Polytype</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>...</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
      <td>parcel</td>
    </tr>
    <tr>
      <td>LinkID</td>
      <td>49-9</td>
      <td>2-10-1</td>
      <td>38-296</td>
      <td>18-29-2</td>
      <td>75-6</td>
      <td>2-42-4</td>
      <td>4-40</td>
      <td>38-124-1</td>
      <td>4-35-1</td>
      <td>38-136</td>
      <td>...</td>
      <td>67-30</td>
      <td>40-55-1</td>
      <td>20-53</td>
      <td>4-4</td>
      <td>71-2</td>
      <td>2-56</td>
      <td>2-19</td>
      <td>5-14</td>
      <td>41-1</td>
      <td>20-26-1</td>
    </tr>
    <tr>
      <td>Hist_Dist</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Pine_Tr_Zn</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Lot_Sub</td>
      <td>9</td>
      <td>10-1</td>
      <td>296</td>
      <td>29-2</td>
      <td>6</td>
      <td>42-4</td>
      <td>40</td>
      <td>124-1</td>
      <td>35-1</td>
      <td>136</td>
      <td>...</td>
      <td>30</td>
      <td>55-1</td>
      <td>53</td>
      <td>4</td>
      <td>2</td>
      <td>56</td>
      <td>19</td>
      <td>14</td>
      <td>1</td>
      <td>26-1</td>
    </tr>
    <tr>
      <td>FD</td>
      <td>4</td>
      <td>18B</td>
      <td>13</td>
      <td>17</td>
      <td>1</td>
      <td>18</td>
      <td>4</td>
      <td>12</td>
      <td>4</td>
      <td>12</td>
      <td>...</td>
      <td>1</td>
      <td>12</td>
      <td>18</td>
      <td>4</td>
      <td>12</td>
      <td>18</td>
      <td>5</td>
      <td>4</td>
      <td>9</td>
      <td>18</td>
    </tr>
    <tr>
      <td>X10D_ID</td>
      <td>1049009000</td>
      <td>1002010001</td>
      <td>1038296000</td>
      <td>1018029002</td>
      <td>1075006000</td>
      <td>1002042004</td>
      <td>1004040000</td>
      <td>1038124001</td>
      <td>1004035001</td>
      <td>1038136000</td>
      <td>...</td>
      <td>1067030000</td>
      <td>1040055001</td>
      <td>1020053000</td>
      <td>1004004000</td>
      <td>1071002000</td>
      <td>1002056000</td>
      <td>1002019000</td>
      <td>1005014000</td>
      <td>1041001000</td>
      <td>1020026001</td>
    </tr>
    <tr>
      <td>OBJECTID_1</td>
      <td>5996</td>
      <td>433</td>
      <td>4898</td>
      <td>1946</td>
      <td>7342</td>
      <td>488</td>
      <td>956</td>
      <td>4713</td>
      <td>939</td>
      <td>4727</td>
      <td>...</td>
      <td>7018</td>
      <td>5484</td>
      <td>2087</td>
      <td>800</td>
      <td>7233</td>
      <td>539</td>
      <td>447</td>
      <td>1099</td>
      <td>5501</td>
      <td>2059</td>
    </tr>
    <tr>
      <td>Vision_PID</td>
      <td>102384</td>
      <td>184</td>
      <td>4310</td>
      <td>1430</td>
      <td>6734</td>
      <td>100584</td>
      <td>611</td>
      <td>4125</td>
      <td>602</td>
      <td>4139</td>
      <td>...</td>
      <td>6425</td>
      <td>4901</td>
      <td>1656</td>
      <td>510</td>
      <td>6635</td>
      <td>119952</td>
      <td>198</td>
      <td>746</td>
      <td>4918</td>
      <td>1626</td>
    </tr>
    <tr>
      <td>GISID</td>
      <td>49-9</td>
      <td>2-10-1</td>
      <td>38-296</td>
      <td>18-29-2</td>
      <td>75-6</td>
      <td>2-42-4</td>
      <td>4-40</td>
      <td>38-124-1</td>
      <td>4-35-1</td>
      <td>38-136</td>
      <td>...</td>
      <td>67-30</td>
      <td>40-55-1</td>
      <td>20-53</td>
      <td>4-4</td>
      <td>71-2</td>
      <td>2-56</td>
      <td>2-19</td>
      <td>5-14</td>
      <td>41-1</td>
      <td>20-26-1</td>
    </tr>
    <tr>
      <td>F10D_ID</td>
      <td>1049009000</td>
      <td>1002010001</td>
      <td>1038296000</td>
      <td>1018029002</td>
      <td>1075006000</td>
      <td>1002042004</td>
      <td>1004040000</td>
      <td>1038124001</td>
      <td>1004035001</td>
      <td>1038136000</td>
      <td>...</td>
      <td>1067030000</td>
      <td>1040055001</td>
      <td>1020053000</td>
      <td>1004004000</td>
      <td>1071002000</td>
      <td>1002056000</td>
      <td>1002019000</td>
      <td>1005014000</td>
      <td>1041001000</td>
      <td>1020026001</td>
    </tr>
    <tr>
      <td>St_Num</td>
      <td>26</td>
      <td>594</td>
      <td>69</td>
      <td>480</td>
      <td>21</td>
      <td>10</td>
      <td>888</td>
      <td>21</td>
      <td>463</td>
      <td>257</td>
      <td>...</td>
      <td>20</td>
      <td>0</td>
      <td>440</td>
      <td>331</td>
      <td>0</td>
      <td>45</td>
      <td>439</td>
      <td>23</td>
      <td>3</td>
      <td>424</td>
    </tr>
    <tr>
      <td>Street</td>
      <td>FERRY LN</td>
      <td>ALFRED ST</td>
      <td>ADAMS ST</td>
      <td>ELM ST</td>
      <td>GRANITE POINT RD</td>
      <td>HEALTHCARE DR</td>
      <td>POOL ST</td>
      <td>STONE ST</td>
      <td>WEST ST</td>
      <td>MAIN ST</td>
      <td>...</td>
      <td>SEA SPRAY DR</td>
      <td>GOOCH ST (OFF OF)</td>
      <td>ALFRED ST</td>
      <td>WEST ST</td>
      <td>LINCOLN ST</td>
      <td>BOULDER WAY</td>
      <td>SOUTH ST</td>
      <td>ELIZABETH RD</td>
      <td>MAIN ST</td>
      <td>ALFRED ST</td>
    </tr>
    <tr>
      <td>Location</td>
      <td>26 FERRY LN</td>
      <td>594 ALFRED ST</td>
      <td>69 ADAMS ST</td>
      <td>480 ELM ST</td>
      <td>21 GRANITE POINT RD</td>
      <td>10 HEALTHCARE DR</td>
      <td>888 POOL ST</td>
      <td>21 STONE ST</td>
      <td>463 WEST ST</td>
      <td>257 MAIN ST</td>
      <td>...</td>
      <td>20 SEA SPRAY DR</td>
      <td>GOOCH ST (OFF OF)</td>
      <td>440 ALFRED ST</td>
      <td>331 WEST ST</td>
      <td>LINCOLN ST</td>
      <td>45 BOULDER WAY</td>
      <td>439 SOUTH ST</td>
      <td>23 ELIZABETH RD</td>
      <td>3 MAIN ST</td>
      <td>424 ALFRED ST</td>
    </tr>
    <tr>
      <td>Land_Val</td>
      <td>157900</td>
      <td>138500</td>
      <td>160900</td>
      <td>132400</td>
      <td>161300</td>
      <td>167000</td>
      <td>167600</td>
      <td>141900</td>
      <td>24800</td>
      <td>135400</td>
      <td>...</td>
      <td>901000</td>
      <td>23100</td>
      <td>1000800</td>
      <td>1136200</td>
      <td>1002600</td>
      <td>1350700</td>
      <td>364900</td>
      <td>1651000</td>
      <td>118500</td>
      <td>2948000</td>
    </tr>
    <tr>
      <td>Build_Val</td>
      <td>0</td>
      <td>20600</td>
      <td>0</td>
      <td>28800</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>26000</td>
      <td>143300</td>
      <td>36600</td>
      <td>...</td>
      <td>0</td>
      <td>1041000</td>
      <td>94300</td>
      <td>0</td>
      <td>149500</td>
      <td>0</td>
      <td>1196600</td>
      <td>0</td>
      <td>2714000</td>
      <td>221200</td>
    </tr>
    <tr>
      <td>Total_Val</td>
      <td>157900</td>
      <td>159100</td>
      <td>160900</td>
      <td>161200</td>
      <td>161300</td>
      <td>167000</td>
      <td>167600</td>
      <td>167900</td>
      <td>168100</td>
      <td>172000</td>
      <td>...</td>
      <td>901000</td>
      <td>1064100</td>
      <td>1095100</td>
      <td>1136200</td>
      <td>1152100</td>
      <td>1350700</td>
      <td>1561500</td>
      <td>1651000</td>
      <td>2832500</td>
      <td>3169200</td>
    </tr>
    <tr>
      <td>Zone</td>
      <td>SR1</td>
      <td>B2</td>
      <td>MSRD2</td>
      <td>R1A</td>
      <td>CR</td>
      <td>I3</td>
      <td>RF CR</td>
      <td>MSRD1</td>
      <td>RF</td>
      <td>MSRD1</td>
      <td>...</td>
      <td>CR</td>
      <td>MSRD3</td>
      <td>R1A</td>
      <td>IN</td>
      <td>MSRD3</td>
      <td>I3</td>
      <td>RF SR1</td>
      <td>CR</td>
      <td>W2</td>
      <td>I-3</td>
    </tr>
    <tr>
      <td>Room_Cnt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Build_SF</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Year_Built</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Land_Area</td>
      <td>121968</td>
      <td>21780</td>
      <td>67082</td>
      <td>52272</td>
      <td>81893</td>
      <td>258746</td>
      <td>368082</td>
      <td>28750</td>
      <td>2613600</td>
      <td>16117</td>
      <td>...</td>
      <td>28314</td>
      <td>2178</td>
      <td>1089000</td>
      <td>16069720</td>
      <td>21780004</td>
      <td>327135</td>
      <td>5645376</td>
      <td>1586020</td>
      <td>60113</td>
      <td>1014948</td>
    </tr>
    <tr>
      <td>Build_Styl</td>
      <td>Vacant Land</td>
      <td>NaN</td>
      <td>Vacant Land</td>
      <td>NaN</td>
      <td>Vacant Land</td>
      <td>Vacant Land</td>
      <td>Vacant Land</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>Vacant Land</td>
      <td>Vacant Land</td>
      <td>NaN</td>
      <td>Vacant Land</td>
      <td>NaN</td>
      <td>Vacant Land</td>
      <td>NaN</td>
      <td>Vacant Land</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Bath_Cnt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Exempt_Val</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Bed_Cnt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Tax_Total</td>
      <td>157900</td>
      <td>159100</td>
      <td>160900</td>
      <td>161200</td>
      <td>161300</td>
      <td>167000</td>
      <td>167600</td>
      <td>167900</td>
      <td>168100</td>
      <td>172000</td>
      <td>...</td>
      <td>901000</td>
      <td>1064100</td>
      <td>1095100</td>
      <td>1136200</td>
      <td>1152100</td>
      <td>1350700</td>
      <td>1561500</td>
      <td>1651000</td>
      <td>2832500</td>
      <td>3169200</td>
    </tr>
    <tr>
      <td>LU_CODE</td>
      <td>1300</td>
      <td>1060</td>
      <td>1300</td>
      <td>3910</td>
      <td>1300</td>
      <td>9035</td>
      <td>9100</td>
      <td>337V</td>
      <td>7180</td>
      <td>337V</td>
      <td>...</td>
      <td>1320</td>
      <td>4420</td>
      <td>9100</td>
      <td>9040</td>
      <td>9035</td>
      <td>322A</td>
      <td>4200</td>
      <td>9000</td>
      <td>9035</td>
      <td>3900</td>
    </tr>
    <tr>
      <td>LU_DESC4</td>
      <td>RES ACLNDV  MDL-00</td>
      <td>AC LND OUTBLDG</td>
      <td>RES ACLNDV  MDL-00</td>
      <td>POT DEVEL</td>
      <td>RES ACLNDV  MDL-00</td>
      <td>TOWN-PROP  MDL-00</td>
      <td>CHARI/BENO  MDL-00</td>
      <td>PARK LOT  MDL-00</td>
      <td>PASTURE  MDL-00</td>
      <td>PARK LOT  MDL-00</td>
      <td>...</td>
      <td>RES ACLNUD  MDL-00</td>
      <td>IND LD UD  MDL-00</td>
      <td>CHARI/BENO  MDL-00</td>
      <td>LIT/SCIENT  MDL-00</td>
      <td>TOWN-PROP  MDL-00</td>
      <td>COMM BLDG  MDL-00</td>
      <td>PUB TANKS</td>
      <td>US GOVT  MDL-00</td>
      <td>TOWN-PROP  MDL-00</td>
      <td>DEVEL LAND  MDL-00</td>
    </tr>
    <tr>
      <td>Sq_Foot_Val</td>
      <td>1.2946</td>
      <td>7.30487</td>
      <td>2.39856</td>
      <td>3.08387</td>
      <td>1.96964</td>
      <td>0.645421</td>
      <td>0.455333</td>
      <td>5.84</td>
      <td>0.0643174</td>
      <td>10.672</td>
      <td>...</td>
      <td>31.8217</td>
      <td>488.567</td>
      <td>1.0056</td>
      <td>0.0707044</td>
      <td>0.0528971</td>
      <td>4.12888</td>
      <td>0.276598</td>
      <td>1.04097</td>
      <td>47.1196</td>
      <td>3.12252</td>
    </tr>
    <tr>
      <td>Build_SQ_FT_Val</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Land_Val_Per</td>
      <td>1</td>
      <td>0.870522</td>
      <td>1</td>
      <td>0.82134</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.845146</td>
      <td>0.147531</td>
      <td>0.787209</td>
      <td>...</td>
      <td>1</td>
      <td>0.0217085</td>
      <td>0.913889</td>
      <td>1</td>
      <td>0.870237</td>
      <td>1</td>
      <td>0.233686</td>
      <td>1</td>
      <td>0.0418358</td>
      <td>0.930203</td>
    </tr>
  </tbody>
</table>
<p>35 rows × 100 columns</p>
</div>




```python

```
