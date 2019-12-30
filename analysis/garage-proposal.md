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

```
millrate = 19.98 / 1000
example_home = 227100
tax_value_pre_homestead = example_home * millrate
print(tax_value_pre_homestead)


4537.4580000000005
```


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

 ```
 garage_10_year_total_rev = 16407604
garage_30_year_total_rev = 39772744

# 10 year slope check
average_10_year = garage_10_year_total_rev / 10

# 30 year slope check
average_30_year = garage_30_year_total_rev / 30

print(average_10_year)
1640760.4


print(average_30_year)
1325758.1333333333
```

In overall property value increase to the tax base, proposals need to consisently maintain these valuations:

- **30 Years:**`$66,354,300`
- **10 Years:**`$82,120,100` 

which equates to roughly 2.5% for 30 years and 3.1% total increases of property valuations for Biddeford 


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

```
gooch_st_becomes_saco_falls = (7037400 * 2) - 1041000

gooch_st_improvements_10_years_percent = (gooch_st_becomes_saco_falls / value_increase_10_years) * 100
gooch_st_improvements_30_years_percent = (gooch_st_becomes_saco_falls / value_increase_30_years) * 100

print(gooch_st_improvements_30_years_percent, gooch_st_improvements_10_years_percent)

19.64274760625015 15.87162415670198

```


Doing a rough calculation and assuming that cloning the build value pf 75 Saco Falls is a likely possibility for the two Gooch street populations the new value would cover ~ `20%` and `16%` for the `30 year` and `10 year` benchmarks.


These improvements alone will not suffice. Main Street improvements and the Lester B. Orcutt BLVD parcel(s) look promising.


##### Lester B. Orcutt BLVD

<img src="9-lb-orcutt-blvd-street-view.jpg" />



Quickly looking at the build to land values and the total values these properties on their own are not going to make noticble dents to reaching the garage value add goals.

This analysis probably should also look at: 
- land value as a percent of of total value
- build value per sq foot

This is so time is not spent any further looking at properties where adding or modifying the development on said land would produce negible effects on reaching the garage valuation hurdle.


#### Main St. Parcels 

**47 and 49 Main St.**

Notice the left side is an empty mill building while across the street is some of the most productive real estate in the city. 

<img src="images/main-street-street-view.jpg" />



**145 Main St.***

Notice that the style is similar but renovated, one-two stories taller and next to a small park.

<img src="images/145-main-st-street-view.jpg" />


Note:
- Sq_Foot_Val: `7.166755e+02 ($71.66)`
- Build_SQ_FT_Val: `5.066459e+01 ($50.66)`


##### Value Generation Ideas:

- Wonder how much new value would be added if the second part of 2 Main Street's building was valued at the same rate of the first part on a sq foot basis?
- What does 6 Main Street look like with build sq valution at the mean instead of the low end?

#### 2 Main Street Build Improvements

```
# new value of part B of 2 Main St if build value becomes comparable to A 
# by multplying it by Build_SQ_FT_Val of A
two_main_st_B_potenial_value_add = (156.392 * 12366) - 88900
print(two_main_st_B_potenial_value_add)

1845043.4719999998
```

## Conclusions


This analysis has not looked at other municipalities and what are normal sq foot valuations in Maine or in the US but to achieve similar valuation increases as the proposed garage project means multiple large for Biddeford development projects (8-12 Saco Falls Way redevelopment projects) or making already relatively very productive streets much more productive (i.e. making the max building value per square foot the average).