# PLURAL: Place-level urban-rural indices
  
PLURAL is a framework to create continuous classifications of "rurality" or "urbanness" based on the spatial configuration of populated places.
PLURAL makes use of the concept of "remoteness" to characterize the level of spatial isolation of a populated place with respect to its neighbors.
There are two implementations of PLURAL, including (a) PLURAL-1, based on distances to the nearest places of user-specified population classes, and (b) PLURAL-2, based on neighborhood characterization derived from spatial networks.
PLURAL requires simplistic input data, i.e., the coordinates (x,y) and population p of populated places (villages, towns, cities) in a given point in time. Due to its simplistic input, the PLURAL rural-urban classification scheme can be applied to historical data, as well as to data from data-scarce settings.

Using the PLURAL framework, we created place-level rural-urban indices for the conterminous United States from 1930 to 2018. These data are publicly available and con be downloaded at URL.

<img width="450" src="https://github.com/johannesuhl/plural/blob/main/inv750ms_PLURAL_2_scaled_across_years_equal_weights.gif"> <img width="450" src="https://github.com/johannesuhl/plural/blob/main/invSUBSET_750ms_PLURAL_2_scaled_across_years_equal_weights.gif">

Rural-urban classifications are essential for analyzing geographic, demographic, environmental, and social processes across the rural-urban continuum. Most existing classifications are, however, only available at relatively aggregated spatial scales, such as at the county scale in the United States. The absence of rurality or urbanness measures at high spatial resolution poses significant problems when the process of interest is highly localized, as with the incorporation of rural towns and villages into encroaching metropolitan areas. Moreover, existing rural-urban classifications are often inconsistent over time, or require complex, multi-source input data (e.g., remote sensing observations or road network data), thus, prohibiting the longitudinal analysis of rural-urban dynamics. We developed a set of distance- and spatial-network-based methods for consistently estimating the remoteness and rurality of places at fine spatial resolution, over long periods of time. Based on these methods, we constructed indices of urbanness for 30,000 places in the United States from 1930 to 2018. We call these indices the place-level urban-rural index (PLURAL), enabling long-term, fine-grained analyses of urban and rural change in the United States. The method paper has been peer-reviewed and is accepted for publication in "Landscape and Urban Planning":

Uhl, J.H., Hunter, L.M., Leyk, S., Connor, D.S., Nieves, J.J., Hester, C., Talbot, C.B. and Gutmann, M., 2022. Place-level urban-rural indices for the United States from 1930 to 2018. arXiv preprint arXiv:2202.13767. (https://arxiv.org/abs/2202.13767)

Funding Sources:  
United States Department of Health and Human Services. National Institutes of Health. Eunice Kennedy Shriver National Institute of Child Health and Human Development (2P2CHD066613-06); United States Department of Health and Human Services. National Institutes of Health. Eunice Kennedy Shriver National Institute of Child Health and Human Development (5R21HD098717-02) 

Previous versions of the PLURAL methodology:
Uhl, Johannes H., Stefan Leyk, Lori M. Hunter, Catherine B. Talbot, Dylan S. Connor, Jeremiah J. Nieves, and Myron Gutmann. “A Fine-Grained, Versatile Index of Remoteness to Characterize Place-Level Rurality.” In Annual Meeting of the Population Association of America (PAA). online, 2021. https://doi.org/10.48550/arXiv.2202.08496.

Data sources for the sample data in this repository: Place-level population counts (1980-2010) and place locations 1930 - 2018 were obtained from IPUMS NHGIS, (University of Minnesota, www.nhgis.org; Manson et al. 2022). Place-level population counts 1930 were digitized from historical census records (U.S. Census Bureau 1942, 1964).

References:

Steven Manson, Jonathan Schroeder, David Van Riper, Tracy Kugler, and Steven Ruggles. IPUMS National Historical Geographic Information System: Version 16.0 [dataset]. Minneapolis, MN: IPUMS. 2021. http://doi.org/10.18128/D050.V16.0

U.S. Census Bureau (1942). U.S. Census of Population: 1940. Vol. I, Number of Inhabitants. U.S. Government Printing Office, Washington, D.C.

U.S. Census Bureau (1964). U.S. Census of Population: 1960. Vol. I, Characteristics of the Population. Part I, United States Summary. U.S. Government Printing Office, Washington, D.C. 
