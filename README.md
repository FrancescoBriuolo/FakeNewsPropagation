<p align= "center">
<img src="https://www.unisannio.it/sites/default/files/emblema.png.pagespeed.ce.L9uvAVRynq.png" alt="Unisannio" width= 30%>
</p>
<p align="center">
    <img src="https://img.shields.io/badge/Python-v3-blue" alt="Python">
    <img src="https://img.shields.io/badge/Unisannio-Evoluzione e Qualità del Software" alt="Unisannio">
</p>

# Evaluation of graph-based techniques for the analysis of fake news and propagation metrics
This thesis project entitled “Evaluation of graph-based techniques for the analysis of fake news and propagation metrics” involved two parallel projects. The first consists of extracting topic-specific data from the fact-checking site PolitiFact by web scraping. These data are used to construct a bipartite graph of sources and keywords, which is analyzed by considering relative projections in order to extract information about the veracity of the news. The second experiment is a study of a hierarchical propagation network for fake news detection. The goal is to use network features to implement multiple machine learning models and a Feedforward neural network for fake news detection using data provided by the FakeNewsNet dataset and perform a sensitivity analysis of the classification algorithms used with respect to the features considered. ​

## Bipartite graph
This project contains three Python's scripts. Two of these (init_archive_with_keywords.py and update_archive_with_keywords.py) are for the scraping activity from the fact-checking website "Politifact". 
The "init_archive_with_keywords.py" script enables the creation of two CSV files. Entering a keyword creates the 'archive_with_keywords.csv' file, which contains the latest news related to the first n pages (where n is an input parameter) for search with the keyword entered. If a keyword is not entered (""), the 'archive.csv' file is created and populated with the most recent statements for the first n pages specified. 
The "update_archive_with_keywords.py" script enables to update the previously created files. Entering a keyword updates the 'archive_with_keywords.csv' file adding to the top of the list all the most recent stuff relative to the keywords with which
the search was performed. If a keyword is not entered (""), the 'archive.csv' file is updated with the most recent statements for the first n pages specified. 
It is necessary to specify that for the purposes of this thesis project only scripts related to keyword searches were used, although the others are useful for possible future development.
The CSV file 'archive_with_keywords.csv' consists of 6 fields: author (the fact-checker), statement (news content), source (source of the news - person, social post, organization), date (of the review), target (label assigned by the fact-checker: True, Mostly True, Half True, Mostly False, False, Pants on Fire), keyword (with which the search is executed).

