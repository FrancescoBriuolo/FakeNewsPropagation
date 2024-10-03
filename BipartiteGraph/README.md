<p align= "center">
<img src="https://www.unisannio.it/sites/default/files/emblema.png.pagespeed.ce.L9uvAVRynq.png" alt="Unisannio" width= 30%>
</p>
<p align="center">
    <img src="https://img.shields.io/badge/Python-v3-blue" alt="Python">
    <img src="https://img.shields.io/badge/Unisannio-Evoluzione e Qualità del Software" alt="Unisannio">
</p>

This project contains three Python's scripts. Two of these (init_archive_with_keywords.py and update_archive_with_keywords.py) are for the scraping activity from the fact-checking website "Politifact". 
The "init_archive_with_keywords.py" script enables the creation of two CSV files. Entering a keyword creates the 'archive_with_keywords.csv' file, which contains the latest news related to the first n pages (where n is an input parameter) for search with the keyword entered. If a keyword is not entered (""), the 'archive.csv' file is created and populated with the most recent statements for the first n pages specified. 
The "update_archive_with_keywords.py" script enables to update the previously created files. Entering a keyword updates the 'archive_with_keywords.csv' file adding to the top of the list all the most recent stuff relative to the keywords with which
the search was performed. If a keyword is not entered (""), the 'archive.csv' file is updated with the most recent statements for the first n pages specified. 
It is necessary to specify that for the purposes of this thesis project only scripts related to keyword searches were used, although the others are useful for possible future development.
