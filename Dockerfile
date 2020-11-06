### Dockerfile
# Created 11/5/20
# Include data file and script in image. 

FROM rocker/verse:latest
RUN R -e "install.packages(c('tidyverse','tidymodels','caret','xgboost','h2o','visdat','viridis','reshape2','RColorBrewer','ggridges','DMwR','MLmetrics','e1071'),dependencies=TRUE,repos = 'http://cran.us.r-project.org')"
COPY diagnosis-of-covid-19-and-its-clinical-spectrum.csv /home/rstudio/
COPY README.Rmd /home/rstudio/

