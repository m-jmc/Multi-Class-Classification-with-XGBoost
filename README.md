---
title: "README.md"
output: 
  html_document:
    toc: true
    toc_depth: 3
    keep_md: TRUE
---



# COVID Severity

**Overview:** In this project my goal was to identify metabolic differences positive COVID-19 patients which might indicate increased acuity. It represents a step forward from my previous projects in code technique and extensibility, better data hygiene and elimination of data leakage, greater command of the Caret modeling process, and (still in process) more robust outcome evaluation.
<br>

**Highlights:**

*	Missing Data Visualizations.
*	Introduction of the recipes package.

*	XGBoost:
    +	Hyperparameter tuning using grid search.
    +	Caret multi-class summary.
    +	10-fold cross validation.
    +	SMOTE subsampling for improved imbalanced classification.
    
* PCA Evaluation

<br>

**Outcome:** The resulting model was able to achieve acceptable accuracy owing to high sensitivity of the dominate class (Outpatient treatment patients). However, minority classification performance was poor, and the model failed to achieve statistical significance (Floor and ICU patients). I believe this to be a result of poor representation within subsamples, limited number of features, and limited nature of the data set. 

<br>

## About the Data

This dataset contains anonymized data from patients seen at the Hospital Israelita Albert Einstein, at São Paulo, Brazil, and who had samples collected to perform the SARS-CoV-2 RT-PCR and additional laboratory tests during a visit to the hospital. All data were anonymized following the best international practices and recommendations. All clinical data were standardized to have a mean of zero and a unit standard deviation. We aimed at including laboratory tests more commonly order during a visit to the emergency room



<br>





<br>

## Something is Missing {.tabset .tabset-fade}

Upon loading the data set you're immediately struck by the amount of missing values. Plotting the count of NA by feature we can better see the true extent. <br>


### Percent Missing by Feature


```
## `summarise()` regrouping output by 'col', 'total' (override with `.groups` argument)
```

![](README_files/figure-html/unnamed-chunk-2-1.png)<!-- -->

### Missing by Row

![](README_files/figure-html/unnamed-chunk-3-1.png)<!-- -->
<br>

### Admission Recoding

Many of the present values are the TRUE/FALSE patient admission status, with no overlap across acuity levels these will be recoded into a single admission column where floor (general ward) = 1, step (semi intensive care) and icu = 2, and no status is assumed to be outpatient (OP) = 0. 


```r
data$admission <- base::ifelse(data$floor == "TRUE", "1", 
                               base::ifelse(data$step == "TRUE", "2",
                                            base::ifelse(data$icu == "TRUE", "2", "0"))) %>% base::as.numeric()

# remove the old floor/step/icu columns, filter for covid positive patients only, then reorder the columns to put admission target first
positive <- data %>% 
              select(-floor, 
                     -step,
                     -icu,
                     -patient_id) %>%
              filter(COVID == "positive") %>%
              select(admission, 
                     everything())
```


<br>

## Recipies {.tabset .tabset-fade}

Then we evaluate non-numeric columns for ordinal features. While many are simply boolean values (ie detected / not_detected) urine micro results represent an ordinal condition (such as aspect: clear, lightly cloudy, cloudy) must be recoded. <br>

This is where we'll introduce the recipes package to create integer features, drop our zero and near zero variance variables and stratify the training set based off the target "admission" variable. Given the poor distribution of the target variable, we'll use a subsampling technique during our training process. <br>

<br>

### Split


```r
# Selecting non numeric columns into a df, looking for features which may have ordinality
nominal <- positive %>% dplyr::select_if(~!is.numeric(.x))

# Create Training and Testing Set: because there are 506 unclassified admissions and 52 positive cases, we'll use the stratify feature of the recipes package to ensure that our target variable is distributed evenly across 70/30 split training and testing subsets

table(positive$admission) %>% prop.table()
```

```
## 
##          0          1          2 
## 0.90681004 0.06451613 0.02867384
```

```r
set.seed(333)
split  <- initial_split(positive, 
                        prop = 0.7, 
                        strata = "admission")

train  <- training(split)
test   <- testing(split)

# The result is a proportionally distributed target variable in our training split
table(train$admission) %>% prop.table()
```

```
## 
##          0          1          2 
## 0.90025575 0.06649616 0.03324808
```

```r
table(test$admission) %>% prop.table()
```

```
## 
##          0          1          2 
## 0.92215569 0.05988024 0.01796407
```
<br>

### Designing "DNA" with Recipies

Building our recipes DNA to perform the following steps: <br>
- Remove zero and near zero variance variables <br>
- Create ordinal integers for specified nominal columns <br>
- Center and scale all numeric variables <br>
- One-hot encode remaining nominal features <br>
- Once the recipe is created, we prep the sample to estimate the outcome, then bake our training and test samples. <br>
- The preped mRNA then processes ("pro") the training set to create the protrain df... protrain... like protein, get it? Cause the mRNA...<br>

Next steps will be to build this directly into my caret training process, but I ran into issues when evaluating the prediction. <br>


```r
# specify admission as an outcome column
dna <- recipe(admission ~ ., data = positive) %>%
   step_nzv(all_predictors())  %>%
   step_integer(matches("urine_color|urine_aspect|urine_hemoglobin|urine_leukocytes")) %>%
   step_center(all_numeric(), -all_outcomes()) %>%
   step_scale(all_numeric(), -all_outcomes()) %>%
   step_dummy(all_nominal(), one_hot = TRUE)
 
mrna <- prep(dna, training = train)

# pro = processed, resulting in "pro-train", like protein, get it?
protrain <- bake(mrna, new_data = train)
protest <- bake(mrna, new_data = test)
```

<br>

## XGBoost with Caret {.tabset .tabset-fade}

  Using XGBoost for its ability to handle and incorporate missing values. Given that we're primarily working with patient medical lab data, it would be inappropriate to impute missing values. This precludes us from being able to evaluate for highly correlated features, with non-boosted tree models (random forest) this could cause correlated features which are otherwise strong predictors to be under represented in the resulting ensemble. With XGBoost if two predictors are perfectly correlated the descent gradient cost function will essentially "pick one" and carry it forward for the remaining ensembles. We wont know if this happens without calculating correlations, but the main impact would be in our data storytelling. <br> 



<br>

### Tuning + SMOTE

Setting up our Tuning grid to find best performing hyperparameters. Using the Caret train function to perform 10-fold cross validation with multiclass summary. To better accommodate our imbalanced classification, we're using the hybrid subsampling method SMOTE (synthetic Minority Over Sampling Technique) performed within each cross validation subset. 


```r
xgb_grid = expand.grid(
                  nrounds = 1000,
                  eta = c(0.1), #0.05, 0.01), # limited for processing brevity
                  max_depth = c(2, 3, 4, 5, 6, 7, 8),
                  gamma = 0,
                  colsample_bytree=1,
                  min_child_weight=c(1, 2, 3, 4 ,5),
                  subsample=1)

control <- trainControl(method ="cv", 
                        number = 10,
                        classProbs = TRUE, 
                        summaryFunction = multiClassSummary,
                        sampling = "smote")


set.seed(333)
xgb_caret <- train(x=X_train, 
                   y=Y_train_char$Y_train, 
                   method ='xgbTree',
                   metric ="logLoss",
                   #nthreads = 6,
                   trControl = control,
                   tuneGrid = xgb_grid
                   ) 

# Best tune parameters from hyper parameter grid search
#xgb_caret$results
xgb_caret$bestTune
```

```
##    nrounds max_depth eta gamma colsample_bytree min_child_weight subsample
## 19    1000         5 0.1     0                1                4         1
```

<br>

### Design

![](README_files/figure-html/unnamed-chunk-9-1.png)<!-- -->

<br>

### Performance


```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  OP Floor ICU
##      OP    145     3   0
##      Floor   0     0   0
##      ICU     9     7   3
## 
## Overall Statistics
##                                          
##                Accuracy : 0.8862         
##                  95% CI : (0.828, 0.9301)
##     No Information Rate : 0.9222         
##     P-Value [Acc > NIR] : 0.9637324      
##                                          
##                   Kappa : 0.3704         
##                                          
##  Mcnemar's Test P-Value : 0.0002734      
## 
## Statistics by Class:
## 
##                      Class: OP Class: Floor Class: ICU
## Sensitivity             0.9416      0.00000    1.00000
## Specificity             0.7692      1.00000    0.90244
## Pos Pred Value          0.9797          NaN    0.15789
## Neg Pred Value          0.5263      0.94012    1.00000
## Prevalence              0.9222      0.05988    0.01796
## Detection Rate          0.8683      0.00000    0.01796
## Detection Prevalence    0.8862      0.00000    0.11377
## Balanced Accuracy       0.8554      0.50000    0.95122
```

<br>

### Feature Importance


![](README_files/figure-html/unnamed-chunk-11-1.png)<!-- -->

<br>



<br>

## Feature Exploration {.tabset .tabset-fade}

<br>

### Violin 

![](README_files/figure-html/unnamed-chunk-13-1.png)<!-- -->

<br>

### Density 

![](README_files/figure-html/unnamed-chunk-14-1.png)<!-- -->
<br>

### Box and Jitter 

![](README_files/figure-html/unnamed-chunk-15-1.png)<!-- -->

<br>

### Ridge


```
## Picking joint bandwidth of 0.316
```

![](README_files/figure-html/unnamed-chunk-16-1.png)<!-- -->


<br>

# Identification of COVID Positive Patients

Failing to properly classify patient acuity, we may yet be able to identify COVID positive patients. We remove non-inpatient patients who have a greater portion of missing data.  


```r
# Select admitted patients only
covid_status <- data %>%
                filter(floor == "TRUE" | step == "TRUE" | icu == "TRUE") 

#recode admission and COVID status columns
covid_status$admission <- base::ifelse(covid_status$floor == "TRUE", "1", 
                               base::ifelse(covid_status$step == "TRUE", "2",
                                            base::ifelse(covid_status$icu == "TRUE", "2", "0"))) %>% base::as.numeric()
covid_status$COVID <- base::ifelse(covid_status$COVID == "positive", "1","0")

# remove the old floor/step/icu columns, then reorder the columns to put covid target first
covid_status <- covid_status %>% 
                dplyr::select(-floor, 
                       -step,
                       -icu,
                       -patient_id) %>%
                dplyr::select(COVID, 
                       everything())


# Subset into covid stratified training and testing sets
set.seed(333)
split_status  <- initial_split(covid_status, 
                        prop = 0.7, 
                        strata = "COVID")

train_s  <- training(split_status)
test_s   <- testing(split_status)
```


Making slight adjustments to our recipe to target COVID status. Adding an additional imputation step using bagged decision trees to address missing data. 

  


## PCA {.tabset .tabset-fade}

Principle Component Analysis is a method to reduce feature space while allowing the greatest amount of variability or information to be explained. By examining covariance across features they are combined into multiple uncorrelated subsets called principal components (PCs). The eigenvector that corresponds to the largest eigenvalue explains the greatest proportion of feature variability. The greatest covariance is the line that minimizes the total squared distance from each point to its orthagonal projection onto a line.





### Variance Explained / Scree 

Proportion of Variance Explained (PVE) identifies the optimal number of PC's based on the variability explained. Here, PVE (aka Scree plot) and CVE (Cumulative variance explained) are plotted. 

![](README_files/figure-html/unnamed-chunk-20-1.png)<!-- -->

### Eigenvalue Criterion 

The sum of eigenvalues is equal to the number of variables entered into the PCA. eig of 1 means the pc would explain one features worth of variability. Therefore we look for the PC's which have a eigenvalue sum greater than 1. In this case, PC's 1-17. 


```
##  [1]  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
```


### Biplot

Rework using ggbiplot




## Model {.tabset .tabset-fade}

### Build


```r
# protrain_s <- bake(mrna_s, new_data = train_s)
# protest_s <- bake(mrna_s, new_data = test_s)

train_pca <- data.frame(covid = protrain_s$COVID_X1, pca$x)

#transform preprocessed test dataframe into PCA
test_pca <- predict(pca, newdata = protest_s)
test_pca <- as.data.frame(test_pca)
test_pca <- data.frame(covid = protest_s$COVID_X1, test_pca)

# limit to only the first 16 PC's (per eigenvalue criterion)
train_pca_final <- train_pca[,1:17]
test_pca_final <- test_pca[,1:17]

# create xgb matrix 
X_train_pca <- as.matrix(train_pca_final[setdiff(names(train_pca_final), "covid")])
Y_train_pca <- train_pca_final$covid
X_test_pca <- as.matrix(test_pca_final[setdiff(names(test_pca_final), "covid")])
Y_test_pca <- test_pca_final$covid


xgb_grid_pca = expand.grid(
                  nrounds = 1000,
                  eta = c(0.1), #0.05, 0.01), # limited for processing brevity
                  max_depth = c(2, 3, 4, 5, 6, 7, 8),
                  gamma = 0,
                  colsample_bytree=1,
                  min_child_weight=c(1, 2, 3, 4 ,5),
                  subsample=1)

control_pca <- trainControl(method ="cv", 
                            number = 5)


set.seed(333)
xgb_caret_pca <- train(x=X_train_pca, 
                   y=as.factor(Y_train_pca), 
                   method ='xgbTree',
                   metric ="Accuracy",
                   objective = "binary:logistic",
                   #nthreads = 6,
                   trControl = control_pca,
                   tuneGrid = xgb_grid_pca
                   ) 

# Best tune parameters from hyper parameter grid search
#xgb_caret_pca$results
xgb_caret_pca$bestTune
```

```
##   nrounds max_depth eta gamma colsample_bytree min_child_weight subsample
## 2    1000         2 0.1     0                1                2         1
```


### Design

![](README_files/figure-html/unnamed-chunk-24-1.png)<!-- -->


## Performance / Outcome

Performance results in an accuracy of 88%, sensitivity (recall) of 94%.


```r
# prediction on the test set, with no parameters to add or adjust I'm just using the best tune selected parameters:
test_pred_pca <- predict(xgb_caret_pca, X_test_pca)
caret::confusionMatrix(test_pred_pca, as.factor(Y_test_pca))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  0  1
##          0 33  4
##          1  2 11
##                                           
##                Accuracy : 0.88            
##                  95% CI : (0.7569, 0.9547)
##     No Information Rate : 0.7             
##     P-Value [Acc > NIR] : 0.002494        
##                                           
##                   Kappa : 0.703           
##                                           
##  Mcnemar's Test P-Value : 0.683091        
##                                           
##             Sensitivity : 0.9429          
##             Specificity : 0.7333          
##          Pos Pred Value : 0.8919          
##          Neg Pred Value : 0.8462          
##              Prevalence : 0.7000          
##          Detection Rate : 0.6600          
##    Detection Prevalence : 0.7400          
##       Balanced Accuracy : 0.8381          
##                                           
##        'Positive' Class : 0               
## 
```






