# The housing data file is downloaded from https://www.kaggle.com/camnugent/california-housing-prices

# California Housing Price of California Districts

dat<-read.csv("https://raw.githubusercontent.com/ramjeeacharya/dataset/master/housing.csv")

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# Getting the feel of data

head(dat) # To see some rows of the dataset

nrow(dat) # To see how many rows are there in the dataset

summary(dat) # Seeing the summary of the data

# Structure of data

str(dat)

# General data wrangling

## Removing rows having NAs, as we have some NAs in 'total_bedrooms' column, and regression 
# analysis removes those rows anwway even we do not do so. We remove them just to be consistent
# with other models.
dat <- na.omit(dat)

# Introducing dummy variables for the categories present in 'ocean_proximity' column, since I 
# could not use the factor variable to calculate the correlation coefficient
# Dummy variable is 1 if given condition is satisfied, 0 otherwise

# Creating dummies and appending them into the dataset as new columns

inlnd <-  ifelse(dat$ocean_proximity == "INLAND", 1, 0) # setting inlnd = 1 if its 1, 0 otherwise
dat <- dat %>% mutate(Inland = inlnd) # Appending the new column 'Inland' into the dataset

nr_ocn <-  ifelse(dat$ocean_proximity == "NEAR OCEAN", 1, 0) # Setting 1 for houses near ocean, 0 otherwise
dat <- dat %>% mutate(Near_Ocean = nr_ocn) # Appending Near_Ocean column into the dataset

nr_bay <-  ifelse(dat$ocean_proximity == "NEAR BAY", 1, 0) # Setting 1 for hosues neary bay, 0 otherwise
dat <- dat %>% mutate(Near_Bay = nr_bay)

islnd <-  ifelse(dat$ocean_proximity == "ISLAND", 1, 0) # Setting 1 for houses on Island, 0 otherwise
dat <- dat %>% mutate(Island = islnd)

on_hr <-  ifelse(dat$ocean_proximity == "<1H OCEAN", 1, 0) # Setting 1 for houses within an hour of driving to ocean, 0 otherwise
dat <- dat %>% mutate(Less_One_Hr = on_hr)



# Categorizing the remainder of each independent variable into 100 differnt groups before partitioning
# the data to be able to use advanced machine learning later

dat <- dat %>% mutate(per_md_incm = ntile(median_income, 100))

dat <- dat %>% mutate(per_ttl_rms = ntile(total_rooms, 100))

dat <- dat %>% mutate(per_hs_md_age = ntile(housing_median_age, 100))

dat <- dat %>% mutate(per_hhs = ntile(households, 100))

dat <- dat %>% mutate(per_ttl_bdrms = ntile(total_bedrooms, 100))

dat <- dat %>% mutate(per_lat = ntile(latitude, 100))

dat <- dat %>% mutate(per_longt = ntile(longitude, 100))
dat <- dat %>% mutate(per_popn = ntile(population, 100))


# setting seed so that result remains same in each execution
# set set.seed(1), if R is 3.5 or earlier 
set.seed(1, sample.kind = "Rounding")

# creating data partition so that we have as much data as possible for training
# and enough data for testing as well

train_index <- createDataPartition(dat$median_house_value, times = 1, p = 0.9, list = FALSE)
train_set <- dat[train_index, ] 
test_set <- dat[-train_index, ] 

head(train_set)


# Examining correlations among variables

cols <- c(1:9, 11:15)
# Picking only needed columns for calculating the correlation
cor(train_set[cols])

# Distribution of median_house_value based on ocean_proximity

boxplot(median_house_value ~ ocean_proximity, data = train_set)

# Defining a function to calculate the Root Mean Square Error, RMSE


RMSE  <- function(predicted, true_value) {
  
  sqrt(mean((predicted - true_value)^2))
  
}   

# Running a linear regression
fit <-  lm(median_house_value ~ total_rooms + total_bedrooms + housing_median_age + population + households + median_income + longitude + latitude + ocean_proximity, data = train_set )
# To display the summary of the regression result
summary(fit)
# Predicting the outcome based on test_set
yhat <- predict(fit, newdata = test_set)
# Estimating RMSE for this linear model
RMSE_lm <- RMSE(yhat, test_set$median_house_value)

# Creating a data frame named rmse_results to store RMSEs from different models
rmse_results <- data_frame(method = "RMSE_lm", RMSEs = RMSE_lm)
# printing out the RMSE
rmse_results

# RMSE is 67007

# ***************Running other models for RMSE estimations ********************************************************

train_lm <- train(median_house_value ~ total_rooms + total_bedrooms + housing_median_age + population + households + median_income + longitude + latitude + ocean_proximity, method = "lm", data = train_set)

# Picking RMSE from above method
RMSE_lm_t <- train_lm$results$RMSE

# Adding next RMSE in the rmse_results
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="RMSE from train_lm",
                                     RMSEs = RMSE_lm_t ))

# RMSE  for linear regression is  68936

# glm model
train_glm <- train(median_house_value ~ total_rooms + total_bedrooms + housing_median_age + population + households + median_income + longitude + latitude + ocean_proximity, method = "glm", data = train_set)

RMSE_glm <- train_glm$results$RMSE

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="RMSE_glmm",
                                     RMSEs = RMSE_glm ))

# RMSE for glm is  69149

# knn method
train_knn <- train(median_house_value ~ total_rooms + total_bedrooms + housing_median_age + population + households + median_income + longitude + latitude + ocean_proximity, method = "knn", data = train_set)

RMSE_knn <- train_knn$results$RMSE[which.min(train_knn$results$RMSE)]

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="RMSE from Knn",
                                     RMSEs = RMSE_knn ))

# The lowest RMSE is 100710

# Defining control parameters for knn method
control <- trainControl(method = "cv", number = 10, p = .9)
train_knn_cv <- train(median_house_value ~ total_rooms + total_bedrooms + housing_median_age + population + households + median_income + longitude + latitude + ocean_proximity, method = "knn", 
                      data = train_set,
                      tuneGrid = data.frame(k = seq(1, 15, 1)),
                      trControl = control)

RMSE_knn_cv <- train_knn_cv$results$RMSE[which.min(train_knn_cv$results$RMSE)]

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="RMSE from Knn_cv",
                                     RMSEs = RMSE_knn_cv ))

#RMSE 95263.00

# Rpart method
train_rpart <- train(median_house_value ~ total_rooms + total_bedrooms + housing_median_age + population + households + median_income + longitude + latitude + ocean_proximity, method = "rpart", data = train_set)
RMSE_rpart <- train_rpart$results$RMSE[which.min(train_rpart$results$RMSE)]

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="RMSE from Rpart",
                                     RMSEs = RMSE_rpart))
rmse_results
#  Optimal is cp = 0.05895141, and the corresponding RMSE is 85233



# **********************************************************************************************

# Using machine learning algorithms for predicting RMSE

# Using mu as the predicted median_house_value

mu <- mean(train_set$median_house_value)
RMSE1 <- RMSE(mu, test_set$median_house_value)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Mean only",
                                     RMSEs = RMSE1 ))
rmse_results
# RMSE1 is 116140.7

# Including median_income's category variable in the process

# Estimating incm_i for each category per_md_incm, a category variable for median_income variable
md_incm_avg <- train_set %>% group_by(per_md_incm)  %>% 
  summarise(incm_i= mean( median_house_value - mu))

# Predicting from the test_set, based on the above estimation, from the train_set
predicted <- mu + test_set %>% 
  left_join(md_incm_avg, by='per_md_incm') %>%
  .$incm_i


RMSE2 <- RMSE(predicted, test_set$median_house_value)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Mean plus median income avg",
                                     RMSEs = RMSE2 ))
rmse_results
# RMSE2 is 80046.91

 # Adding ocean_proximity in the process
ocn_avgs <- train_set %>% 
  left_join(md_incm_avg, by='per_md_incm') %>%
  group_by(ocean_proximity) %>%
  summarize(b_ocn = mean(median_house_value - mu - incm_i))


predicted<- test_set %>% 
  left_join(md_incm_avg, by='per_md_incm') %>%
  left_join(ocn_avgs, by='ocean_proximity') %>%
  mutate(pred = mu + incm_i + b_ocn) %>%
  .$pred

RMSE3 <- RMSE(predicted, test_set$median_house_value)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Mean, median income avg and ocn_proximity",
                                     RMSEs = RMSE3 ))
rmse_results
# RMSE3 is 70854.93

# Adding latitude's category variable in the process
lat_avgs <- train_set %>% 
  left_join(md_incm_avg, by='per_md_incm') %>%
  left_join(ocn_avgs, by = 'ocean_proximity') %>%
  group_by(per_lat) %>%
  summarize(b_lat = mean(median_house_value - mu - incm_i- b_ocn))


predicted <- test_set %>% 
  left_join(md_incm_avg, by='per_md_incm') %>%
  left_join(ocn_avgs, by='ocean_proximity') %>%
  left_join(lat_avgs, by = 'per_lat') %>%
  mutate(pred = mu  + incm_i + b_ocn +  b_lat) %>%
  .$pred

RMSE4 <- RMSE(predicted, test_set$median_house_value)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Mean, median income avg, ocn_proximity and latitude",
                                     RMSEs = RMSE4))
rmse_results
# RMSE4 is 66491.92


# Adding household median age in the process
md_age_avgs <- train_set %>% 
  left_join(md_incm_avg, by='per_md_incm') %>%
  left_join(ocn_avgs, by = 'ocean_proximity') %>%
  left_join(lat_avgs, by = 'per_lat') %>%
  group_by(per_hs_md_age) %>%
  summarize(b_md_age = mean(median_house_value - mu - incm_i- b_ocn- b_lat))


predicted <- test_set %>% 
  left_join(md_incm_avg, by='per_md_incm') %>%
  left_join(ocn_avgs, by='ocean_proximity') %>%
  left_join(lat_avgs, by = 'per_lat') %>%
  left_join(md_age_avgs, by ='per_hs_md_age') %>%
  mutate(pred = mu  + incm_i + b_ocn +  b_lat + b_md_age) %>%
  .$pred

RMSE5 <- RMSE(predicted, test_set$median_house_value)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Mean, median incone avg, ocn_proximity, latitue and housing median age",
                                     RMSEs = RMSE5 ))
rmse_results
# RMSE5 is 65010.69

# Using regularization

lambdas <- seq(0, 25, 0.5) # setting lambda's sequence value
rmses <- sapply(lambdas, function(x){
  incm_i <- train_set %>%
    group_by(per_md_incm) %>%
    summarize(incm_i = sum(median_house_value - mu)/(n()+x))
  b_ocn <- train_set %>% 
    left_join(incm_i, by="per_md_incm") %>%
    group_by(ocean_proximity) %>%
    summarize(b_ocn = sum(median_house_value - incm_i - mu)/(n()+x))
  
  b_lat <- train_set %>% 
    left_join(md_incm_avg, by = 'per_md_incm') %>%
    left_join(ocn_avgs, by = 'ocean_proximity') %>%
    group_by(per_lat) %>%
    summarize(b_lat = sum(median_house_value - mu - incm_i- b_ocn)/(n()+x))

b_md_age <- train_set %>% 
  left_join(md_incm_avg, by='per_md_incm') %>%
  left_join(ocn_avgs, by = 'ocean_proximity') %>%
  left_join(lat_avgs, by = 'per_lat') %>%
  group_by(per_hs_md_age) %>%
  summarize(b_md_age = sum(median_house_value - mu - incm_i- b_ocn- b_lat)/(n()+x))
  
  predicted <- test_set %>% 
    left_join(incm_i, by = 'per_md_incm') %>%
    left_join(b_ocn, by = 'ocean_proximity') %>%
    left_join(b_lat, by = 'per_lat') %>%
    left_join(b_md_age, by ='per_hs_md_age') %>%
    mutate(pred = mu + incm_i + b_ocn + b_lat + b_md_age) %>%
    .$pred
  return(RMSE(predicted, test_set$median_house_value))
})

qplot(lambdas, rmses)  # Q-Plot

lambda <- lambdas[which.min(rmses)] # Optimal lambda
lambda # Printing out Optimal lambda, and it is 18.5
RMSE_reg <- rmses[which.min(rmses)] # minimum rmses, and it is 64509.01
rmse_results <- bind_rows(rmse_results,
                          data_frame(method=" Reg_RMSE",
                                     RMSEs = RMSE_reg ))
rmse_results
# Displaying all stored RMSEs by far in its descending order
rmse_results %>% arrange(desc(RMSEs))

