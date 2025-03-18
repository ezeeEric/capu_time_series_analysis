##################################
# Enrollment Time Series Modelling
# v2025-01-22
##################################

# STEPS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. Install packages
# 2. Assign values to variables
# 3. Import and prepare data
# 4. Visualize data
# 5. Fit models to training set
# 6. Compare test set accuracy measures
# 7. Check training set residuals
# 8: Select the best model
# 9. Forecast future using the best model
# 10. (Optional Step) Compare forecasting results between candidate models
# 11. Export forecasting results to Excel
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~
# 1. Install packages
#~~~~~~~~~~~~~~~~~~~~

# install.packages("fpp2")
library(fpp2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2. Assign values to variables
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This step defines the level of analysis. Adjust as needed

levl <- "CapU"           # Level.     Options:  "CapU", "AS", "BPS", "EHHD", "FAA", "GCS"
resd <- "Domestic"       # Residency. Options:  "Domestic", "International"
mt <- "CourseEnrolment"  # Metric.    Options:  "Headcount", "CourseEnrolment", "AttemptedCredits"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3. Import and prepare data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Import data. Use forward slashes (/) for file paths

file_path <- ".../.../.../1_raw data_CapU and faculty enrolment 200820-202310_v20230213.csv"
df_raw <- read.csv(file_path, header=TRUE)

df_subset <- subset(df_raw, Level==levl & Residency==resd, select = c(1:4, which(colnames(df_raw)==mt) ))
df <- df_subset[order(df_subset$TermCode),]
View(df)

# Convert df into a time series object, and split it into training & test sets (80/20)
# Full dataset 15Yr h=45. Training 12Yr h=36. Test 3Yr h=9
df_start_year <- strtoi(substr(df$TermCode[1], 1, 4))
df_start_term <- strtoi(substr(df$TermCode[1], 5, 5))
train_end_year <- strtoi(substr(df$TermCode[36], 1, 4))
train_end_term <- strtoi(substr(df$TermCode[36], 5, 5))
test_start_year <- strtoi(substr(df$TermCode[37], 1, 4))
test_start_term <- strtoi(substr(df$TermCode[37], 5, 5))

ts_df <- ts(df[,5], start=c(df_start_year,df_start_term), frequency=3)   
train <- window(ts_df, end=c(train_end_year,train_end_term))
test <- window(ts_df, start=c(test_start_year,test_start_term))

#~~~~~~~~~~~~~~~~~~
# 4. Visualize data
#~~~~~~~~~~~~~~~~~~

# split the metric name by uppercase
mt_name <- gsub("(?!^)(?=[[:upper:]])", " ", mt, perl=T)

autoplot(train, color="blue", series="Training") + 
  forecast::autolayer(test, color="red", series="Test Set") + 
  ggtitle(paste(df$Level, df$Residency, mt_name, "", df$TermCode[1], "-", tail(df$TermCode, 1))) + 
  xlab("Year") + 
  ylab(mt_name) +
  scale_color_manual(values = c("blue", "red")) +  
  guides(colour=guide_legend(title="Data Set"))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 5. Fit models to training set
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Seasonal Naive
# Each forecast is equal to the last observation from the same season. Can serve as a benchmark for evaluating other advanced models (ETS and ARIMA).
fit.snaive <- snaive(train, h=length(test))

# ETS Model
# The ets() function auto-selects the best ETS model by optimizing parameters (minimizing AICc)
fit.ets <- ets(train) 
summary(fit.ets)

# ARIMA Model
# The auto.arima() function automatically selects the best Arima model by differencing, minimizing AICc, and MLE
fit.arima <- auto.arima(train, stepwise=FALSE, approximation=FALSE)
summary(fit.arima)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 6. Compare test set accuracy measures
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Accuracy measures can provide a quick comparison between models about how accurate the forecasts are. The forecast accuracy is only assessed for the test set

fit.ets.test <- forecast(fit.ets, h=length(test))
fit.arima.test <- forecast(fit.arima, h=length(test))

cbind(test, fit.ets.test$mean, fit.arima.test$mean, fit.snaive$mean)

autoplot(test, size=0.8, color="black") +
  autolayer(ts_df, color="black") + 
  forecast::autolayer(fit.ets.test$mean, series="Best ETS") +   
  forecast::autolayer(fit.arima.test$mean, series="Best ARIMA") + 
  forecast::autolayer(fit.snaive, series="Seasonal Naive", PI=FALSE) +
  ggtitle("Model Forecasts on Test Set") +
  xlab("Year") + ylab(mt_name) +
  guides(colour=guide_legend(title="Models"))

rbind(best.ets = accuracy(fit.ets.test, test)[2,c(2,3)],
      best.arima = accuracy(fit.arima.test, test)[2,c(2,3)],
      snaive = accuracy(fit.snaive, test)[2,c(2,3)])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 7. Check training set residuals
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This step ensures that the model fits the data well and that the residuals (errors) resemble white noise. 
# The checkresiduals() function performs several diagnostic tests to check if the residuals of the fitted model resemble white noise. 
# 1) Residual plot: A random scatter of residuals around zero would suggest that the model has captured the data well.
# 2) ACF plot: If there are no significant spikes, it means the residuals are uncorrelated, which is a good sign.
# 3) Ljung-Box Test: If the p-value is large (>=0.05), you can conclude that there’s no significant autocorrelation in the residuals.
# 4) Mean of residuals: Ideally, the mean should be close to zero. If it's significantly different from zero, the model might be biased. 
# Models that didn't pass all of the residual tests may still be used for forecasting but the prediction intervals may not be accurate due to the correlated residuals. In practice, we would normally use the best model we could find, even if it didn't pass all of the tests.

checkresiduals(fit.snaive)
mean(residuals(fit.snaive), na.rm=TRUE) 

checkresiduals(fit.ets)
mean(residuals(fit.ets), na.rm=TRUE)

checkresiduals(fit.arima)
mean(residuals(fit.arima), na.rm=TRUE)

#~~~~~~~~~~~~~~~~~~~~~~~~~
# 8: Select the best model
#~~~~~~~~~~~~~~~~~~~~~~~~~
# Based on the accuracy measures (on test set) and residual analysis (on training set), we will choose the best model. In general, the model with the lowest RMSE, MAE and MAPE values, and residuals that resemble white noise, will be the best model.
# Rules:
# 1) If the accuracy measures (RMSE, MAE) are consistent: Choose the model with the lowest values across all metrics.
# 2) If the accuracy measures (RMSE, MAE) give conflicting results, since RMSE penalizes larger errors heavier, prioritize this metric to avoid large forecasting errors. MAE treats all errors equally and is easier to understand.
# 3) ETS and ARIMA models are generally preferred over SNAIVE as they can capture both trend and seasonality. SNAIVE ignores trends.
# 4) Evaluate residuals: Regardless of the accuracy metrics, always check if the residuals show any patterns that indicate the model has not captured the underlying structure.

# Apply the previously fitted model parameters (based on the training set) to the full data set to forecast for the real future (in our case, the next 3 years)

# The auto ets() model is selected as the best model as it has the best accuracy on test set and performs well on residual tests. 
fit.ets.forecast <- forecast(ets(ts_df, model=fit.ets, use.initial.values=TRUE), h=length(test))

best.forecast <- fit.ets.forecast  # Replace the best model as needed
summary(best.forecast)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 9. Forecast future using the best model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

best.forecast  # .000=Spring, .333=Summer, .667=Fall. 

future_start <- paste(as.character(strtoi(substr(df$TermCode[1],1,4)) + 15), substr(df$TermCode[1],5,6), sep="")
future_end <- paste(as.character(strtoi(substr(df$TermCode[45],1,4)) + 3), substr(df$TermCode[45],5,6), sep="")

autoplot(best.forecast) + 
  ggtitle(paste(df$Level, df$Residency, mt_name, "Forecasts", future_start, "-", future_end)) + 
  xlab("Year") + 
  ylab(mt_name)
# The further ahead we forecast, the more uncertainty is associated with the forecast, and thus the wider the prediction intervals.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 10. (Optional Step) Compare forecasting results between candidate models
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This step may add an extra layer of validation/comparison to ensure that the best model produces reliable forecasts for the future

fit.arima.forecast <- forecast(Arima(ts_df, model=fit.arima), h=length(test))
fit.snaive.forecast <- snaive(ts_df, h=9)

cbind(best.ets = fit.ets.forecast$mean,
      best.arima = fit.arima.forecast$mean,
      snaive = fit.snaive.forecast$mean)

autoplot(ts_df) +
  forecast::autolayer(fit.ets.forecast$mean, series="Best ETS") +
  forecast::autolayer(fit.arima.forecast$mean, series="Best ARIMA") + 
  forecast::autolayer(fit.snaive.forecast, series="Seasonal Naive", PI=FALSE) +
  ggtitle("Forecasts for the Future 202320-202610") +
  xlab("Year") + ylab("Course Enrolment") +
  guides(colour=guide_legend(title="Forecast Model"))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 11. Export forecasting results to Excel
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# install.packages("dplyr")
# install.packages("openxlsx")
library(dplyr)      # for case_when()
library(openxlsx)   # for write.xlsx() and read.xlsx()

forecast_result <- data.frame(
  Level = levl,
  Residency = resd,
  Analysis_Type = mt_name,
  Term_Code = case_when(
                substr(time(best.forecast$mean), 6, 6) == "" ~ paste0(substr(time(best.forecast$mean), 1, 4), "10"),
                substr(time(best.forecast$mean), 6, 6) == 3 ~ paste0(substr(time(best.forecast$mean), 1, 4), "20"),
                substr(time(best.forecast$mean), 6, 6) == 6 ~ paste0(substr(time(best.forecast$mean), 1, 4), "30")
                ),
  Time = time(best.forecast$mean),
  Point_Forecast = best.forecast$mean,
  Lo_80 = best.forecast$lower[,1],
  Hi_80 = best.forecast$upper[,1],
  Lo_95 = best.forecast$lower[,2],
  Hi_95 = best.forecast$upper[,2],
  Model = best.forecast$method
)

# Path to the Excel output file. Use forward slashes (/) for file paths
file_path <- ".../.../.../3_output.xlsx"

# Check if the file exists: if yes, append data to the existing sheet; if not, create a new file
if (file.exists(file_path)) {
  wb <- loadWorkbook(file_path)
  existing_data <- read.xlsx(wb, sheet = "Sheet 1")
  start_row <- nrow(existing_data) + 1
  writeData(wb, sheet = "Sheet 1", forecast_result, startRow = start_row, startCol = 1, colNames = FALSE, rowNames = FALSE)
  saveWorkbook(wb, file_path, overwrite = TRUE)
} else {
  write.xlsx(forecast_result, file_path)
}


### Repeat the whole process for all levels of analysis (defined in Step 2) to generate a complete Excel output file.

