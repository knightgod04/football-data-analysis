############################################################
# Project: Football Data Analysis
# File: Manchester United Performance Analysis
#
# Description:
# This script contains the complete statistical analysis
# of Manchester United in the post-Ferguson era.
#
# Methods Used:
# - Exploratory Data Analysis
# - Welch’s t-test
# - Shapiro–Wilk normality test
# - Two-way ANCOVA (xG, xGA)
#
# Tools: R
# Author: Anubhav Roy
############################################################


# Install and load required libraries
install.packages("readxl")
install.packages("ggplot2")
library(readxl)
library(ggplot2)

# Load data and remove the first row
df <- read_excel("Football.xlsx", sheet = "Home XG ALL")[-1, ]

# Extract United and City columns
united_cols <- grep("United", names(df), value = TRUE)
city_cols <- grep("City", names(df), value = TRUE)

# Actual season names (adjust if needed)
season_names <- c("2017-18", "2018-19", "2019-20", "2020-21",
                  "2021-22", "2022-23", "2023-24")

# Initialize empty data frame
xg_data <- data.frame()

# Loop through each season
for (i in seq_along(season_names)) {
  xg_data <- rbind(xg_data, data.frame(
    xG = as.numeric(c(df[[united_cols[i]]], df[[city_cols[i]]])),
    Team = rep(c("United", "City"), each = nrow(df)),
    Season = season_names[i]
  ))
}

# Plot
ggplot(xg_data, aes(x = Season, y = xG, fill = Team)) +
  geom_boxplot() +
  scale_fill_manual(values = c("skyblue", "red")) +
  theme_minimal() +
  labs(title = "Home xG by Season", y = "Expected Goals (xG)", x = "Season") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Install and load required libraries
install.packages("readxl")
install.packages("ggplot2")
library(readxl)
library(ggplot2)

# Load data and remove the first row
df1 <- read_excel("Football.xlsx", sheet = "Home xGA ALL")

# Extract United and City columns
united_cols <- grep("United", names(df), value = TRUE)
city_cols <- grep("City", names(df), value = TRUE)

# Actual season names (adjust if needed)
season_names <- c("2017-18", "2018-19", "2019-20", "2020-21",
                  "2021-22", "2022-23", "2023-24")

# Initialize empty data frame
xga_data <- data.frame()

# Loop through each season
for (i in seq_along(season_names)) {
  xga_data <- rbind(xga_data, data.frame(
    xGA = as.numeric(c(df1[[united_cols[i]]], df1[[city_cols[i]]])),
    Team = rep(c("United", "City"), each = nrow(df1)),
    Season = season_names[i]
  ))
}

# Plot
ggplot(xg_data, aes(x = Season, y = xGA, fill = Team)) +
  geom_boxplot() +
  scale_fill_manual(values = c("skyblue", "red")) +
  theme_minimal() +
  labs(title = "Home xG by Season", y = "Expected Goals Against (xG)", x = "Season") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Install and load required libraries
install.packages("readxl")
install.packages("ggplot2")
library(readxl)
library(ggplot2)

# Load data and remove the first row
home_xga_raw <- read_excel("Football.xlsx", sheet = "Home xGA ALL")[-1, ]

# Extract United and City columns
united_cols <- grep("United", names(home_xga_raw), value = TRUE)
city_cols <- grep("City", names(home_xga_raw), value = TRUE)

# Actual season names (adjust if needed)
season_names <- c("2017-18", "2018-19", "2019-20", "2020-21",
                  "2021-22", "2022-23", "2023-24", "2024-25")

# Initialize empty data frame
xga_data <- data.frame()

# Loop through each season
for (i in seq_along(season_names)) {
  xga_data <- rbind(xga_data, data.frame(
    xGA = as.numeric(c(home_xga_raw[[united_cols[i]]], home_xga_raw[[city_cols[i]]])),
    Team = rep(c("United", "City"), each = nrow(home_xga_raw)),
    Season = season_names[i]
  ))
}

# Plot
ggplot(xga_data, aes(x = Season, y = xGA, fill = Team)) +
  geom_boxplot() +
  scale_fill_manual(values = c("skyblue", "red")) +
  theme_minimal() +
  labs(title = "Home xGA by Season", y = "Expected Goals Against (xGA)", x = "Season") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Load required libraries
library(readxl)
library(ggplot2)

# Load data from "Away xG ALL" and remove first row
away_xg_raw <- read_excel("Football.xlsx", sheet = "Away xG ALL")[-1, ]

# Get columns
united_cols <- grep("United", names(away_xg_raw), value = TRUE)
city_cols <- grep("City", names(away_xg_raw), value = TRUE)

# Season names
season_names <- c("2017-18", "2018-19", "2019-20", "2020-21",
                  "2021-22", "2022-23", "2023-24", "2024-25")

# Combine data
away_xg_data <- data.frame()
for (i in seq_along(season_names)) {
  away_xg_data <- rbind(away_xg_data, data.frame(
    xG = as.numeric(c(away_xg_raw[[united_cols[i]]], away_xg_raw[[city_cols[i]]])),
    Team = rep(c("United", "City"), each = nrow(away_xg_raw)),
    Season = season_names[i]
  ))
}

# Plot
ggplot(away_xg_data, aes(x = Season, y = xG, fill = Team)) +
  geom_boxplot() +
  scale_fill_manual(values = c("skyblue", "red")) +
  theme_minimal() +
  labs(title = "Away xG by Season", y = "Expected Goals (xG)", x = "Season") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


print(names(away_xg_data))




# Load data from "Away xGA ALL" and remove first row
away_xga_raw <- read_excel("Football.xlsx", sheet = "Away xGA ALL")[-1, ]

# Get columns
united_cols <- grep("United", names(away_xga_raw), value = TRUE)
city_cols <- grep("City", names(away_xga_raw), value = TRUE)

# Combine data
away_xga_data <- data.frame()
for (i in seq_along(season_names)) {
  away_xga_data <- rbind(away_xga_data, data.frame(
    xGA = as.numeric(c(away_xga_raw[[united_cols[i]]], away_xga_raw[[city_cols[i]]])),
    Team = rep(c("United", "City"), each = nrow(away_xga_raw)),
    Season = season_names[i]
  ))
}

# Plot
ggplot(away_xga_data, aes(x = Season, y = xGA, fill = Team)) +
  geom_boxplot() +
  scale_fill_manual(values = c("skyblue", "red")) +
  theme_minimal() +
  labs(title = "Away xGA by Season", y = "Expected Goals Against (xGA)", x = "Season") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


library("readxl")
data = read_excel("Football.xlsx", sheet="Average Poss ALL")
United_avg=data$`Manchester United`
City_avg=data$`Manchester City`
shapiro.test(United_avg)
shapiro.test(City_avg)


t.test(City_avg, United_avg, alternative = "greater", var.equal = FALSE)



library("readxl")
data1 = read_excel("Football.xlsx", sheet="Average SA (FOR) ALL")
United_avg1=data1$`Manchester United`
City_avg1=data1$`Manchester City`
shapiro.test(United_avg1)
shapiro.test(City_avg1)

t.test(City_avg1, United_avg1, alternative = "greater", var.equal = FALSE)


library("readxl")
data2 = read_excel("Football.xlsx", sheet="Average SA (AGAINST) ALL")
United_avg2=data2$`Manchester United`
City_avg2=data2$`Manchester City`
shapiro.test(United_avg2)
shapiro.test(City_avg2)

t.test(City_avg2, United_avg2, alternative = "greater", var.equal = FALSE)


library("readxl")
data3 = read_excel("Football.xlsx", sheet="Average PA (FOR) ALL")
United_avg3=data3$`Manchester United`
City_avg3=data3$`Manchester City`
shapiro.test(United_avg3)
shapiro.test(City_avg3)

t.test(City_avg3, United_avg3, alternative = "greater", var.equal = FALSE)


library("readxl")
data4 = read_excel("Football.xlsx", sheet="Average PA (AGAINST) ALL")
United_avg4=data4$`Manchester United`
City_avg4=data4$`Manchester City`
shapiro.test(United_avg4)
shapiro.test(City_avg4)

t.test(City_avg4, United_avg4, alternative = "greater", var.equal = FALSE)




library("readxl")
data = read_excel("Football.xlsx", sheet="Average Poss ALL")
United_avg=data$`Manchester United`
City_avg=data$`Manchester City`
var.test(United_avg,City_avg)
var.test(United_avg,City_avg,alternative = "greater")






library("readxl")
data5 = read_excel("Football.xlsx", sheet="Formations Managers")
# Shapiro-Wilk test of normality within each Manager group
library(dplyr)

data5 %>%
  group_by(Formation) %>%
  summarise(p_value = shapiro.test(Count)$p.value)

# Levene's Test for equal variances across Managers
library(car)
leveneTest(Count ~ Formation, data = data5)



library("readxl")
data6= read_excel("Football.xlsx",sheet = "Manager xG")
str(data6)
data6$xG <- as.numeric(as.character(data$xG))
str(data6)
data6$xG=as.numeric(data6$xG)
str(data6)



# Perform the Kruskal-Wallis test
kruskal_result <- kruskal.test(xG ~ Manager, data = data6)

# View the results
print(kruskal_result)




library("readxl")
data7 = read_excel("Football.xlsx",sheet = "Manager xGA")
str(data7)
data6$xG <- as.numeric(as.character(data$xG))
str(data6)
data6$xG=as.numeric(data6$xG)
str(data6)



# Perform the Kruskal-Wallis test
kruskal_result <- kruskal.test(xGA ~ Manager, data = data7)

# View the results
print(kruskal_result)



# Install if you haven't already
install.packages("FSA")

# Load the package
library(FSA)

# Run Dunn test with Bonferroni correction
dunnTest(xGA ~ Manager, data = data7, method = "bonferroni")




library("readxl")
data8 = read_excel("Football.xlsx",sheet = "Average Poss United")
Possession = data8$`Average Possession`
Seasons= data8$Seasons


Possession
# Step 2: Line plot
plot(Seasons, Possession, type="o", col="blue", xlab="Season", ylab="Average Possession (%)",
     main="Manchester United's Avg Possession (2017–24)")


acf(Possession)



# Fit AR(1) and AR(2) using Yule-Walker method
ar1 <- ar(Possession, aic = FALSE, order.max = 1, method = "yw")
ar2 <- ar(Possession, aic = FALSE, order.max = 2, method = "yw")

# View model summaries
ar1
ar2

# Compare models using AIC
ar(Possession, method = "yw")  # This automatically chooses best AR order based on AIC


# Install and load the package if needed
install.packages("tseries")
library(tseries)

# Run the ADF test
adf.test(Possession)
Possession
?adf.test
# Create differenced series
Possession_diff <- diff(Possession)

# Check ADF again
adf.test(Possession_diff)


library(readxl)
data9=read_excel("Football.xlsx",sheet="MLR")
data9

xG=data9$xG
Possession1=data9$Possession
Pass=data9$`Passing Accuracy`
Shot=data9$`Shooting Accuracy`
GS=data9$`Goals Scored`
GC=data9$`Goals Conceded`
xGA=data9$xGA


model=lm(xG ~ Possession1 + Pass + Shot + GS + GC + xGA ,data = data9)
summary(model)



model1=lm(xG ~ Possession1 + Shot + xGA ,data = data9)
summary(model1)




# Plot standard diagnostics
par(mfrow = c(2, 2))
plot(model1)


hist(residuals(model1), main = "Histogram of Residuals",
     xlab = "Residuals", col = "lightblue", border = "black")


shapiro.test(residuals(model1))


# Install once if not already installed
install.packages("corrplot")

# Load the library
library(corrplot)


library(readxl)
data9=read_excel("Football.xlsx",sheet="MLR")
data9

xG=data9$xG
Possession1=data9$Possession
Pass=data9$`Passing Accuracy`
Shot=data9$`Shooting Accuracy`
GS=data9$`Goals Scored`
GC=data9$`Goals Conceded`
xGA=data9$xGA


# Step 3: Correlation matrix (Pearson by default)
cor_mat <- cor(data9, method = "pearson")
print(round(cor_matrix, 2))  # View the correlation values (rounded)

str(data9)
data9= subset(data9,select=-Seasons)
str(data9)

cor_mat <- cor(data9, method = "pearson")
print(round(cor_mat, 2))  # View the correlation values (rounded)




?corrplot

corrplot(cor_mat,method="color",type="upper",tl.col="black",tl.srt=45,addCoef.col = "black",number.cex = 0.8,col=colorRampPalette(c("red","white","blue"))(200),title="Correlation Heatmap of Team Performance Metrics",mar=c(0,0,1,0))


library(readxl)
data10= read_excel("Football.xlsx",sheet="Manager ANOVA")
data10
seasons=data10$Seasons
Manager=data10$Manager
avgxG=data10$`Avg xG`

model_anova=aov(avgxG ~ Manager, data=data10)

summary(model_anova)


shapiro.test(avgxG)
library(car)
leveneTest(avgxG ~ Manager, data=data10)


library(readxl)
data11= read_excel("Football.xlsx",sheet="Manager Summary")
data11

# Step 3: Convert to matrix
data_matrix <- as.matrix(data11[, c("Wins", "Draws", "Losses")])
rownames(data_matrix) <- data11$Manager
data_chi <- as.table(data_matrix)

data_chi
chisq.test(data_chi)





install.packages("readxl")
library(readxl)

formation_data <- read_excel("Football.xlsx",sheet="Formation Data")
head(formation_data)

# Frequency table of formations
formation_freq <- table(formation_data$Formation)
print(formation_freq)

# Convert to proportions
formation_prop <- prop.table(formation_freq) * 100
print(round(formation_prop, 2))


# Barplot of Formation Usage
barplot(formation_freq,
        main = "Formation Usage Across 7 Seasons",
        xlab = "Formation",
        ylab = "Number of Matches",
        col = "skyblue",
        las = 2)  # Rotate x labels for better readability



# Cross-tab of Formation vs Result
formation_result_table <- table(formation_data$Formation, formation_data$Results)
print(formation_result_table)

# (Optional) Chi-square Test (only if enough counts)
chisq.test(formation_result_table)



library(readxl)
data12=read_excel("Football.xlsx",sheet="xG xGA")
str(data12)

data12 <- data12[, -c(4, 5)]
str(data12)

a=data12$`xG - GF`
b=data12$`xGA-GA`

shapiro.test(a)

shapiro.test(b)




# Your data
a

# Known variance you are testing against
sigma0_sq <- 1  # remember to square sigma0

# Chi-square test statistic
n <- length(a)
s2 <- var(a)
test_stat <- (n - 1) * s2 / sigma0_sq

# p-value (two-tailed)
p_value <- 2 * min(
  pchisq(test_stat, df = n - 1),
  pchisq(test_stat, df = n - 1, lower.tail = FALSE)
)

# Print results
cat("Chi-square statistic:", test_stat, "\n")
cat("p-value:", p_value, "\n")



sigma0_squared=1
n <- length(a)
n
s2 <- var(a)   # sample variance
test_stat <- (n - 1) * s2 / sigma0_squared
p_value <- pchisq(test_stat, df = n - 1, lower.tail = TRUE)  # because we want "less than"
p_value



library(readxl)
data13=read_excel("Football.xlsx",sheet="Manager ANOVA 2")
data13
str(data13)

season1=data13$Seasons
manager=data13$Manager
avgxGA= data13$`Avg xGA`

shapiro.test(avgxGA)

library(car)
leveneTest(avgxGA ~ manager, data=data13)

model_anova1=aov(avgxGA ~ manager, data=data13)

summary(model_anova1)


library(readxl)
data14=read_excel("Football.xlsx",sheet="Point Biserial")

c=data14$xG
d=data14$Results
e=data14$xGA


# Point-Biserial Correlation: xG ~ Match Outcome
cor.test(d, c, method = "pearson")


# Point-Biserial Correlation: xGA ~ Match Outcome
cor.test(d, e, method = "pearson")


City_avg
United_avg

City_avg1
United_avg1
City_avgshot=City_avg1
United_avgshot=United_avg1


United_avgshot2=United_avg2
City_avgshot2=City_avg2

t.test(United_avgshot2,City_avgshot2,alternative = "greater",var.equal = FALSE)

t.test(City_avgshot,United_avgshot,alternative = "greater",var.equal = FALSE)

City_avg3
United_avg3

United_avgpass=United_avg3
City_avgpass=City_avg3



t.test(City_avgpass,United_avgpass,alternative = "greater",var.equal = FALSE)

City_avg4
United_avg4

United_avgpass2=United_avg4
City_avgpass2=City_avg4

t.test(United_avgpass2,City_avgpass2,alternative = "greater",var.equal = FALSE)


var.test(United_avg,City_avg,alternative = "greater")

var.test(United_avgpass,City_avgpass,alternative = "greater")

var.test(United_avgpass2,City_avgpass2,alternative = "less")

var.test(United_avgshot,City_avgshot,alternative = "greater")

var.test(United_avgshot2,City_avgshot2,alternative = "less")




var.test(United_avg,City_avg,alternative = "less")


var.test(United_avgpass,City_avgpass,alternative = "less")

var.test(United_avgpass2,City_avgpass2,alternative = "greater")






# Load necessary libraries
library(ggplot2)
library(car)       # for residualPlots and outlierTest
library(lmtest)    # for bptest
library(nortest)   # for Anderson-Darling or Shapiro-Wilk
library(readxl)
# Read data
install.packages("nortest")
data15=read_excel("Football.xlsx",sheet="xG xGA") 
str(data15)

data15=data15[,c(-4,-5)]
str(data15)
xG=data15$xG
GF=data15$GF
# Fit the model
model <- lm(GF ~ xG, data = data15)
summary(model)

library(car)
linearHypothesis(model, c("xG"))

library(lmtest)
dwtest(model)

library(lmtest)
bptest(model)


shapiro.test(residuals(model))

library(car)
outlierTest(model)

model <- glm(GF ~ xG, family = poisson(link = "log"), data = data15)


mean_val <- mean(data15$GF)
var_val <- var(data15$GF)
mean_val; var_val        # Compare these

# Statistical test for overdispersion:
install.packages("AER")
library(AER)
dispersiontest(model)    # p < 0.05 means overdispersion is present

library(car)
boxTidwell(GF ~ xG, data = data15)  # Assesses if the log link is appropriate


sum(data15$GF == 0)           # Count of zero outcomes
prop.table(table(data15$GF))  # See proportion of 0s

model
summary(model)

data15$pred_GF <- predict(model, type = "response")
plot(data15$xG, data15$GF, col = "blue", pch = 16)
points(data15$xG, data15$pred_GF, col = "red", pch = 16)
legend("topleft", legend=c("Actual GF", "Predicted GF"), col=c("blue", "red"), pch=16)














# Load MASS package
library(MASS)

# Fit Negative Binomial model
nb_model <- glm.nb(GF ~ xG, data = data15)

# View summary
summary(nb_model)

# Predict values
predicted_nb <- predict(nb_model, type = "response")

# Plot: Actual vs Predicted GF
plot(data15$xG, data15$GF, 
     col = "blue", pch = 20, 
     xlab = "data15$xG", ylab = "data15$GF")
points(data15$xG, predicted_nb, 
       col = "red", pch = 20)
legend("topleft", legend = c("Actual GF", "Predicted GF"), 
       col = c("blue", "red"), pch = 20)




library(readxl)
data16=read_excel("Football.xlsx",sheet="Log Reg")
str(data16)

install.packages("nnet")       # For multinom
install.packages("car")        # For VIF
install.packages("mlogit")     # Optional, for some data handling
install.packages("pscl")       # For Pseudo R-squared

library(nnet)
library(car)
library(pscl)

# Sample linear model just to calculate VIF
vif_model <- lm(as.numeric(Result) ~ xG + xGA + Possession , data = data16)
vif(vif_model)






library(readxl)
data17=read_excel("TIME SERIES.xlsx", sheet="Time Date")
data17
plot("Date","Y")
?plot
library(base)
plot(data17$Date,data17$Y,xlim=2013,ylim=900.23,xlab="Date",ylab="Value",col="red")
adf.test?
??adf.test



install.packages("forecast")
install.packages("tseries")

library(forecast)
library(tseries)
?adf.test

x=data17$Y
adf.test(x,alternative = "explosive")

?auto.arima
k=auto.arima(x)
k
summary(k)
plot(k)


a=forecast(k,h=12)
plot(a)


install.packages("rpart")
library(rpart)
v=data(iris)
summary(v)
v
library(rpart.plot)



install.packages("readxl")
library(readxl)
str(data3)

a=data3$`Manchester United`
b=data3$`Manchester City`

shapiro.test(a)
shapiro.test(b)

t.test(City_avgpass2,United_avgpass2,alternative = "less")
t.test(United_avgpass2,City_avgpass2,alternative = "greater")

install.packages("readxl")
library(readxl)
data12=read_excel("Football.xlsx",sheet = "xga")

excel_sheets("Football.xlsx")

library(openxlsx)
sheet_names <- getSheetNames("Football.xlsx")
print(sheet_names)
getwd(Football.xlsx)


str(data12)
e=data12$`xGA-GA`
test=wilcox.test(e,mu=0,alternative = "two.sided",exact = FALSE)
print(test)


test1=wilcox.test(e,mu=0,alternative = "less",exact = FALSE)
print(test1)


test2=wilcox.test(e,mu=0,alternative = "greater",exact = FALSE)
print(test2)
j=data12$`xG - GF`
test3=wilcox.test(j,mu=0,alternative = "less")
test3

shapiro.test(e)
shapiro.test(j)
str(data)


str(United_avg)
str(City_avg)

qqnorm(United_avg,main = "Q-Q Plots of Manchester United's Average Possession since 2017-18 season.")
qqline(United_avg,col="blue",lwd=4)

qqnorm(City_avg,main = "Q-Q Plots of Manchester City's Average Possession since 2017-18 season.")
qqline(City_avg,col="blue",lwd=4)

qqnorm(United_avg1,main = "Q-Q Plots of Manchester United's Average Shooting Accuracy since 2017-18 season")
qqline(United_avg1,col="red",lwd=4)

qqnorm(City_avg1,main = "Q-Q Plots of Manchester City's Average Shooting Accuracy since 2017-18 season")
qqline(City_avg1,col="red",lwd=4)

qqnorm(United_avg2,main = "Q-Q Plots of Manchester United's Average Shooting Accuracy (Against) since 2017-18 season")
qqline(United_avg2,col="darkgreen",lwd=4)

qqnorm(City_avg2,main = "Q-Q Plots of Manchester City's Average Shooting Accuracy (Against) since 2017-18 season")
qqline(City_avg2,col="darkgreen",lwd=4)


qqnorm(United_avg3,main = "Q-Q Plots of Manchester United's Average Passing Accuracy since 2017-18 season")
qqline(United_avg3,col="orange",lwd=4)

qqnorm(City_avg3,main = "Q-Q Plots of Manchester City's Average Passing Accuracy since 2017-18 season")
qqline(City_avg3,col="orange",lwd=4)

qqnorm(United_avg4,main = "Q-Q Plots of Manchester United's Average Passing (Against) Accuracy since 2017-18 season")
qqline(United_avg4,col="brown",lwd=4)

qqnorm(City_avg4,main = "Q-Q Plots of Manchester City's Average Passing (Against) Accuracy since 2017-18 season")
qqline(City_avg4,col="brown",lwd=4)

str(avgxG)
shapiro.test(avgxG)

qqnorm(avgxG,main = "Q-Q Plots of Average Expected Goals of Manchester United since 2017-18 season")
qqline(avgxG,col="purple",lwd=4)

qqnorm(avgxGA,main = "Q-Q Plots of Average Expected Goals Against of Manchester United since 2017-18 season")
qqline(avgxGA,col="purple",lwd=4)
shapiro.test(avgxGA)

qqnorm(j,main = "Q-Q Plots of Manchester United's Attacking Performance")
qqline(j,col="red",lwd=3)

qqnorm(e,main = "Q-Q Plots of Manchester United's Defensive Performance")
qqline(e,col="red",lwd=3)


xG
xGA=data12$xGA
xGA
Results=data14$Results

# If 'Result' is coded as 1 = Win, 0 = Loss/Draw
pairs(data.frame(xG, xGA, Results), 
      main = "Pairs Plot: xG, xGA vs Match Result", 
      pch = 21, bg = c("red", "green")[Results + 1])

library(ggplot2)
ggplot(data, aes(x = xG, fill = factor(Results))) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 30) +
  labs(title = "Distribution of xG by Match Result", fill = "Results") +
  theme_minimal()




ggplot(data, aes(x = Results, y = xG, fill = Results)) +
  geom_boxplot() +
  labs(title = "xG Distribution by Match Result", x = "Result", y = "xG") +
  theme_minimal()





library(ggplot2)

# Basic bar plot of mean xG per manager
ggplot(data1, aes(x = Manager, y = avgxG, fill = Manager)) +
  stat_summary(fun = mean, geom = "bar") +
  labs(title = "Average xG by Manager", y = "Mean xG") +
  theme_minimal()




library(ggplot2)

# Assuming you have avg xG for Wins and Non-Wins
ggplot(data, aes(x = Result, y = xG, fill = Result)) +
  geom_col() +
  labs(title = "Average xG by Match Outcome", y = "xG") +
  theme_minimal()


avgxG

data_chi


