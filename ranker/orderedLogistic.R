library(MASS)
library(ggplot2)
individual_data <- read.table('/Users/cmaclell/Projects/pairwise/ranker/individual.csv', header=T, sep=",", quote="\"", check.names=F)
individual_data$rating_factor <- as.factor(individual_data$rating_value)
individual_data$judge_id <- as.factor(individual_data$judge_id)
individual_data$item_id <- as.factor(individual_data$item_id)

m1 = lm(rating_value ~ judge_id + item_id,data=individual_data)
AIC(m1)
summary(m1)
coef_m1 <- coefficients(m1)[22:118]

m2 = polr(rating_factor ~ judge_id + item_id,data=individual_data, Hess=TRUE)
AIC(m2)
summary(m2)
coef_m2 <- coefficients(m2)[21:117]

table(individual_data$rating_factor)
ggplot(individual_data) + geom_boxplot(aes(y=item_id, x=rating_factor))
plot(coef_m1, coef_m2)

