#Correlation test

#Spearman correlation was done in python and now we test correlation between high
#correlated parameters of real patients
DT <- read.csv(file.choose(),sep = ";")
attach(DT)

cor.test(DT$rho1,DT$alpha2,method = "spearman")
cor.test(DT$alpha3,DT$alpha4,method = "spearman")
cor.test(DT$rho2,DT$rho3,method = "spearman")

regm <- lm(df$rho2 ~ df$rho3)
summary(regm)

#Pearson correlation was done in python and now we test correlation between high
#correlated parameters of real patients
DT2 <- read.csv(file.choose(),sep = ",")
attach(DT2)

cor.test(DT2$rho1,DT2$a2,method = "pearson")
cor.test(DT2$a3,DT2$a4,method = "pearson")
cor.test(DT2$rho2,DT2$rho3,method = "pearson")

# fonction importation
source(file.choose())
# rho1/alpha2
cor.dif.test(0.819, 0.821, 200)
# alpha3/alpha4
cor.dif.test(0.936, 0.954, 200)
# rho2/rho3
cor.dif.test(0.885, 0.928, 200)






volume <- c(54, 47, 48, 170, 190, 50)
doses <- c(124, 80, 74, 135, 40, 130)
shapiro.test(doses)
mean(doses)
sd(doses)
hist(doses)

cor.test(doses, volume)
cor(doses, volume)
