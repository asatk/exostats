library(dplyr)
library(ggplot2)

data = read.csv("~/code/rice/exostats/tables-merged/nasa_exo.csv")

data_hj = subset(data, (pl_bmassj > 0.25) & (pl_bmassj < 13) & (log10(pl_orbsmax) < 1.5))
rm(data)
colnames(data_hj)

hist(data_hj$sy_pnum)

ggplot(data=data_hj, aes(x=pl_orbsmax, fill=factor(sy_pnum))) + geom_histogram() + xlim(0, 100) + ylim(0,1000)

plot(log10(data_hj$pl_orbsmax), data_hj$st_age)
plot(data_hj$st_age, log10(data_hj$pl_orbsmax))
plot(data_hj$st_age, tanh(data_hj$pl_orbsmax))
plot(data_hj$pl_bmassj, tanh(data_hj$pl_orbsmax))
plot(data_hj$st_mass, tanh(data_hj$pl_orbsmax))
plot(data_hj$sy_snum, data_hj$pl_orbsmax)
plot(data_hj$sy_pnum, data_hj$pl_orbsmax)

model = glm(tanh(pl_orbsmax) ~ st_age, data=data_hj, family=binomial())
plot(model)

factor(data_hj$sy_snum)

fit = lm(pl_orbsmax ~ pl_bmassj + pl_radj + pl_orbeccen + pl_orbincl +
           factor(sy_snum) + factor(sy_pnum) +
           I(sy_vmag - sy_kmag) +
           st_mass + st_age + st_teff + st_rotp + st_met + st_rad +
           factor(tran_flag) + factor(rv_flag) + factor(ima_flag), data=data_hj)
summary(fit)

fit = lm(pl_orbsmax ~ 
           pl_bmassj +
           pl_radj +
           pl_orbeccen +
           pl_orbincl +
           # factor(sy_snum) +  #5 632.6967
           # factor(sy_pnum) +  #1 639.7964
           # I(sy_vmag - sy_kmag) + #4 634.2594
           # st_mass + #3 635.9599
           st_age +
           # st_teff + #2 637.7965
           st_rotp +
           st_met +
           # st_rad + #7 631.2146
           factor(tran_flag) +
           factor(ima_flag),
           # factor(rv_flag), #6 631.4155
           data=data_hj)
AIC(fit)

# best linear model from the starting set of params
fit = lm(pl_orbsmax ~ 
           pl_bmassj +
           pl_radj +
           pl_orbeccen +
           pl_orbincl +
           st_age +
           st_rotp +
           st_met +
           factor(tran_flag) +
           factor(ima_flag),
         data=data_hj)

summary(fit)
# most significant parameters for best linear fit are
# - planetary mass
# - orbital inclination
# - stellar rotation period
# - transit vs not

AIC(fit)

plot(fit)
