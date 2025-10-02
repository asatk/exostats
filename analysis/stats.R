library(leaps)
library(mgcv)

data = read.csv("tables-merged/alfven_data.csv")
#df = subset(data, select=c(pl_orbsmax, pl_orbeccen, pl_rade, pl_bmasse, st_rad,
#                           Ro, sy_dist, st_mass, sy_vmag, sy_kmag, Prot,
#                           VK_color, st_teff, st_age, Tauc, RA, ASHC, OHZ))
df = subset(data, select=c(pl_orbsmax, pl_orbeccen, pl_rade, pl_bmasse, st_rad,
                           sy_dist, st_mass, sy_vmag, sy_kmag, Prot, VK_color,
                           st_teff, st_age))

fit = lm(Prot ~ ., data=df)
plot(fit)
summary(fit)

#df_sub = subset(df, select=c(st_rad, sy_dist, st_mass, sy_vmag, sy_kmag, st_age, Prot))
df_sub = subset(df, Prot < 50, select=c(st_rad, st_mass, VK_color, st_age, Prot))
fit_sub = lm(Prot ~ ., data=df_sub)
summary(fit_sub)
plot(fit_sub)
coefs = coef(fit_sub)

vstr = "st_age"
y = df_sub[,"Prot"]
x = df_sub[,vstr]
m = coefs[vstr]
plot(x, y)
xline = sort(fit_sub$model[,vstr])
yline = fitted(fit_sub)[order(fit_sub$model[,vstr])]
lines(xline, yline, col="red")

plot(df_sub)

# radius-mass
ystr = "Prot"
xstr = "VK_color"
#y = df_sub[,ystr]
y = log(df_sub[,ystr])
#x = df_sub[,xstr]
x = log(df_sub[,xstr])
fit_slr = lm(y ~ x)
fit_slr = lm(y ~ poly(x, 6))
fit_slr = gam(y ~ s(x))
summary(fit_slr)
coef(fit_slr)
plot(fit_slr)

plot(x, y)
xline = sort(x)
yline = fitted(fit_slr)[order(x)]
lines(xline, yline, col="red")

# identify bad lev pts
n = nrow(df_sub)
lev = 1 / n + (x - mean(x))^2 / sum((x - mean(x))^2)
where_lev = lev > 4 / n
where_out = abs(residuals(fit_slr) / sigma(fit_slr) / sqrt(1 - lev)) > 2
where_bad = where_lev & where_out
nrow(df_sub[where_bad,])

# re-fit after removing outliers
#y = df_sub[!where_bad,ystr]
y = log(df_sub[!where_bad,ystr])
#x = df_sub[!where_bad,xstr]
x = log(df_sub[!where_bad,xstr])
fit_slr = lm(y ~ x)
fit_slr = lm(y ~ poly(x, 6))
fit_slr = gam(y ~ s(x))
summary(fit_slr)
coef(fit_slr)
plot(fit_slr)

plot(x, y)
xline = sort(x)
yline = fitted(fit_slr)[order(x)]
lines(xline, yline, col="red")

